# train_model.py
"""
TFT Model Training Script (All Stores, Percent Split)
====================================================
This script trains a Temporal Fusion Transformer (TFT) using simulated InnovateMart
daily sales data. It:
  - Preprocesses data and assigns covariates to correct categories,
  - Splits data by time using a percent cutoff (default 80/20),
  - Trains a single global model across ALL stores (store_id as a categorical),
  - Uses QuantileLoss with multiple quantiles (probabilistic forecasts),
  - Saves the best checkpoint to disk.

Run:
    python train_model.py --data_path data/simulated_innovatemart_daily_sales.csv \
                          --save_dir checkpoints \
                          --encoder_len 30 \
                          --pred_len 7 \
                          --train_ratio 0.8 \
                          --batch_size 512 \
                          --max_epochs 10
"""


import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


def set_seed(seed: int = 42) -> None:
    """
    Set common random seeds for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Seed value for RNGs, by default 42.
    """
    pl.seed_everything(seed, workers=True)


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load the simulated CSV and perform basic preprocessing:
      - Parse dates
      - Sort within each store by time
      - Create a monotonic `time_idx`
      - Ensure categoricals are strings

    Parameters
    ----------
    csv_path : str
        Path to the input simulated CSV.

    Returns
    -------
    pd.DataFrame
        Preprocessed long-format dataframe.
    """
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["store_id", "Date"]).reset_index(drop=True)

    # Monotonic time index across the whole dataset (OK for global model)
    df["time_idx"] = (df["Date"] - df["Date"].min()).dt.days

    # Cast categoricals to string (required by pytorch-forecasting)
    cat_cols = [
        "store_id",
        "store_size",
        "day_of_week",
        "month",
        "is_weekend",
        "promotion_active",
        "holiday_flag",
        "school_holiday_flag",
        "competitor_opened_flag",
    ]
    for c in cat_cols:
        df[c] = df[c].astype(str)

    return df


def build_datasets(
    df: pd.DataFrame,
    encoder_len: int,
    pred_len: int,
    train_ratio: float,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Build TimeSeriesDataSet objects for training and validation.

    Notes
    -----
    - We keep a single global model across all stores by using `group_ids=["store_id"]`.
    - Split is performed by a global time cutoff (percent of rows), preserving chronology.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe with required columns and `time_idx`.
    encoder_len : int
        Max encoder length (lookback window).
    pred_len : int
        Max prediction length (forecast horizon).
    train_ratio : float
        Percentage of rows used for training, e.g., 0.8 for 80%.

    Returns
    -------
    (training_ds, validation_ds) : Tuple[TimeSeriesDataSet, TimeSeriesDataSet]
    """
    # Global chronological cutoff
    cutoff = int(len(df) * train_ratio)

    training_ds = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= cutoff],
        time_idx="time_idx",
        target="daily_sales",
        group_ids=["store_id"],
        max_encoder_length=encoder_len,
        max_prediction_length=pred_len,
        # Known categoricals that are aligned to the date index
        time_varying_known_categoricals=[
            "day_of_week",
            "month",
            "is_weekend",
            "promotion_active",
            "holiday_flag",
            "school_holiday_flag",
            "competitor_opened_flag",
        ],
        # Known real-valued covariates over time
        time_varying_known_reals=[
            "time_idx",
            "competitor_impact",
        ],
        # Unknown real-valued covariates (contain the target)
        time_varying_unknown_reals=["daily_sales"],
        # Store-level static metadata
        static_categoricals=["store_id", "store_size"],
        static_reals=["city_population", "base_mean", "annual_growth_rate", "competitor_impact"],
        # Normalize target per store
        target_normalizer=GroupNormalizer(groups=["store_id"]),
        # Useful auto-features
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Validation dataset mirrors training configuration, but uses full df
    validation_ds = TimeSeriesDataSet.from_dataset(
        training_ds, df, predict=True, stop_randomization=True
    )

    return training_ds, validation_ds


class TftLightning(pl.LightningModule):
    """
    Lightning wrapper around a TemporalFusionTransformer model.

    This wrapper delegates forward and loss computation to the underlying
    TFT model while handling optimizer configuration and training/validation logs.
    """

    def __init__(self, model: TemporalFusionTransformer, learning_rate: float = 3e-2) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])  # keep LR, avoid saving `model` twice
        self.model = model.cuda(0) if torch.cuda.is_available()  else  model   # Move model to GPU if available
        self.loss = model.loss  # QuantileLoss (or any loss passed to model)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass via underlying TFT model."""
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """One training step with loss logging."""
        x, y = batch
        y_hat = self.model(x)["prediction"]
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """One validation step with loss logging."""
        x, y = batch
        y_hat = self.model(x)["prediction"]
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Adam optimizer with the requested learning rate."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def build_model(
    training_ds: TimeSeriesDataSet,
    encoder_len: int,
    pred_len: int,
    quantiles: List[float],
) -> TemporalFusionTransformer:
    """
    Instantiate a TFT model from a dataset with QuantileLoss.

    Important:
      - With QuantileLoss, `output_size` MUST equal the number of quantiles.
      - The forecast horizon is controlled by `max_prediction_length` in the dataset,
        not by `output_size`.

    Parameters
    ----------
    training_ds : TimeSeriesDataSet
        Prepared training dataset.
    encoder_len : int
        Max encoder length (for logging/consistency—model is built from `training_ds` anyway).
    pred_len : int
        Max prediction length (for logging/consistency).
    quantiles : List[float]
        List of quantiles e.g. [0.1, 0.5, 0.9].

    Returns
    -------
    TemporalFusionTransformer
        Configured TFT model ready for Lightning.
    """
    model = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=3e-2,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=pred_len,           # **number of quantiles**
        loss=QuantileLoss(quantiles=quantiles),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    return model


def save_run_artifacts(
    save_dir: Path,
    best_ckpt_path: str,
    dataset_meta: Dict,
) -> None:
    """
    Save lightweight run artifacts helpful for later inference.

    Parameters
    ----------
    save_dir : Path
        Base directory where checkpoints/configs are stored.
    best_ckpt_path : str
        Path to the best model checkpoint file.
    dataset_meta : Dict
        Small JSON payload with dataset/model settings for reproducible loading.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    # 1) Save best path in a small text file for downstream scripts
    (save_dir / "BEST_CHECKPOINT.txt").write_text(best_ckpt_path)

    # 2) Save dataset/model meta
    with open(save_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=2)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a TFT model on InnovateMart simulated sales")
    parser.add_argument("--data_path", type=str, default="data/simulated_innovatemart_daily_sales.csv", help="Path to simulated CSV.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints/configs.")
    parser.add_argument("--encoder_len", type=int, default=30, help="Max encoder (lookback) length.")
    parser.add_argument("--pred_len", type=int, default=7, help="Forecast horizon (max prediction length).")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Percent of rows for training (0-1).")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")
    parser.add_argument("--max_epochs", type=int, default=10, help="Max training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--gpus", type=int, default=1, help="#GPUs to use if available (0 for CPU).")
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.5, 0.9], help="Quantiles for QuantileLoss.")
    return parser.parse_args()


def main() -> None:
    """
    Orchestrate training: load data, build datasets/model, train, and save artifacts.
    """
    args = parse_args()
    set_seed(args.seed)

    data_path = Path(args.data_path)
    save_dir = Path(args.save_dir)

    # ---------------- Load & prepare data ----------------
    df = load_and_prepare_data(str(data_path))

    # ---------------- Build datasets ----------------
    training_ds, validation_ds = build_datasets(df, args.encoder_len, args.pred_len, args.train_ratio)

    # ---------------- Dataloaders ----------------
    train_loader = training_ds.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_loader = validation_ds.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)

    # ---------------- Build model (Quantile) ----------------
    tft = build_model(training_ds, args.encoder_len, args.pred_len, args.quantiles)

    # ---------------- Lightning module ----------------
    lit = TftLightning(model=tft, learning_rate=3e-2)

    # ---------------- Checkpointing ----------------
    ckpt_cb = ModelCheckpoint(
        dirpath=str(save_dir),
        filename="best-model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=False,
        verbose=True,
    )

    # ---------------- Trainer ----------------
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.gpus if torch.cuda.is_available() and args.gpus > 0 else 1,
        callbacks=[ckpt_cb],
        log_every_n_steps=10,
    )

    # ---------------- Fit ----------------
    trainer.fit(lit, train_loader, val_loader)

    # ---------------- Save artifacts ----------------
    best_ckpt = ckpt_cb.best_model_path
    print(f"\nBest model saved at: {best_ckpt}")

    dataset_meta = {
        "encoder_len": args.encoder_len,
        "pred_len": args.pred_len,
        "train_ratio": args.train_ratio,
        "categoricals": {
            "static": ["store_id", "store_size"],
            "time_varying_known": [
                "day_of_week",
                "month",
                "is_weekend",
                "promotion_active",
                "holiday_flag",
                "school_holiday_flag",
                "competitor_opened_flag",
            ],
        },
        "reals": {
            "static": ["city_population", "base_mean", "annual_growth_rate", "competitor_impact"],
            "time_varying_known": ["time_idx", "competitor_impact"],
            "time_varying_unknown": ["daily_sales"],
        },
        "quantiles": args.quantiles,
        "target": "daily_sales",
        "group_ids": ["store_id"],
    }
    save_run_artifacts(save_dir, best_ckpt, dataset_meta)


if __name__ == "__main__":
    main()
