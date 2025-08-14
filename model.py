"""
TFT Model Training Script
-------------------------
This script trains a Temporal Fusion Transformer (TFT) model using simulated daily sales data.
It demonstrates handling of static and time-varying covariates, dataset preparation with
TimeSeriesDataSet, and training with PyTorch Lightning. The best model checkpoint is saved automatically.
"""

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


# Define Lightning Module wrapper
class TftLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for TemporalFusionTransformer
    Handles training, validation, and optimizer configuration.
    """

    def __init__(self, model: TemporalFusionTransformer, learning_rate: float = 0.03):
        super().__init__()
        self.model = model
        self.model.cuda(0)  # Move model to GPU if available
        self.learning_rate = learning_rate
        self.loss = model.loss

    def forward(self, x):
        """Forward pass through TFT model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step logic."""
        x, y = batch
        output = self.model(x)["prediction"]
        loss = self.loss(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step logic."""
        x, y = batch
        output = self.model(x)["prediction"]
        loss = self.loss(output, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure Adam optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
# Load and preprocess data
DATA_PATH = "data/simulated_innovatemart_daily_sales.csv"

# Define dataset parameters
MAX_ENCODER_LENGTH = 30
MAX_PREDICTION_LENGTH = 7  # Change to 1 for 1-day forecast
TRAIN_RATIO = 0.8
BATCH_SIZE = 512
MAX_EPOCHS = 10


# Load data
data = pd.read_csv(DATA_PATH)

# Convert date column to datetime type
data["Date"] = pd.to_datetime(data["Date"])

# Sort by store and date
data = data.sort_values(["store_id", "Date"])

# Create time index for model
data["time_idx"] = (data["Date"] - data["Date"].min()).dt.days

# Convert categorical columns to string type
categorical_cols = [
    "store_id", "store_size", "day_of_week", "month", "is_weekend",
    "promotion_active", "holiday_flag", "school_holiday_flag",
    "competitor_opened_flag"
]
for col in categorical_cols:
    data[col] = data[col].astype(str)



# Determine training cutoff index
training_cutoff = int(len(data) * TRAIN_RATIO)

# Create TimeSeriesDataSet for training
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="daily_sales",
    group_ids=["store_id"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    time_varying_known_categoricals=[
        "day_of_week", "month", "is_weekend", "promotion_active",
        "holiday_flag", "school_holiday_flag", "competitor_opened_flag"
    ],
    time_varying_known_reals=["time_idx", "competitor_impact"],
    time_varying_unknown_reals=["daily_sales"],
    static_categoricals=["store_id", "store_size"],
    static_reals=["city_population", "base_mean", "annual_growth_rate", "competitor_impact"],
    target_normalizer=GroupNormalizer(groups=["store_id"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Create validation dataset from training dataset
validation = TimeSeriesDataSet.from_dataset(
    training,
    data,
    predict=True,
    stop_randomization=True
)

# Create dataloaders
train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE)

# Initialize TFT model
tft_model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=MAX_PREDICTION_LENGTH, 
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)


# Training setup
tft_lightning = TftLightning(tft_model, learning_rate=0.03)

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="best-model-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=False,
)

trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices=1,
    callbacks=[checkpoint_callback]
)

# Train the model
trainer.fit(tft_lightning, train_dataloader, val_dataloader)

# Load best model
best_model_path = checkpoint_callback.best_model_path
best_lightning_model = TftLightning.load_from_checkpoint(
    best_model_path,
    model=tft_model,
    learning_rate=0.03
)