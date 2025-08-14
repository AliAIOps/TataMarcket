"""
Streamlit 7-Day Sales Forecast Dashboard with Model Interpretability
-------------------------------------------------------------------
This app displays historical sales for each InnovateMart store, predicts
the next 7 days using a trained Temporal Fusion Transformer (TFT) model,
and shows a variable importance plot for interpretability.

Features:
- Load best model checkpoint automatically from BEST_CHECKPOINT.txt
- Display historical sales (all available days) for selected store
- Show 7-day sales forecast alongside actual data
- Show variable importance plot
"""

import streamlit as st
import pandas as pd
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer
import plotly.graph_objects as go
import os



# ------------------ Load Checkpoint Path ------------------
def load_checkpoint_path(default_file="checkpoints/BEST_CHECKPOINT.txt"):
    """
    Load model checkpoint path from file. If file missing or empty, return None.
    """
    if os.path.exists(default_file):
        with open(default_file, "r") as f:
            path = f.read().strip()
        if path:
            return path
    return None

# ------------------ Prediction Function ------------------
def predict_store(store_id: str, training_ds: TimeSeriesDataSet, full_df: pd.DataFrame):
    """
    Predicts the next PRED_HORIZON days for a given store_id.
    Returns a DataFrame with Date and predicted sales.
    """
    store_mask = full_df["store_id"] == store_id
    store_val_ds = TimeSeriesDataSet.from_dataset(
        training_ds,
        full_df[store_mask],
        predict=True,
        stop_randomization=True
    )
    val_loader = store_val_ds.to_dataloader(train=False, batch_size=512)

    preds, dec_dates = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = {k: v.to(DEVICE) for k, v in x.items()}
            out = best_model(x)["prediction"].detach().cpu().numpy()  # [B,H,1]
            out = out[..., 0]  # squeeze last dim -> [B,H]
            B, H = out.shape

            # decoder time_idx
            last_enc = x["decoder_time_idx"][:, -1].cpu().numpy()[:, None]  # [B,1]
            dec_tidx = last_enc + np.arange(1, H+1)  # [B,H]

            preds.append(out.reshape(-1))
            dec_dates.append(dec_tidx.reshape(-1))

    preds = np.concatenate(preds)
    dec_tidx = np.concatenate(dec_dates)
    pred_dates = pd.to_datetime(dec_tidx.astype(int), unit="D", origin=origin_date)

    return pd.DataFrame({"Date": pred_dates, "y_pred": preds})



# ------------------ Lightning Wrapper ------------------
class TftLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for TemporalFusionTransformer"""
    def __init__(self, model, learning_rate=0.03):
        super().__init__()
        self.model = model.to(DEVICE)
        self.learning_rate = learning_rate
        self.loss = model.loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = {k: v.to(DEVICE) for k, v in x.items()}
        y = y.to(DEVICE)
        output = self.model(x)["prediction"]
        loss = self.loss(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = {k: v.to(DEVICE) for k, v in x.items()}
        y = y.to(DEVICE)
        output = self.model(x)["prediction"]
        loss = self.loss(output, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    

# ------------------ Prediction Function ------------------
def predict_validation(store_id):
    """
    Predicts on validation dataset for the selected store_id
    Returns DataFrame with Date, actual sales, and predicted sales
    """
    store_mask = data["store_id"] == store_id
    store_val_ds = TimeSeriesDataSet.from_dataset(
        training,
        data[store_mask],
        predict=True,
        stop_randomization=True
    )
    val_loader = store_val_ds.to_dataloader(train=False, batch_size=512)

    preds = []
    with torch.no_grad():
        for x, y in val_loader:
            x = {k: v.to(DEVICE) for k, v in x.items()}
            out = best_model(x)["prediction"].detach().cpu().numpy()[..., 0]  # [B,H]
            for batch_pred in out:
                preds.extend(batch_pred)

    # مقادیر واقعی validation
    val_dates = data[store_mask]["Date"].values[-len(preds):]
    actuals = data[store_mask]["daily_sales"].values[-len(preds):]

    return pd.DataFrame({"Date": val_dates, "Actual": actuals, "Predicted": preds})








# ------------------ Config ------------------
DATA_PATH = "data/simulated_innovatemart_daily_sales.csv"
BEST_CHECKPOINT_FILE = "checkpoints/BEST_CHECKPOINT.txt"
MAX_ENCODER_LENGTH = 30
PRED_HORIZON = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {DEVICE}")

# ------------------ Load Data ------------------
data = pd.read_csv(DATA_PATH)
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["store_id", "Date"])
data["time_idx"] = (data["Date"] - data["Date"].min()).dt.days

# categorical columns as string
categorical_cols = ["store_id", "store_size", "day_of_week", "month", "is_weekend",
                    "promotion_active", "holiday_flag", "school_holiday_flag",
                    "competitor_opened_flag"]
for col in categorical_cols:
    data[col] = data[col].astype(str)

origin_date = data["Date"].min()  # for reconstructing prediction dates

# ------------------ TimeSeriesDataSet ------------------
training_cutoff = int(len(data) * 0.8)
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="daily_sales",
    group_ids=["store_id"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=PRED_HORIZON,
    time_varying_known_categoricals=[
        "day_of_week","month","is_weekend","promotion_active",
        "holiday_flag","school_holiday_flag","competitor_opened_flag"
    ],
    time_varying_known_reals=["time_idx","competitor_impact"],
    time_varying_unknown_reals=["daily_sales"],
    static_categoricals=["store_id","store_size"],
    static_reals=["city_population","base_mean","annual_growth_rate","competitor_impact"],
    target_normalizer=GroupNormalizer(groups=["store_id"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# ------------------ Load Model ------------------
checkpoint_path = load_checkpoint_path(BEST_CHECKPOINT_FILE)
checkpoint_path = st.text_input(
    "Checkpoint Path (override if needed):",
    value=checkpoint_path if checkpoint_path else ""
)
if not checkpoint_path or not os.path.exists(checkpoint_path):
    st.warning("Valid checkpoint path required!")
    st.stop()


tft_model = TemporalFusionTransformer.from_dataset(training)
best_model = TftLightning.load_from_checkpoint(checkpoint_path, model=tft_model).to(DEVICE)
best_model.eval()


# ------------------ Streamlit UI ------------------
st.title("7-Day Sales Forecast Dashboard (Validation)")

store_ids = data["store_id"].unique()
selected_store = st.selectbox("Select store_id", store_ids)

store_data = data[data["store_id"] == selected_store]

# ------------------ Get Predictions ------------------
pred_val_df = predict_validation(selected_store)

# ------------------ Plot Historical + Predicted ------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=store_data["Date"],
    y=store_data["daily_sales"],
    mode="lines+markers",
    name="Historical Sales"
))

fig.add_trace(go.Scatter(
    x=pred_val_df["Date"],
    y=pred_val_df["Predicted"],
    mode="lines+markers",
    name="Predicted Sales"
))

st.plotly_chart(fig, use_container_width=True)
