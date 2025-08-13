import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# --- خواندن و پیش‌پردازش داده‌ها ---
data = pd.read_csv("data/simulated_innovatemart_daily_sales.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["store_id", "Date"])
data["time_idx"] = (data["Date"] - data["Date"].min()).dt.days
for col in ["store_id", "store_size", "day_of_week", "month", "is_weekend",
            "promotion_active", "holiday_flag", "school_holiday_flag",
            "competitor_opened_flag"]:
    data[col] = data[col].astype(str)

max_encoder_length = 30
max_prediction_length = 7
training_cutoff = int(len(data) * 0.8)

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="daily_sales",
    group_ids=["store_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
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

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
batch_size = 512
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size)

# --- مدل TFT ---
tft_model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=max_prediction_length,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)
######################
class TftLightning(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.model.cuda(0)
        self.learning_rate = learning_rate
        self.loss = model.loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)['prediction']
        loss = self.loss(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)['prediction']
        loss = self.loss(output, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# --- ایجاد wrapper و Trainer ---
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
    max_epochs=2,
    accelerator="auto",
    devices=1,
    callbacks=[checkpoint_callback]
)

trainer.fit(tft_lightning, train_dataloader, val_dataloader)

# --- لود بهترین مدل wrapper ---
best_model_path = checkpoint_callback.best_model_path
best_lightning_model = TftLightning.load_from_checkpoint(
    best_model_path,
    model=tft_model,
    learning_rate=0.03
)