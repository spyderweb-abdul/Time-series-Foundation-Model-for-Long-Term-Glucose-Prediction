import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.metrics import mape, rmse
from darts.dataprocessing.transformers import Scaler

# Load data
df = pd.read_csv("glucose_netiob_data.csv")
df['LBDTC'] = pd.to_datetime(df['LBDTC'])
df = df[df['USUBJID'] == '1014']
df.set_index('LBDTC', inplace=True)
df = df[['LBORRES', 'NETIOB']].resample('15min').mean().interpolate()

# Create TimeSeries objects
target_series = TimeSeries.from_dataframe(df, value_cols="LBORRES")
covariates = TimeSeries.from_dataframe(df, value_cols="NETIOB")

# Scale the series
scaler = Scaler()
target_scaled = scaler.fit_transform(target_series)

# Split
context_window = 1024
prediction_window = 96
train_cutoff = len(target_scaled) - prediction_window

target_train = target_scaled[:train_cutoff]
target_val = target_scaled[train_cutoff - context_window:]  # to preserve context

def evaluate_and_plot(forecast, model_name):
    forecast_unscaled = scaler.inverse_transform(forecast)
    actual_unscaled = scaler.inverse_transform(target_val[-prediction_window:])

    print(f"{model_name} - MAPE: {mape(actual_unscaled, forecast_unscaled):.2f}")
    print(f"{model_name} - RMSE: {rmse(actual_unscaled, forecast_unscaled):.2f}")

    plt.figure(figsize=(12, 6))
    target_scaled[-context_window:].plot(label="Context (Past)", lw=2)
    forecast_unscaled.plot(label=f"{model_name} Forecast", lw=2)
    actual_unscaled.plot(label="Actual Future", lw=2)
    plt.axvline(x=target_scaled.time_index()[-prediction_window], color='red', linestyle='--', label='Forecast Start')
    plt.title(f"Glucose Forecast using {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
### TimesNet

from darts.models import TimesNet

model_timesnet = TimesNet(
    input_chunk_length=context_window,
    output_chunk_length=prediction_window,
    n_epochs=100,
    batch_size=32,
    random_state=42,
    model_name="timesnet_glucose",
    save_checkpoints=True,
    force_reset=True
)

model_timesnet.fit(series=target_train, verbose=True)
forecast_timesnet = model_timesnet.predict(n=prediction_window)
evaluate_and_plot(forecast_timesnet, "TimesNet")

### Informer

from darts.models import InformerModel

model_informer = InformerModel(
    input_chunk_length=context_window,
    output_chunk_length=prediction_window,
    n_epochs=100,
    batch_size=32,
    model_name="informer_glucose",
    force_reset=True,
    save_checkpoints=True,
    random_state=42
)

model_informer.fit(series=target_train, verbose=True)
forecast_informer = model_informer.predict(n=prediction_window)
evaluate_and_plot(forecast_informer, "Informer")


### PatchTST

from darts.models import PatchTST

model_patchtst = PatchTST(
    input_chunk_length=context_window,
    output_chunk_length=prediction_window,
    n_epochs=100,
    batch_size=32,
    model_name="patchtst_glucose",
    force_reset=True,
    save_checkpoints=True,
    random_state=42
)

model_patchtst.fit(series=target_train, verbose=True)
forecast_patchtst = model_patchtst.predict(n=prediction_window)
evaluate_and_plot(forecast_patchtst, "PatchTST")

