import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
import numpy as np

from prophet import Prophet
import logging
from river import drift

from alibi_detect.datasets import fetch_nab

matplotlib.rcParams.update({'font.size': 14})

# ---- CPU and RAM ----

df_cpu = pd.read_csv("./results/Prophet_cpu_usage.csv")
df_ram = pd.read_csv("./results/Prophet_ram_usage.csv")

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df_cpu.value, label = "CPU")
axs.legend(fontsize=20)
plt.yticks(fontsize=10)
plt.xticks([])
plt.ylabel('CPU Usage [%]')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/Prophet_cpu_usage.png') 

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df_ram.value, label = "RAM")
axs.legend(fontsize=20)
plt.yticks(fontsize=10)
plt.xticks([])
plt.ylabel('RAM Usage [%]')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/Prophet_ram_usage.png') 

# ---- Forecast and actual data ---- 

def train_model(df_prophet, training_set_size):
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    
    drifts = []
    adwin = drift.ADWIN(delta=0.001,
                    max_buckets=10,
                    grace_period=10,
                    min_window_length=10,
                    clock=20
                   )
    
    initial_training_set_size = training_set_size

    train = df_prophet[:initial_training_set_size]
    remainder = df_prophet[initial_training_set_size:]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(train)
    
    final_forecast = pd.DataFrame(index=df_prophet.index, columns=["yhat", "yhat_lower", "yhat_upper"])
    
    future = model.make_future_dataframe(periods=len(remainder), freq='W')
    forecast = model.predict(future)
    final_forecast.iloc[:len(forecast), :] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].values
    
    for i in range(len(remainder)):
        val = remainder['y'].iloc[i]
        adwin.update(val)
        
        if adwin.drift_detected:
            drift_point = initial_training_set_size + i
            drifts.append(drift_point)
            #print(f"Change detected at index {drift_point}, input value: {val}")
            
            train_start = max(0, drift_point - training_set_size)
            train_end = min(len(df_prophet), drift_point)
            train = df_prophet[train_start:train_end]
            model = Prophet()
            model.fit(train)
            
            forecastlen = len(df_prophet) - drift_point
            future = model.make_future_dataframe(periods=forecastlen, freq='W')
            forecast = model.predict(future)
            
            forecast_length = min(len(forecast), len(final_forecast) - drift_point)
            final_forecast.iloc[drift_point:drift_point + forecast_length, :] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].iloc[:forecast_length].values
    
    if not drifts or drifts[-1] != len(df_prophet):
        drift_point = drifts[-1] if drifts else initial_training_set_size
        forecastlen = len(df_prophet) - drift_point
        future = model.make_future_dataframe(periods=forecastlen, freq='W')
        forecast = model.predict(future)
        forecast_length = min(len(forecast), len(final_forecast) - drift_point)
        final_forecast.iloc[drift_point:drift_point + forecast_length, :] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].iloc[:forecast_length].values

    final_forecast = final_forecast.apply(pd.to_numeric, errors='coerce')
    final_forecast = final_forecast.dropna()

    if not pd.api.types.is_datetime64_any_dtype(final_forecast.index):
        final_forecast.index = pd.to_datetime(final_forecast.index)


    return final_forecast[800:]
    

df = pd.read_csv("../../melbourne_weekly_anomalies.csv")
df_prophet = df.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})

forecast = train_model(df_prophet, 800)
predictions = forecast["yhat"]

ground_truth_values = np.array(df_prophet.y[800:])

errors = np.abs(ground_truth_values - predictions)
squared_errors = np.square(errors)

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df.index[801:], df_prophet.y[801:], label = "Temperature")
plt.plot(df.index[800:], predictions, label = "Prophet Forecast", color='red')
axs.legend(fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Temperature [°C]')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/Prophet_forecast_and_true_data.png') 

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
plt.plot(df.index[800:], errors, label = "Squared Error", color='orange')
axs.legend(fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
axs.set_ylim(0.0, 150.0) 
plt.ylabel('Squared Error')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/Prophet_error.png') 


fig, axs = plt.subplots(1, 1, figsize=(15, 6))
sns.kdeplot(errors, ax=axs, fill=True)
axs.set_xlabel('Temperature [°C]')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('./plots/Prophet_error_distribution.png') 


