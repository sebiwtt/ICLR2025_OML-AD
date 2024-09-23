import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
import numpy as np

from prophet import Prophet
import logging

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

def train_model(df_prophet):
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    
    window_size = 200  # Number of data points to train on and forecast
    total_data_points = len(df_prophet)
    
    # Initialize the final forecast DataFrame
    final_forecast = pd.DataFrame(index=df_prophet.index, columns=["yhat", "yhat_lower", "yhat_upper"])
    
    # Start the process
    start = 0
    
    while start + window_size < total_data_points:
        # Train on the current `window_size` data points
        train = df_prophet[start:start + window_size]
        
        model = Prophet()
        model.fit(train)
        
        # Forecast the next `window_size` data points, excluding the training period
        future = model.make_future_dataframe(periods=window_size, freq='D')
        forecast = model.predict(future)
        
        # Only include the forecasts that are outside of the training data
        forecast = forecast.iloc[window_size:]  # Skip the training period's "forecast"
        
        # Update the final forecast DataFrame
        forecast_length = min(len(forecast), total_data_points - (start + window_size))
        final_forecast.iloc[start + window_size:start + window_size + forecast_length, :] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].iloc[:forecast_length].values
        
        # Move to the next window
        start += window_size
    
    # If there are remaining data points after the last full window, handle them
    if start < total_data_points:
        remaining_train = df_prophet[start:]
        
        model = Prophet()
        model.fit(remaining_train)
        
        future = model.make_future_dataframe(periods=total_data_points - start, freq='D')
        forecast = model.predict(future)
        
        # Only include forecasts for the future, not the training data
        forecast = forecast.iloc[len(remaining_train):]
        
        forecast_length = min(total_data_points - start, len(forecast))
        final_forecast.iloc[start:start + forecast_length, :] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].iloc[:forecast_length].values

        final_forecast = final_forecast.apply(pd.to_numeric, errors='coerce')
        final_forecast = final_forecast.dropna()
        
        if not pd.api.types.is_datetime64_any_dtype(final_forecast.index):
            final_forecast.index = pd.to_datetime(final_forecast.index)
    
    return final_forecast
    

df = pd.read_csv("../../synth.csv")
df_prophet = df.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})

forecast = train_model(df_prophet)
predictions = forecast["yhat"]

ground_truth_values = np.array(df_prophet.y[200:])

errors = np.abs(ground_truth_values - predictions)
squared_errors = np.square(errors)

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df.index[201:], df_prophet.y[201:], label = "Temperature")
plt.plot(df.index[200:], predictions, label = "Prophet Forecast", color='red')
axs.legend(fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('y')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/Prophet_forecast_and_true_data.png') 

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
plt.plot(df.index[200:], errors, label = "Squared Error", color='orange')
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
axs.set_xlabel('y')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('./plots/Prophet_error_distribution.png') 


