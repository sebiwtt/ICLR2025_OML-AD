import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
import numpy as np

import statsmodels.api as sm

from alibi_detect.datasets import fetch_nab

matplotlib.rcParams.update({'font.size': 14})

# ---- CPU and RAM ----

df_cpu = pd.read_csv("./results/ARIMA_cpu_usage.csv")
df_ram = pd.read_csv("./results/ARIMA_ram_usage.csv")

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df_cpu.value, label = "CPU")
axs.legend(fontsize=20)
plt.yticks(fontsize=10)
plt.xticks([])
plt.ylabel('CPU Usage [%]')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/ARIMA_cpu_usage.png') 

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df_ram.value, label = "RAM")
axs.legend(fontsize=20)
plt.yticks(fontsize=10)
plt.xticks([])
plt.ylabel('RAM Usage [%]')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/ARIMA_ram_usage.png') 

# ---- Forecast and actual data ---- 


def scheduled_retraining_sarima(df, window_size):

    forecast_horizon = window_size
    
    full_forecast = []
    full_upper_conf_int = [] 
    full_lower_conf_int = []
    train_end = window_size 

    while train_end <= len(df):
       
        train_data = df.value[train_end-window_size:train_end]

        (p, d, q) = (1, 0, 1)
        (P, D, Q, s) = (1, 1, 1, 12)

        sarima_model = sm.tsa.SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
        model = sarima_model.fit(disp=False)

        forecast_horizon = min(forecast_horizon, len(df) - train_end)

        if forecast_horizon <= 0:
            break

        forecast = model.get_forecast(steps=forecast_horizon)
        forecast_values = forecast.predicted_mean
        confidence_intervals = forecast.conf_int(alpha=0.05)
      
        full_forecast.extend(forecast_values)
        full_lower_conf_int.extend(confidence_intervals['lower value'])
        full_upper_conf_int.extend(confidence_intervals['upper value'])

        train_end += forecast_horizon

        if train_end > len(df):
            break

    full_forecast = np.array(full_forecast)
    full_upper_conf_int = np.array(full_upper_conf_int)
    full_lower_conf_int = np.array(full_lower_conf_int)

    return full_forecast, full_upper_conf_int, full_lower_conf_int


df = fetch_nab("realAWSCloudwatch/rds_cpu_utilization_e47b3b")
rawdata = df["data"].value

forecast_values, upper_confidence_intervals, lower_confidence_intervals  = scheduled_retraining_sarima(df["data"], 800)

predictions = forecast_values

ground_truth_values = df["data"].value[800:]

errors = np.abs(np.array(ground_truth_values) - np.array(predictions))

squared_errors = np.square(errors)

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df["data"].index[800:], ground_truth_values, label = "rds_cpu_utilization_e47b3b")
plt.plot(df["data"].index[800:], predictions, label = "ARIMA Forecast", color='red')
axs.legend(fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('%')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/ARIMA_forecast_and_true_data.png') 

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
plt.plot(df["data"].index[800:], errors, label = "Squared Error", color='orange')
axs.legend(fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
axs.set_ylim(0.0, 150.0) 
plt.ylabel('Squared Error')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/ARIMA_error.png') 


fig, axs = plt.subplots(1, 1, figsize=(15, 6))
sns.kdeplot(errors, ax=axs, fill=True)
axs.set_xlabel('Traffic Value (Gbits)')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('./plots/ARIMA_error_distribution.png') 


