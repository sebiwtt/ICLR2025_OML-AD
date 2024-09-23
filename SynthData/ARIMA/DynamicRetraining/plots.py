import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
import numpy as np

import statsmodels.api as sm
from river import drift

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


def train_sarima_model(train_data, p, d, q, P, D, Q, s):
    sarima_model = sm.tsa.SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
    model = sarima_model.fit(disp=False)
    return model


def dynamic_retraining_with_drift_detection(df, initial_train_size=200):
    values = df.value
    adwin = drift.ADWIN(delta=0.001,
                    max_buckets=10,
                    grace_period=10,
                    min_window_length=10,
                    clock=20
                   )
    
    drifts = []
    all_forecasts = pd.DataFrame(index=df.index, columns=["forecast", "lower_bound", "upper_bound"])

    train_start = 0
    train_end = initial_train_size

    train_data = values[train_start:train_end]

    (p, d, q) = (2, 0, 2)
    (P, D, Q, s) = (2, 1, 2, 12)

    sarima_model = sm.tsa.SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
    model = sarima_model.fit(disp=False)

    last_drift_point = train_end

    i = train_end
    while i < len(values):
        adwin.update(values[i])
        if adwin.drift_detected:
            print(f"Change detected at index {i}, input value: {values[i]}")
            drifts.append(i)

            forecast_steps = i - last_drift_point
            if forecast_steps > 0:
                forecast_obj = model.get_forecast(steps=forecast_steps)
                forecast = forecast_obj.predicted_mean
                conf_int = forecast_obj.conf_int()
                all_forecasts.iloc[last_drift_point:i, 0] = forecast.values
                all_forecasts.iloc[last_drift_point:i, 1] = conf_int.iloc[:, 0].values
                all_forecasts.iloc[last_drift_point:i, 2] = conf_int.iloc[:, 1].values

            last_drift_point = i

      
            train_end = i
            train_start = max(0, train_end - initial_train_size)
            model = train_sarima_model(values[train_start:train_end], p, d, q, P, D, Q, s)

        i += 1

    if i > last_drift_point:
        forecast_steps = len(df) - last_drift_point
        forecast_obj = model.get_forecast(steps=forecast_steps)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()

        all_forecasts.iloc[last_drift_point:, 0] = forecast.values
        all_forecasts.iloc[last_drift_point:, 1] = conf_int.iloc[:, 0].values
        all_forecasts.iloc[last_drift_point:, 2] = conf_int.iloc[:, 1].values

    all_forecasts.dropna(inplace=True)

    return all_forecasts


df = pd.read_csv("../../synth.csv") 
rawdata = df.value

window_size = 200

final_forecast = dynamic_retraining_with_drift_detection(df, window_size)

final_forecast = final_forecast.apply(pd.to_numeric, errors='coerce')
final_forecast = final_forecast.dropna()

if not pd.api.types.is_datetime64_any_dtype(final_forecast.index):
    final_forecast.index = pd.to_datetime(final_forecast.index)

predictions = final_forecast.forecast
lower_bounds = final_forecast.lower_bound
upper_bounds = final_forecast.upper_bound

ground_truth_values = df.value[200:]

errors = np.abs(np.array(ground_truth_values) - np.array(predictions))

squared_errors = np.square(errors)

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df.index[200:], ground_truth_values, label = "Temperature")
plt.plot(df.index[200:], predictions, label = "ARIMA Forecast", color='red')
axs.legend(fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('y')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/ARIMA_forecast_and_true_data.png') 

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
plt.plot(df.index[200:], errors, label = "Squared Error", color='orange')
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
axs.set_xlabel('y')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('./plots/ARIMA_error_distribution.png') 




