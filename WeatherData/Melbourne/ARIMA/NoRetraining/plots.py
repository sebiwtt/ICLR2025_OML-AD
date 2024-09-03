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


def train_single_ARIMA(df):

    train = df.value[:800]
    test = df.value[801:]

    (p, d, q) = (1, 0, 1)
    (P, D, Q, s) = (1, 1, 1, 52)
    sarima_model = sm.tsa.SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,s))
    model = sarima_model.fit(disp=False)
    
    forecast_steps = len(test)  
    return model.get_forecast(steps=forecast_steps)


df = pd.read_csv("../../melbourne_weekly_anomalies.csv")  

forecast = train_single_ARIMA(df)
predictions = forecast.predicted_mean

ground_truth_values = df.value[801:]

errors = np.abs(np.array(ground_truth_values) - np.array(predictions))

squared_errors = np.square(errors)

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df.index[801:], df.value[801:], label = "Temperature")
plt.plot(df.index[800:], predictions, label = "ARIMA Forecast", color='red')
axs.legend(fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Temperature [°C]')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/ARIMA_forecast_and_true_data.png') 

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
plt.plot(df.index[800:], errors, label = "Squared Error", color='orange')
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
axs.set_xlabel('Temperature [°C]')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('./plots/ARIMA_error_distribution.png')  


