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

def single_prophet_model(df, train_size):
    train = df[:train_size]
    test = df[train_size+1:]
    
    model = Prophet()
    model.fit(train)
    
    forecastlen = len(test)
    
    future = model.make_future_dataframe(periods=forecastlen,freq ='h')
    
    forecast = model.predict(future)

    return forecast[800:]

df = fetch_nab("realAWSCloudwatch/rds_cpu_utilization_e47b3b")
df_prophet = df["data"]
df_prophet = df_prophet.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})

forecast = single_prophet_model(df_prophet, 800)
predictions = forecast["yhat"]

ground_truth_values = df_prophet.y[801:]

errors = np.abs(ground_truth_values - predictions)
squared_errors = np.square(errors)

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df["data"].index[801:], df_prophet.y[801:], label = "rds_cpu_utilization_e47b3b")
plt.plot(df["data"].index[801:], predictions, label = "Prophet Forecast", color='red')
axs.legend(fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('%')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/Prophet_forecast_and_true_data.png') 

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
plt.plot(df["data"].index[800:], errors, label = "Squared Error", color='orange')
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
axs.set_xlabel('Traffic Value (Gbits)')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('./plots/Prophet_error_distribution.png') 


