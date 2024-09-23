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
plt.savefig('./plots/synth.png') 

# ---- Forecast and actual data ---- 

def single_prophet_model(df, train_size):
    train = df[:train_size]
    test = df[train_size+1:]
    
    model = Prophet()
    model.fit(train)
    
    forecastlen = len(test)
    
    future = model.make_future_dataframe(periods=forecastlen,freq ='D')
    
    forecast = model.predict(future)

    return forecast[300:]

df = pd.read_csv("../../synth.csv")
df_prophet = df.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})

forecast = single_prophet_model(df_prophet, 300)
predictions = forecast["yhat"]

ground_truth_values = df_prophet.y[301:]

errors = np.abs(ground_truth_values - predictions)
squared_errors = np.square(errors)

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df.index[301:], df_prophet.y[301:], label = "Temperature")
plt.plot(df.index[301:], predictions, label = "Prophet Forecast", color='red')
axs.legend(fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('y')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/Prophet_forecast_and_true_data.png') 

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
plt.plot(df.index[300:], errors, label = "Squared Error", color='orange')
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


