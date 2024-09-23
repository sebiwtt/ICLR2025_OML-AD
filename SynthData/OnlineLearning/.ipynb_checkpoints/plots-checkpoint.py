import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
import numpy as np

from river import anomaly
from river import time_series
from river import preprocessing
from river import linear_model
from river import optim
from alibi_detect.datasets import fetch_nab

matplotlib.rcParams.update({'font.size': 14})

# ---- CPU and RAM ----

df_cpu = pd.read_csv("./results/PAD_cpu_usage.csv")
df_ram = pd.read_csv("./results/PAD_ram_usage.csv")

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df_cpu.value, label = "CPU")
axs.legend(fontsize=20)
plt.yticks(fontsize=10)
plt.xticks([])
plt.ylabel('CPU Usage [%]')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/PAD_cpu_usage.png') 

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df_ram.value, label = "RAM")
axs.legend(fontsize=20)
plt.yticks(fontsize=10)
plt.xticks([])
plt.ylabel('RAM Usage [%]')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/PAD_ram_usage.png') 

# ---- Forecast and actual data ---- 

df = pd.read_csv("../synth.csv")
rawdata = df.value

predictive_model = time_series.SNARIMAX(
    p=2,
    d=1,
    q=2,
    m=1,
    sd=0,
    sq = 2,
    sp = 2,
    regressor=(
        preprocessing.StandardScaler()
        | linear_model.LinearRegression(
            optimizer=optim.SGD(0.001),
            l2 = 0.01,
            intercept_lr=1e-10
        )
    ),
    )
    
PAD = anomaly.PredictiveAnomalyDetection(
    predictive_model,
    horizon=1,
    n_std=3.0,
    warmup_period=20
)


scores = []
predictions = []
errors = []

for y in rawdata:

    score = PAD.score_one(None, y)
    scores.append(score)

    pred = PAD.predictive_model.forecast(PAD.horizon)[0]
    squared_error = (pred - y) ** 2
    errors.append(squared_error)
    predictions.append(pred)

    PAD = PAD.learn_one(None, y)

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
axs.plot(df.index, rawdata, label = "Temperature")
plt.plot(df.index, predictions, label = "PAD Forecast", color='red')
axs.legend(fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('y')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/PAD_forecast_and_true_data.png') 

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
plt.plot(df.index, errors, label = "Squared Error", color='orange')
axs.legend(fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
axs.set_ylim(0.0, 150.0) 
plt.ylabel('Squared Error')
plt.xlabel('Time')
plt.tight_layout()
plt.savefig('./plots/PAD_error.png') 


fig, axs = plt.subplots(1, 1, figsize=(15, 6))
sns.kdeplot(errors, ax=axs, fill=True)
axs.set_xlabel('y')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('./plots/PAD_error_distribution.png') 


