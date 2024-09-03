from multiprocessing import Process, Array
from time import sleep
import psutil
import numpy as np
import pandas as pd
import math

import statsmodels.api as sm

from alibi_detect.datasets import fetch_nab


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


def cpu_measure(a):
    cnt = 0
    while True:
        a[cnt] = psutil.cpu_percent()
        cnt += 1
        sleep(0.1)


def ram_measure(a):
    cnt = 0
    while True:
        a[cnt] = psutil.virtual_memory().percent
        cnt += 1
        sleep(0.1)

def train(df):

    forecast_values, upper_confidence_intervals, lower_confidence_intervals  = scheduled_retraining_sarima(df, 800)


def simulation(data):
    repetitions = 100
    for i in range(repetitions):
        train(data)
        
if __name__ == '__main__':
    cpu_arr = Array('f', 1000)
    ram_arr = Array('f', 1000)

    ts = fetch_nab("realAWSCloudwatch/rds_cpu_utilization_e47b3b")  
    df = ts["data"]

    p_ram = Process(target=ram_measure, args=(ram_arr,))
    p_cpu = Process(target=cpu_measure,  args=(cpu_arr,))
    p_simulation = Process(target=simulation, args=(df,))
    p_ram.start()
    p_cpu.start()

    p_simulation.start()

    sleep(5)

    # p_simulation.join()
    p_ram.terminate()
    p_ram.join()
    p_cpu.terminate()
    p_cpu.join()

    cpu = np.array(cpu_arr[:])
    ram = np.array(ram_arr[:])
    ram = ram[ram != 0]
    cpu = cpu[cpu != 0]

    df = pd.DataFrame({"value": cpu})
    df = df.set_index('value')
    df.to_csv("./results/ARIMA_cpu_usage.csv")

    df = pd.DataFrame({"value": ram})
    df = df.set_index('value')
    df.to_csv("./results/ARIMA_ram_usage.csv")