from multiprocessing import Process, Array
from time import sleep
import psutil
import numpy as np
import pandas as pd
import math

from prophet import Prophet
import logging
from river import drift

from alibi_detect.datasets import fetch_nab


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


def train_model(df_prophet, training_set_size):
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    
    drifts = []
    adwin = drift.ADWIN()
    
    initial_training_set_size = training_set_size

    train = df_prophet[:initial_training_set_size]
    remainder = df_prophet[initial_training_set_size:]

    model = Prophet()
    model.fit(train)
    
    final_forecast = pd.DataFrame(index=df_prophet.index, columns=["yhat", "yhat_lower", "yhat_upper"])
    
    future = model.make_future_dataframe(periods=len(remainder), freq='h')
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
            future = model.make_future_dataframe(periods=forecastlen, freq='h')
            forecast = model.predict(future)
            
            forecast_length = min(len(forecast), len(final_forecast) - drift_point)
            final_forecast.iloc[drift_point:drift_point + forecast_length, :] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].iloc[:forecast_length].values
    
    if not drifts or drifts[-1] != len(df_prophet):
        drift_point = drifts[-1] if drifts else initial_training_set_size
        forecastlen = len(df_prophet) - drift_point
        future = model.make_future_dataframe(periods=forecastlen, freq='h')
        forecast = model.predict(future)
        forecast_length = min(len(forecast), len(final_forecast) - drift_point)
        final_forecast.iloc[drift_point:drift_point + forecast_length, :] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].iloc[:forecast_length].values

    final_forecast = final_forecast.apply(pd.to_numeric, errors='coerce')
    final_forecast = final_forecast.dropna()

    if not pd.api.types.is_datetime64_any_dtype(final_forecast.index):
        final_forecast.index = pd.to_datetime(final_forecast.index)


    return final_forecast[800:]
    

def train(df_prophet):

    forecast = train_model(df_prophet, 800)


def simulation(df_prophet):
    repetitions = 100
    for i in range(repetitions):
        train(df_prophet)
        
if __name__ == '__main__':
    
    cpu_arr = Array('f', 1000)
    ram_arr = Array('f', 1000)

    ts = fetch_nab("realAWSCloudwatch/rds_cpu_utilization_e47b3b")  
    df = ts["data"]
    
    df_prophet = df.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})

    p_ram = Process(target=ram_measure, args=(ram_arr,))
    p_cpu = Process(target=cpu_measure,  args=(cpu_arr,))
    p_simulation = Process(target=simulation, args=(df_prophet,))
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
    df.to_csv("./results/Prophet_cpu_usage.csv")

    df = pd.DataFrame({"value": ram})
    df = df.set_index('value')
    df.to_csv("./results/Prophet_ram_usage.csv")