from multiprocessing import Process, Array
from time import sleep
import psutil
import numpy as np
import pandas as pd
import math

from prophet import Prophet
import logging

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


def train_model(df_prophet):
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    
    window_size = 800  # Number of data points to train on and forecast
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
        future = model.make_future_dataframe(periods=window_size, freq='W')
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
        
        future = model.make_future_dataframe(periods=total_data_points - start, freq='W')
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
    

def train(df_prophet):

    forecast = train_model(df_prophet)


def simulation(df_prophet):
    repetitions = 100
    for i in range(repetitions):
        train(df_prophet)
        
if __name__ == '__main__':
    
    cpu_arr = Array('f', 1000)
    ram_arr = Array('f', 1000)

    df = pd.read_csv("../../melbourne_weekly_anomalies.csv")
    
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