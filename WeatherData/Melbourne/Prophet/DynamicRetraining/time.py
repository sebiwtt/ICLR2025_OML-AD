import time
import pandas as pd
import numpy as np

from prophet import Prophet
import logging
from river import drift


def train_model(df_prophet, training_set_size):
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    
    drifts = []
    adwin = drift.ADWIN(delta=0.001,
                    max_buckets=10,
                    grace_period=10,
                    min_window_length=10,
                    clock=20
                   )
    
    initial_training_set_size = training_set_size

    train = df_prophet[:initial_training_set_size]
    remainder = df_prophet[initial_training_set_size:]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(train)
    
    final_forecast = pd.DataFrame(index=df_prophet.index, columns=["yhat", "yhat_lower", "yhat_upper"])
    
    future = model.make_future_dataframe(periods=len(remainder), freq='W')
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
            future = model.make_future_dataframe(periods=forecastlen, freq='W')
            forecast = model.predict(future)
            
            forecast_length = min(len(forecast), len(final_forecast) - drift_point)
            final_forecast.iloc[drift_point:drift_point + forecast_length, :] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].iloc[:forecast_length].values
    
    if not drifts or drifts[-1] != len(df_prophet):
        drift_point = drifts[-1] if drifts else initial_training_set_size
        forecastlen = len(df_prophet) - drift_point
        future = model.make_future_dataframe(periods=forecastlen, freq='W')
        forecast = model.predict(future)
        forecast_length = min(len(forecast), len(final_forecast) - drift_point)
        final_forecast.iloc[drift_point:drift_point + forecast_length, :] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].iloc[:forecast_length].values

    final_forecast = final_forecast.apply(pd.to_numeric, errors='coerce')
    final_forecast = final_forecast.dropna()

    if not pd.api.types.is_datetime64_any_dtype(final_forecast.index):
        final_forecast.index = pd.to_datetime(final_forecast.index)


    return final_forecast[800:]


def benchmark_model(df_prophet, n_runs=2, n_loops=100):
    results = []

    for run in range(n_runs):
        for loop in range(n_loops):
            start_time = time.time()

            forecast = train_model(df_prophet, 800)

            elapsed_time = time.time() - start_time
            results.append(elapsed_time * 1000)

    return results

df = pd.read_csv("../../melbourne_weekly_anomalies.csv")

df_prophet = df.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})

timing_results = benchmark_model(df_prophet)

mean_time = np.mean(timing_results)
std_time = np.std(timing_results)

print(f"Mean Time: {mean_time:.2f} ms")
print(f"Standard Deviation: {std_time:.2f} ms")

df = pd.DataFrame(timing_results, columns=["Time (ms)"])
df.to_csv("./results/time.csv", index=False)

print("Benchmark results saved to timing_results.csv")