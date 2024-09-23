import time
import pandas as pd
import numpy as np

from prophet import Prophet
import logging

def train_model(df_prophet):
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    
    window_size = 200  # Number of data points to train on and forecast
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
        future = model.make_future_dataframe(periods=window_size, freq='D')
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
        
        future = model.make_future_dataframe(periods=total_data_points - start, freq='D')
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


def benchmark_model(df_prophet, n_runs=2, n_loops=100):
    results = []

    for run in range(n_runs):
        for loop in range(n_loops):
            start_time = time.time()

            forecast = train_model(df_prophet)

            elapsed_time = time.time() - start_time
            results.append(elapsed_time * 1000)

    return results

df = pd.read_csv("../../synth.csv")

df_prophet = df.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})

timing_results = benchmark_model(df_prophet)

mean_time = np.mean(timing_results)
std_time = np.std(timing_results)

print(f"Mean Time: {mean_time:.2f} ms")
print(f"Standard Deviation: {std_time:.2f} ms")

df = pd.DataFrame(timing_results, columns=["Time (ms)"])
df.to_csv("./results/time.csv", index=False)

print("Benchmark results saved to timing_results.csv")