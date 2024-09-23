import time
import pandas as pd
import numpy as np
import statsmodels.api as sm


def scheduled_retraining_sarima(df, window_size):

    forecast_horizon = window_size
    
    full_forecast = []
    full_upper_conf_int = [] 
    full_lower_conf_int = []
    train_end = window_size 

    while train_end <= len(df):
       
        train_data = df.value[train_end-window_size:train_end]

        (p, d, q) = (2, 0, 2)
        (P, D, Q, s) = (2, 1, 2, 12)

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


def benchmark_model(rawdata, n_runs=2, n_loops=100):
    results = []

    for run in range(n_runs):
        for loop in range(n_loops):
            start_time = time.time()

            forecast_values, upper_confidence_intervals, lower_confidence_intervals  = scheduled_retraining_sarima(df, 200)

            elapsed_time = time.time() - start_time
            results.append(elapsed_time * 1000)

    return results

df = pd.read_csv("../../synth.csv") 
timing_results = benchmark_model(df)

mean_time = np.mean(timing_results)
std_time = np.std(timing_results)

print(f"Mean Time: {mean_time:.2f} ms")
print(f"Standard Deviation: {std_time:.2f} ms")

df = pd.DataFrame(timing_results, columns=["Time (ms)"])
df.to_csv("./results/time.csv", index=False)

print("Benchmark results saved to timing_results.csv")