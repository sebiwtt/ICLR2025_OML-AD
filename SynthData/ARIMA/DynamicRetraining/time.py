import time
import pandas as pd
import numpy as np

import statsmodels.api as sm
from river import drift

from alibi_detect.datasets import fetch_nab


def train_sarima_model(train_data, p, d, q, P, D, Q, s):
    sarima_model = sm.tsa.SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
    model = sarima_model.fit(disp=False)
    return model


def dynamic_retraining_with_drift_detection(df, initial_train_size=200):
    values = df.value
    adwin = drift.ADWIN(delta=0.001,
                    max_buckets=10,
                    grace_period=10,
                    min_window_length=10,
                    clock=20
                   )
    drifts = []
    
    all_forecasts = pd.DataFrame(index=df.index, columns=["forecast", "lower_bound", "upper_bound"])

    train_start = 0
    train_end = initial_train_size

    train_data = values[train_start:train_end]

    (p, d, q) = (2, 0, 2)
    (P, D, Q, s) = (2, 1, 2, 12)

    sarima_model = sm.tsa.SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
    model = sarima_model.fit(disp=False)

    last_drift_point = train_end

    i = train_end
    while i < len(values):
        adwin.update(values[i])
        if adwin.drift_detected:
            print(f"Change detected at index {i}, input value: {values[i]}")
            drifts.append(i)

            forecast_steps = i - last_drift_point
            if forecast_steps > 0:
                forecast_obj = model.get_forecast(steps=forecast_steps)
                forecast = forecast_obj.predicted_mean
                conf_int = forecast_obj.conf_int()
                all_forecasts.iloc[last_drift_point:i, 0] = forecast.values
                all_forecasts.iloc[last_drift_point:i, 1] = conf_int.iloc[:, 0].values
                all_forecasts.iloc[last_drift_point:i, 2] = conf_int.iloc[:, 1].values

            last_drift_point = i

      
            train_end = i
            train_start = max(0, train_end - initial_train_size)
            model = train_sarima_model(values[train_start:train_end], p, d, q, P, D, Q, s)

        i += 1

    if i > last_drift_point:
        forecast_steps = len(df) - last_drift_point
        forecast_obj = model.get_forecast(steps=forecast_steps)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()

        all_forecasts.iloc[last_drift_point:, 0] = forecast.values
        all_forecasts.iloc[last_drift_point:, 1] = conf_int.iloc[:, 0].values
        all_forecasts.iloc[last_drift_point:, 2] = conf_int.iloc[:, 1].values

    all_forecasts.dropna(inplace=True)

    return all_forecasts
    

def benchmark_model(rawdata, n_runs=2, n_loops=100):
    results = []

    for run in range(n_runs):
        for loop in range(n_loops):
            start_time = time.time()

            final_forecast = dynamic_retraining_with_drift_detection(df)

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