import time
import pandas as pd
import numpy as np

from prophet import Prophet
import logging

def single_prophet_model(df, train_size):
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    
    train = df[:train_size]
    test = df[train_size+1:]
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
    model.fit(train)
    
    forecastlen = len(test)
    
    future = model.make_future_dataframe(periods=forecastlen,freq ='W')
    
    forecast = model.predict(future)

    return forecast[800:]

def benchmark_model(rawdata, n_runs=2, n_loops=100):
    results = []

    for run in range(n_runs):
        for loop in range(n_loops):
            start_time = time.time()

            forecast = single_prophet_model(rawdata, 800)

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