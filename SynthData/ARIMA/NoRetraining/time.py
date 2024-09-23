import time
import pandas as pd
import numpy as np

import statsmodels.api as sm

def train_single_ARIMA(df):

    train = df.value[:200]
    test = df.value[201:]

    (p, d, q) = (2, 0, 2)
    (P, D, Q, s) = (2, 1, 2, 12)
    sarima_model = sm.tsa.SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,s))
    model = sarima_model.fit(disp=False)
    
    forecast_steps = len(test)  
    return model.get_forecast(steps=forecast_steps)

def benchmark_model(rawdata, n_runs=2, n_loops=100):
    results = []

    for run in range(n_runs):
        for loop in range(n_loops):
            start_time = time.time()
            train_single_ARIMA(df)

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