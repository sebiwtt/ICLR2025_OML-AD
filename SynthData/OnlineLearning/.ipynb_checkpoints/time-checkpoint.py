import time
import pandas as pd
import numpy as np
from river import anomaly
from river import time_series
from river import preprocessing
from river import linear_model
from river import optim

def benchmark_model(rawdata, n_runs=2, n_loops=100):
    results = []

    for run in range(n_runs):
        for loop in range(n_loops):
            start_time = time.time()

            best_params = {'d': 1, 'intercept_lr': 1e-10, 'l2': 0.01, 'learning_rate': 0.0011, 'p': 2, 'period': 52, 'q': 2, 'sd': 0, 'sp': 2, 'sq': 2}

            
            predictive_model = time_series.SNARIMAX(
                p=2,
                d=1,
                q=2,
                m=52,
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
                warmup_period=0
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

            elapsed_time = time.time() - start_time
            results.append(elapsed_time * 1000)

    return results


df = pd.read_csv("../synth.csv")

rawdata = df.value

timing_results = benchmark_model(rawdata)

mean_time = np.mean(timing_results)
std_time = np.std(timing_results)

print(f"Mean Time: {mean_time:.2f} ms")
print(f"Standard Deviation: {std_time:.2f} ms")

df = pd.DataFrame(timing_results, columns=["Time (ms)"])
df.to_csv("./results/time.csv", index=False)

print("Benchmark results saved to timing_results.csv")