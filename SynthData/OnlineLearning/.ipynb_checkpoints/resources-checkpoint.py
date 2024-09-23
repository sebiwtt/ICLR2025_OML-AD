from multiprocessing import Process, Array
from time import sleep
import psutil
import numpy as np
import pandas as pd
import math
from river import anomaly
from river import time_series
from river import preprocessing
from river import linear_model
from river import optim


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
    rawdata = df.value
   
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


def simulation(data):
    repetitions = 100
    for i in range(repetitions):
        train(data)
        
if __name__ == '__main__':
    cpu_arr = Array('f', 1000)
    ram_arr = Array('f', 1000)

    df = pd.read_csv("../synth.csv")

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
    df.to_csv("./results/PAD_cpu_usage.csv")

    df = pd.DataFrame({"value": ram})
    df = df.set_index('value')
    df.to_csv("./results/PAD_ram_usage.csv")

