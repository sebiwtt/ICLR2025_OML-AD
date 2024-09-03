from multiprocessing import Process, Array
from time import sleep
import psutil
import numpy as np
import pandas as pd
import math

import statsmodels.api as sm

from alibi_detect.datasets import fetch_nab


def train_single_ARIMA(df):

    train = df.value[:800]
    test = df.value[801:]

    (p, d, q) = (1, 0, 1)
    (P, D, Q, s) = (1, 1, 1, 52)
    sarima_model = sm.tsa.SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,s))
    model = sarima_model.fit(disp=False)
    
    forecast_steps = len(test)  
    return model.get_forecast(steps=forecast_steps)


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
    forecast = train_single_ARIMA(df)


def simulation(data):
    repetitions = 100
    for i in range(repetitions):
        train(data)
        
if __name__ == '__main__':
    cpu_arr = Array('f', 1000)
    ram_arr = Array('f', 1000)

    df = pd.read_csv("../../melbourne_weekly_anomalies.csv")  

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
    df.to_csv("./results/ARIMA_cpu_usage.csv")

    df = pd.DataFrame({"value": ram})
    df = df.set_index('value')
    df.to_csv("./results/ARIMA_ram_usage.csv")