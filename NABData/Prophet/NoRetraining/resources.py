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


def single_prophet_model(df, train_size):
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    
    train = df[:train_size]
    test = df[train_size+1:]
    
    model = Prophet()
    model.fit(train)
    
    forecastlen = len(test)
    
    future = model.make_future_dataframe(periods=forecastlen,freq ='h')
    
    forecast = model.predict(future)

    return forecast[800:]


def train(df_prophet):

    forecast = single_prophet_model(df_prophet, 800)


def simulation(df_prophet):
    repetitions = 100
    for i in range(repetitions):
        train(df_prophet)
        
if __name__ == '__main__':
    
    cpu_arr = Array('f', 1000)
    ram_arr = Array('f', 1000)

    ts = fetch_nab("realAWSCloudwatch/rds_cpu_utilization_e47b3b")  
    df = ts["data"]
    
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