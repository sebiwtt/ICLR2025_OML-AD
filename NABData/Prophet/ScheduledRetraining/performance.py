import math

from prophet import Prophet
import logging

import pandas as pd
import numpy as np
import math
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from alibi_detect.datasets import fetch_nab


def calculate_anomaly_scores(forecast, ground_truth_values):

    anomaly_scores = np.zeros(len(ground_truth_values))

    for i, true_value in enumerate(ground_truth_values):
        lower_bound = forecast["yhat_lower"][i]
        upper_bound = forecast["yhat_upper"][i] 
        prediction = forecast["yhat"][i]
        
        threshold = np.abs(prediction-upper_bound) * 3
        error = np.abs(true_value - prediction)
    
        if error >= threshold:
            anomaly_scores[i] = 1.0
        else:
            anomaly_scores[i] = error / threshold

    return anomaly_scores


def train_model(df_prophet):
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    
    window_size = 800  
    total_data_points = len(df_prophet)
    
    final_forecast = pd.DataFrame(index=df_prophet.index, columns=["yhat", "yhat_lower", "yhat_upper"])
    
    start = 1
    
    while start + window_size < total_data_points:
        train = df_prophet[start:start + window_size]
        
        model = Prophet()
        model.fit(train)
        
        future = model.make_future_dataframe(periods=window_size, freq='h')
        forecast = model.predict(future)
        
        forecast = forecast.iloc[window_size:]
        
        forecast_length = min(len(forecast), total_data_points - (start + window_size))
        final_forecast.iloc[start + window_size:start + window_size + forecast_length, :] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].iloc[:forecast_length].values
     
        start += window_size
    
    if start < total_data_points:
        remaining_train = df_prophet[start:]
        
        model = Prophet()
        model.fit(remaining_train)
        
        future = model.make_future_dataframe(periods=total_data_points - start, freq='h')
        forecast = model.predict(future)
        
        forecast = forecast.iloc[len(remaining_train):]
        
        forecast_length = min(total_data_points - start, len(forecast))
        final_forecast.iloc[start:start + forecast_length, :] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].iloc[:forecast_length].values

        final_forecast = final_forecast.apply(pd.to_numeric, errors='coerce')
        final_forecast = final_forecast.dropna()
        
        if not pd.api.types.is_datetime64_any_dtype(final_forecast.index):
            final_forecast.index = pd.to_datetime(final_forecast.index)
    
    return final_forecast
    

def train_and_evaluate(df):

    df_prophet = df["data"]

    df_prophet = df_prophet.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})

    no_retraining_forecast = train_model(df_prophet)
    ground_truth_values = df_prophet.y[801:]
    
    scores = calculate_anomaly_scores(no_retraining_forecast.reset_index(), ground_truth_values)
    
    anomaly_scores = np.array(scores)  
    true_labels = np.array(df["target"].is_outlier[801:]) 

    errors = np.abs(np.array(ground_truth_values) - np.array(no_retraining_forecast["yhat"]))

    squared_errors = np.square(errors)
    
    mae = errors.mean()
    mse = squared_errors.mean()
    
    thresholds = np.arange(0.0, 1.01, 0.01) 
    max_f1 = 0
    optimal_threshold = 0
    
    for threshold in thresholds:
        predicted_labels = np.where(anomaly_scores >= threshold, 1, 0)
        f1 = f1_score(true_labels, predicted_labels)
    
        if f1 > max_f1:
            max_f1 = f1
            optimal_threshold = threshold
    
    predicted_labels = np.where(anomaly_scores >= optimal_threshold, 1, 0)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    
    recall = recall_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    roc_auc = roc_auc_score(true_labels, anomaly_scores)

    return (mae, mse, accuracy, tp, fp, tn, fn, recall, precision, f1, roc_auc)

df = fetch_nab("realAWSCloudwatch/rds_cpu_utilization_e47b3b")  

repetitions = 100
results = {
    'mae': [],
    'mse': [],
    'accuracy': [],
    'tp': [],
    'fp': [],
    'tn': [],
    'fn': [],
    'recall': [],
    'precision': [],
    'f1': [],
    'roc_auc': []
}

for i in range(repetitions):
    mae, mse, accuracy, tp, fp, tn, fn, recall, precision, f1, roc_auc = train_and_evaluate(df)
    
    results['mae'].append(mae)
    results['mse'].append(mse)
    results['accuracy'].append(accuracy)
    results['tp'].append(tp)
    results['fp'].append(fp)
    results['tn'].append(tn)
    results['fn'].append(fn)
    results['recall'].append(recall)
    results['precision'].append(precision)
    results['f1'].append(f1)
    results['roc_auc'].append(roc_auc)

results_df = pd.DataFrame(results)

results_df.to_csv("./results/performance.csv", index=False)

print("Results saved to results.csv")
