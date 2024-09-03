import math

import statsmodels.api as sm

import pandas as pd
import numpy as np
import math
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from alibi_detect.datasets import fetch_nab


def calculate_anomaly_scores(forecast, ground_truth_values, upper_confidence_intervals, lower_confidence_intervals):

    anomaly_scores = np.zeros(len(ground_truth_values))

    for i, true_value in enumerate(ground_truth_values):
        lower_bound = lower_confidence_intervals[i],  # Lower bound of CI
        upper_bound = upper_confidence_intervals[i],  # Upper bound of CI
        prediction = forecast[i]
        
        threshold = np.abs(prediction-upper_bound) * 10
        error = np.abs(true_value - prediction)
    
        if error >= threshold:
            anomaly_scores[i] = 1.0
        else:
            anomaly_scores[i] = error / threshold

    return anomaly_scores


def scheduled_retraining_sarima(df, window_size):

    forecast_horizon = window_size
    
    full_forecast = []
    full_upper_conf_int = [] 
    full_lower_conf_int = []
    train_end = window_size 

    while train_end <= len(df):
       
        train_data = df.value[train_end-window_size:train_end]

        (p, d, q) = (1, 0, 1)
        (P, D, Q, s) = (1, 1, 1, 12)

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


def train_and_evaluate(df):
    rawdata = df["data"].value
    
    forecast_values, upper_confidence_intervals, lower_confidence_intervals  = scheduled_retraining_sarima(df["data"], 800)
    
    predictions = forecast_values
    
    ground_truth_values = df["data"].value[800:]

    scores = calculate_anomaly_scores(predictions, ground_truth_values, upper_confidence_intervals, lower_confidence_intervals)

    anomaly_scores = np.array(scores)  
    true_labels = np.array(df["target"].is_outlier)[800:] 

    errors = np.abs(df["data"].value[800:] - predictions)

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
