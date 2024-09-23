import math

import statsmodels.api as sm

import pandas as pd
import numpy as np
import math
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


def calculate_anomaly_scores(forecast, ground_truth_values, confidence_intervals):

    anomaly_scores = np.zeros(len(ground_truth_values))

    for i, true_value in enumerate(ground_truth_values):
        lower_bound = confidence_intervals.iloc[:, 0][i],  # Lower bound of CI
        upper_bound = confidence_intervals.iloc[:, 1][i],  # Upper bound of CI
        prediction = forecast[i]
        
        threshold = np.abs(prediction-upper_bound)
        error = np.abs(true_value - prediction)
    
        if error >= threshold:
            anomaly_scores[i] = 1.0
        else:
            anomaly_scores[i] = error / threshold

    return anomaly_scores


def train_single_ARIMA(df):

    train = df.value[:200]
    test = df.value[201:]

    (p, d, q) = (2, 0, 2)
    (P, D, Q, s) = (2, 1, 2, 12)
    sarima_model = sm.tsa.SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,s))
    model = sarima_model.fit(disp=False)
    
    forecast_steps = len(test)  
    return model.get_forecast(steps=forecast_steps)


def train_and_evaluate(df):
    rawdata = df.value

    forecasts = train_single_ARIMA(df)
    predictions = forecasts.predicted_mean
    confidence_intervals = forecasts.conf_int(alpha=0.05)
    
    ground_truth_values = df.value[201:]
    
    scores = calculate_anomaly_scores(np.array(predictions), ground_truth_values, confidence_intervals.reset_index())
    
    anomaly_scores = np.array(scores)  
    true_labels = np.array(df.anomalous)[201:] 
    
    errors = np.abs(np.array(df.value[201:]) - np.array(predictions))
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

df = pd.read_csv("../../synth.csv")  

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
