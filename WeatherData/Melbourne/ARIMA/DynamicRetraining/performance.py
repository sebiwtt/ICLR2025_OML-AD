import math

import statsmodels.api as sm
from river import drift

import pandas as pd
import numpy as np
import math
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

def calculate_anomaly_scores(forecast, ground_truth_values, lower_bounds, upper_bounds):

    anomaly_scores = np.zeros(len(ground_truth_values))

    for i, true_value in enumerate(ground_truth_values):
        lower_bound = lower_bounds[i], 
        upper_bound = upper_bounds[i],
    
        prediction = forecast[i]
        
        threshold = np.abs(prediction-upper_bound) * 3
        error = np.abs(true_value - prediction)
    
        if error >= threshold:
            anomaly_scores[i] = 1.0
        else:
            anomaly_scores[i] = error / threshold

    return anomaly_scores


def train_sarima_model(train_data, p, d, q, P, D, Q, s):
    sarima_model = sm.tsa.SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
    model = sarima_model.fit(disp=False)
    return model


def dynamic_retraining_with_drift_detection(df, initial_train_size=800):
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

    (p, d, q) = (1, 0, 1)
    (P, D, Q, s) = (1, 1, 1, 52)

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


def train_and_evaluate(df):

    window_size = 800

    rawdata = df.value
    
    final_forecast = dynamic_retraining_with_drift_detection(df, window_size)
    
    final_forecast = final_forecast.apply(pd.to_numeric, errors='coerce')
    final_forecast = final_forecast.dropna()

    if not pd.api.types.is_datetime64_any_dtype(final_forecast.index):
        final_forecast.index = pd.to_datetime(final_forecast.index)

    predictions = final_forecast.forecast
    lower_bounds = final_forecast.lower_bound
    upper_bounds = final_forecast.upper_bound
    
    ground_truth_values = df.value[800:]

    scores = calculate_anomaly_scores(np.array(predictions), ground_truth_values, lower_bounds, upper_bounds) 

    anomaly_scores = np.array(scores)  
    true_labels = np.array(df.anomalous)[800:] 

    errors = np.abs(rawdata - predictions)

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

df = pd.read_csv("../../melbourne_weekly_anomalies.csv") 

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
