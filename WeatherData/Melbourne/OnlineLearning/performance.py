import math
from river import anomaly
from river import time_series
from river import preprocessing
from river import linear_model
from river import optim
import pandas as pd
import numpy as np
import math
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from alibi_detect.datasets import fetch_nab
    
def train_and_evaluate(df):
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

    anomaly_scores = np.array(scores)  
    true_labels = np.array(df.anomalous) 

    errors = np.abs(rawdata - predictions)
    errors = errors[24:] #Only counting from warmup window for a fair comparison, ARIMA and Prophet 

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

df = pd.read_csv("../melbourne_weekly_anomalies.csv")

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
