from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix

import numpy as np


def get_predictions(model, some_it, N):
    y_true = np.zeros((N,), dtype=np.int8)
    y_pred = np.zeros((N,), dtype=np.float32)
    nt = 0
    while nt < N:
        x, y = next(some_it)
        ni = x.shape[0]

        y_true[nt:nt + ni] = y
        y_pred[nt:nt + ni] = model.predict_on_batch(x)[:, 0]

        nt += ni

    return y_true, y_pred


def predict_and_run_func(model, some_it, N, func):
    y_true, y_pred = get_predictions(model, some_it, N)
    return func(y_true, y_pred)


def get_auc_score(model, some_it, N):
    return predict_and_run_func(model, some_it, N, roc_auc_score)


def get_precision_recall(model, some_it, N):
    return predict_and_run_func(model, some_it, N, precision_recall_curve)


def get_confusion_matrix(model, some_it, N, threshold):
    y_true, y_pred = get_predictions(model, some_it, N)

    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    return confusion_matrix(y_true, y_pred)
