#!/usr/bin/env python3
# import pyximport; pyximport.install()
# evaluation tools for classification
import numpy as np


def balanced_error_rate(y_true, y_predict):
    incorrect = (y_true != y_predict)
    fp = np.sum(np.logical_and(y_predict, incorrect))
    pred_negative = np.logical_not(y_predict)
    fn = np.sum(np.logical_and(pred_negative, incorrect))
    n_positive = np.sum(y_true)
    n_negative = len(y_true) - n_positive
    return 0.5 * (fn / n_positive + fp / n_negative)


class BinaryClassificationResult:
    def __init__(self, y_true, y_predict):
        correct = (y_true == y_predict)
        incorrect = np.logical_not(correct)
        pred_negative = np.logical_not(y_predict)
        tp = np.sum(np.logical_and(y_predict, correct))
        fp = np.sum(np.logical_and(y_predict, incorrect))
        tn = np.sum(np.logical_and(pred_negative, correct))
        fn = np.sum(np.logical_and(pred_negative, incorrect))
        self.sensitivity = 0.0 if tp == 0 else tp / (tp + fn)
        self.specificity = 0.0 if tn == 0 else tn / (tn + fp)
        self.precision = 0.0 if tp == 0 else tp / (tp + fp)
        self.accuracy = (tp + tn) / len(y_true)
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn


class MultiClassificationResult:
    def __init__(self, y_true, y_predict, classes):
        n = len(y_true)
        results = []
        for c in classes:
            # one-versus all for this class
            bin_y_true = np.zeros((n, 1))
            bin_y_true[(y_true == c)] = 1
            bin_y_predict = np.zeros((n, 1))
            bin_y_predict[y_predict == c] = 1
            result = BinaryClassificationResult(bin_y_true, bin_y_predict)
            results.append(result)
        self.results = results
