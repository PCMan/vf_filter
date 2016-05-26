#!/usr/bin/env python3
# import pyximport; pyximport.install()
# evaluation tools for classification
import numpy as np
import vf_classify
from sklearn import metrics


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


def custom_score(y_true, y_predict):
    counts = np.bincount(y_true)
    if len(counts) == 2:  # binary classification
        return metrics.fbeta_score(y_true, y_predict, beta=0.5, pos_label=vf_classify.DANGEROUS_RHYTHM, average="binary")
    elif len(counts) == 3:  # AHA classification
        shockable_idx = (y_true == vf_classify.SHOCKABLE)
        n_shockable = np.sum(shockable_idx)
        if n_shockable:
            shockable_f1 = metrics.fbeta_score(y_true[shockable_idx], y_predict[shockable_idx], beta=0.8, pos_label=vf_classify.SHOCKABLE, average="binary")
            shockable_f1 *= len(y_true) / n_shockable
        else:
            shockable_f1 = 0.0

        intermediate_idx = (y_true == vf_classify.INTERMEDIATE)
        n_intermediate = np.sum(intermediate_idx)
        if n_intermediate:
            intermidiate_f1 = metrics.fbeta_score(y_true[intermediate_idx], y_predict[intermediate_idx], beta=0.5, pos_label=vf_classify.INTERMEDIATE, average="binary")
            intermidiate_f1 *= len(y_true) / n_intermediate
        else:
            intermidiate_f1 = 0.0

        non_shockable_idx = (y_true == vf_classify.NON_SHOCKABLE)
        n_non_shockable = np.sum(non_shockable_idx)
        if n_non_shockable:
            non_shockable_f1 = metrics.f1_score(y_true[non_shockable_idx], y_predict[non_shockable_idx], pos_label=vf_classify.NON_SHOCKABLE, average="binary")
            non_shockable_f1 *= len(y_true) / n_non_shockable
        else:
            non_shockable_f1 = 0.0
        return np.mean([shockable_f1 * 0.95, intermidiate_f1 * 0.25, non_shockable_f1 * 0.99]) / (0.95 + 0.25 + 0.99)
    return 0.0
