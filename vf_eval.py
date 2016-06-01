#!/usr/bin/env python3
# import pyximport; pyximport.install()
# evaluation tools for classification
import numpy as np
import vf_classify
from sklearn import metrics


scorer_names = ("ber", "f1", "accuracy", "precision", "f1_weighted",
                "precision_weighted", "f_beta", "max_recall", "custom",
                "f1_macro", "f1_binary")


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


def balanced_error_rate(y_true, y_predict):
    incorrect = (y_true != y_predict)
    fp = np.sum(np.logical_and(y_predict, incorrect))
    pred_negative = np.logical_not(y_predict)
    fn = np.sum(np.logical_and(pred_negative, incorrect))
    n_positive = np.sum(y_true)
    n_negative = len(y_true) - n_positive
    return 0.5 * (fn / n_positive + fp / n_negative)


def to_bin_label(y, pos_label):
    bin_y = np.zeros(y.shape)
    bin_y[y == pos_label] = 1
    return bin_y


def custom_score(y_true, y_predict):
    class_counts = np.bincount(y_true)
    if len(class_counts) == 2:  # binary classification
        return metrics.fbeta_score(y_true, y_predict, beta=0.5, pos_label=vf_classify.DANGEROUS_RHYTHM, average="binary")
    elif len(class_counts) == 3:  # AHA classification
        y_bin_true = to_bin_label(y_true, vf_classify.SHOCKABLE)
        y_bin_predict = to_bin_label(y_predict, vf_classify.SHOCKABLE)
        n_shockable = np.sum(y_bin_true)
        if n_shockable:
            shockable_score = metrics.recall_score(y_bin_true, y_bin_predict, average="binary")
        else:
            shockable_score = 0.0

        y_bin_true = to_bin_label(y_true, vf_classify.INTERMEDIATE)
        y_bin_predict = to_bin_label(y_predict, vf_classify.INTERMEDIATE)
        n_intermediate = np.sum(y_bin_true)
        if n_intermediate:
            intermidiate_score = metrics.fbeta_score(y_bin_true, y_bin_predict, beta=2, average="binary")
        else:
            intermidiate_score = 0.0

        y_bin_true = to_bin_label(y_true, vf_classify.NON_SHOCKABLE)
        y_bin_predict = to_bin_label(y_predict, vf_classify.NON_SHOCKABLE)
        n_non_shockable = np.sum(y_bin_true)
        if n_non_shockable:
            non_shockable_score = metrics.recall_score(y_bin_true, y_bin_predict, average="binary")
        else:
            non_shockable_score = 0.0
        # return np.mean([shockable_score * 0.95, intermidiate_score * 0.25, non_shockable_score * 0.99]) / (0.95 + 0.25 + 0.99)
        return shockable_score
    return 0.0


def max_recall_score(y_true, y_predict):
    class_counts = np.bincount(y_true)
    if len(class_counts) == 2:  # binary classification
        y_bin_true = to_bin_label(y_true, vf_classify.DANGEROUS_RHYTHM)
        y_bin_predict = to_bin_label(y_predict, vf_classify.DANGEROUS_RHYTHM)
    elif len(class_counts) == 3:  # AHA classification
        y_bin_true = to_bin_label(y_true, vf_classify.SHOCKABLE)
        y_bin_predict = to_bin_label(y_predict, vf_classify.SHOCKABLE)
    return metrics.recall_score(y_bin_true, y_bin_predict, average="binary")


# treat the multi-class problem as if it's binary classification, and only care about shock/non-shock
def f1_binary_score(y_true, y_predict):
    class_counts = np.bincount(y_true)
    if len(class_counts) == 3:  # AHA classification: make it a shock/non-shock binary problem
        y_true = to_bin_label(y_true, vf_classify.SHOCKABLE)
        y_predict = to_bin_label(y_predict, vf_classify.SHOCKABLE)
    return metrics.f1_score(y_true, y_predict, average="binary")


def get_scorer(name):
    # build scoring function
    if name == "ber":  # BER-based scoring function
        cv_scorer = metrics.make_scorer(balanced_error_rate, greater_is_better=False)
    elif name == "f1_macro":
        cv_scorer = metrics.make_scorer(metrics.f1_score, average="macro")
    elif name == "f1_binary":
        cv_scorer = metrics.make_scorer(f1_binary_score)
    elif name == "f_beta":
        cv_scorer = metrics.make_scorer(metrics.fbeta_score, beta=2, average="weighted")
    elif name == "custom":  # our custom error function
        cv_scorer = metrics.make_scorer(custom_score)
    elif name == "max_recall":  # only recall of shockable class
        cv_scorer = metrics.make_scorer(max_recall_score)
    else:
        cv_scorer = name
    return cv_scorer

