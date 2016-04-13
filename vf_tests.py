#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import metrics
from sklearn import svm
from sklearn import grid_search
from vf_data import load_data
import multiprocessing as mp
import csv
import os


N_TEST_ITERS = 100
N_CV_FOLDS = 10
CLASS_WEIGHT = None  # "balanced"


def balanced_error_rate(y_true, y_predict):
    incorrect = (y_true != y_predict)
    fp = np.sum(np.logical_and(y_predict, incorrect))
    pred_negative = np.logical_not(y_predict)
    fn = np.sum(np.logical_and(pred_negative, incorrect))
    n_positive = np.sum(y_true)
    n_negative = len(y_true) - n_positive
    return 0.5 * (fn / n_positive + fp / n_negative)


class ClassificationResult:
    def __init__(self, y_true, y_predict):
        correct = (y_true == y_predict)
        incorrect = np.logical_not(correct)
        pred_negative = np.logical_not(y_predict)
        tp = np.sum(np.logical_and(y_predict, correct))
        fp = np.sum(np.logical_and(y_predict, incorrect))
        tn = np.sum(np.logical_and(pred_negative, correct))
        fn = np.sum(np.logical_and(pred_negative, incorrect))
        self.sensitivity = tp / (tp + fn)
        self.specificity = tn / (tn + fp)
        self.precision = tp / (tp + fp)
        self.accuracy = (tp + tn) / len(y_true)
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn


def main():
    n_jobs = (mp.cpu_count() - 1) if mp.cpu_count() > 1 else 1
    output_dir = "reports"

    # load features
    x_data, y_data, x_info = load_data(n_jobs)
    print("Summary:\n", "# of segments:", len(x_data), "# of VT/Vf:", np.sum(y_data), len(x_info))

    # normalize the features
    preprocessing.normalize(x_data)
    x_indicies = list(range(0, len(x_data)))

    # BER-based scoring function
    cv_scorer = metrics.make_scorer(balanced_error_rate, greater_is_better=False)
    # cv_scorer = "f1"
    # cv_scorer = "accuracy"
    # cv_scorer = metrics.make_scorer(metrics.fbeta_score, beta=10.0)

    # build estimators to test
    estimators = []
    # Logistic regression
    estimator = linear_model.LogisticRegressionCV(scoring=cv_scorer, class_weight=CLASS_WEIGHT)
    estimators.append(("Logistic Regression", estimator))

    # Random forest
    estimator = ensemble.RandomForestClassifier(class_weight=CLASS_WEIGHT)
    grid = grid_search.GridSearchCV(estimator, {
                                        "n_estimators": list(range(10, 110, 10))
                                    },
                                    scoring=cv_scorer,
                                    n_jobs=n_jobs, cv=N_CV_FOLDS, verbose=0)
    estimators.append(("Random Forest", grid))

    # Gradient boosting
    estimator = ensemble.GradientBoostingClassifier(learning_rate=0.1)
    grid = grid_search.GridSearchCV(estimator, {
                                        "n_estimators": list(range(150, 250, 10)),
                                        "max_depth": list(range(3, 8))
                                    },
                                    scoring=cv_scorer,
                                    n_jobs=n_jobs, cv=N_CV_FOLDS, verbose=0)
    estimators.append(("Gradient Boosting", grid))

    # SVC with RBF kernel
    estimator = svm.SVC(shrinking=False, cache_size=2048, verbose=False, probability=True, class_weight=CLASS_WEIGHT)
    grid = grid_search.RandomizedSearchCV(estimator, {
                                            "C": np.logspace(-2, 1, 4),
                                            "gamma": np.logspace(-2, 0, 3)
                                          },
                                          scoring=cv_scorer,
                                          n_jobs=n_jobs, cv=N_CV_FOLDS, verbose=0)
    estimators.append(("SVM", grid))

    os.makedirs(output_dir, exist_ok=True)
    csv_fields = ("se", "sp", "ppv", "acc", "sp95", "sp99", "tp", "tn", "fp", "fn")
    for name, estimator in estimators:
        file_path = os.path.join(output_dir, "{0}_tests.csv".format(name))
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            # perform the test for many times
            for iter in range(N_TEST_ITERS):
                print(name, iter)
                row = {}
                # Here we split the indicies of the rows rather than the data array itself.
                x_train_idx, x_test_idx, y_train, y_test = cross_validation.train_test_split(x_indicies, y_data, test_size=0.3, stratify=y_data)
                x_train = x_data[x_train_idx]
                x_test = x_data[x_test_idx]
                x_test_info = x_info[x_test_idx]

                # perform the classification test
                estimator.fit(x_train, y_train)
                y_predict = estimator.predict(x_test)
                result = ClassificationResult(y_test, y_predict)
                row["se"] = result.sensitivity
                row["sp"] = result.specificity
                row["ppv"] = result.precision
                row["acc"] = result.accuracy
                row["tp"] = result.tp
                row["tn"] = result.tn
                row["fp"] = result.fp
                row["fn"] = result.fn

                # prediction with probabilities
                if hasattr(estimator, "predict_proba"):
                    y_predict_scores = estimator.predict_proba(x_test)[:, 1]
                    false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(y_test, y_predict_scores)
                    # find sensitivity at 95% specificity
                    x = np.searchsorted(false_pos_rate, 0.05)
                    row["sp95"] = true_pos_rate[x]

                    x = np.searchsorted(false_pos_rate, 0.01)
                    row["sp99"] = true_pos_rate[x]

                print("  ", row)
                writer.writerow(row)


if __name__ == "__main__":
    main()
