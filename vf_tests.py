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
import argparse


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
    # parse command line arguments
    parser = argparse.ArgumentParser()
    # known estimators
    estimator_names = ("logistic_regression", "random_forest", "gradiant_boosting", "svc")
    parser.add_argument("-m", "--model", type=str, required=True, choices=estimator_names)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-j", "--jobs", type=int, default=-1)
    parser.add_argument("-t", "--iter", type=int, default=1)
    parser.add_argument("-s", "--scorer", type=str, choices=("ber", "f1", "accuracy"), default="ber")
    parser.add_argument("-f", "--fold", type=int, default=10)  # 10 fold CV by default
    parser.add_argument("-p", "--test-percent", type=int, default=30)  # 30% test set size
    args = parser.parse_args()

    # setup testing parameters
    n_jobs = args.jobs
    if n_jobs == -1 or n_jobs > mp.cpu_count():
        n_jobs = (mp.cpu_count() - 1) if mp.cpu_count() > 1 else 1

    print(args)
    n_test_iters = args.iter
    n_cv_folds = args.fold
    test_size = args.test_percent / 100
    if test_size > 1:
        test_size = 0.3

    # build scoring function
    if args.scorer == "ber":  # BER-based scoring function
        cv_scorer = metrics.make_scorer(balanced_error_rate, greater_is_better=False)
    else:
        cv_scorer = args.scorer
        # cv_scorer = metrics.make_scorer(metrics.fbeta_score, beta=10.0)

    # load features
    x_data, y_data, x_info = load_data(n_jobs)
    print("Summary:\n", "# of segments:", len(x_data), "# of VT/Vf:", np.sum(y_data), len(x_info))

    # normalize the features
    preprocessing.normalize(x_data)
    x_indicies = list(range(0, len(x_data)))

    # build estimators to test
    estimator_name = args.model
    model = None
    param_names = []
    if estimator_name == "logistic_regression":
        model = linear_model.LogisticRegressionCV(scoring=cv_scorer,
                                                              n_jobs=n_jobs,
                                                              cv=n_cv_folds,
                                                              class_weight=CLASS_WEIGHT)
    elif estimator_name == "random_forest":
        estimator = ensemble.RandomForestClassifier(class_weight=CLASS_WEIGHT)
        param_grid = {
            "n_estimators": list(range(10, 110, 10))
        }
        model = grid_search.GridSearchCV(estimator, param_grid,
                                        scoring=cv_scorer,
                                        n_jobs=n_jobs, cv=n_cv_folds, verbose=0)
        param_names = param_grid.keys()
    elif estimator_name == "gradient_boosting":
        estimator = ensemble.GradientBoostingClassifier(learning_rate=0.1)
        param_grid = {
            "n_estimators": list(range(150, 250, 10)),
            "max_depth": list(range(3, 8))
        }
        model = grid_search.GridSearchCV(estimator, param_grid,
                                        scoring=cv_scorer,
                                        n_jobs=n_jobs, cv=n_cv_folds, verbose=0)
        param_names = param_grid.keys()
    elif estimator_name == "svc":
        estimator = svm.SVC(shrinking=False, cache_size=2048, verbose=False, probability=True, class_weight=CLASS_WEIGHT)
        param_grid = {
            "C": np.logspace(-2, 1, 4),
            "gamma": np.logspace(-2, 0, 3)
        }
        model = grid_search.RandomizedSearchCV(estimator, param_grid,
                                              scoring=cv_scorer,
                                              n_jobs=n_jobs, cv=n_cv_folds, verbose=0)
        param_names = param_grid.keys()
    else:
        print("unknown estimator")
        return

    # Run the selected test
    csv_fields = ["se", "sp", "ppv", "acc", "se(sp95)", "se(sp99)", "tp", "tn", "fp", "fn"]
    csv_fields.extend(param_names)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        # perform the test for many times
        for iter in range(n_test_iters):
            print(estimator_name, iter)
            row = {}
            # Here we split the indicies of the rows rather than the data array itself.
            x_train_idx, x_test_idx, y_train, y_test = cross_validation.train_test_split(x_indicies,
                                                                                         y_data,
                                                                                         test_size=test_size,
                                                                                         stratify=y_data)
            x_train = x_data[x_train_idx]
            x_test = x_data[x_test_idx]
            x_test_info = x_info[x_test_idx]

            # perform the classification test
            model.fit(x_train, y_train)
            y_predict = model.predict(x_test)
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
            if hasattr(model, "predict_proba"):
                y_predict_scores = model.predict_proba(x_test)[:, 1]
                false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(y_test, y_predict_scores)
                # find sensitivity at 95% specificity
                x = np.searchsorted(false_pos_rate, 0.05)
                row["se(sp95)"] = true_pos_rate[x]

                x = np.searchsorted(false_pos_rate, 0.01)
                row["se(sp99)"] = true_pos_rate[x]

            # best parameters of grid search
            if hasattr(model, "best_params_"):
                row.update(model.best_params_)

            print("  ", row)
            writer.writerow(row)


if __name__ == "__main__":
    main()
