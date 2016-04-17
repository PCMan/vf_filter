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
from vf_eval import *
import multiprocessing as mp
import csv
import argparse


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    # known estimators
    estimator_names = ("logistic_regression", "random_forest", "adaboost", "gradient_boosting", "svc")
    parser.add_argument("-m", "--model", type=str, required=True, choices=estimator_names)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-j", "--jobs", type=int, default=-1)
    parser.add_argument("-t", "--iter", type=int, default=1)
    parser.add_argument("-s", "--scorer", type=str, choices=("ber", "f1", "accuracy", "precision"), default="ber")
    parser.add_argument("-c", "--cv-fold", type=int, default=10)  # 10 fold CV by default
    parser.add_argument("-p", "--test-percent", type=int, default=30)  # 30% test set size
    parser.add_argument("-b", "--balanced-weight", action="store_true")  # used balanced class weighting
    parser.add_argument("-f", "--features", type=int, nargs="+")  # feature selection
    args = parser.parse_args()

    # setup testing parameters
    n_jobs = args.jobs
    if n_jobs == -1 or n_jobs > mp.cpu_count():
        n_jobs = (mp.cpu_count() - 1) if mp.cpu_count() > 1 else 1

    print(args)
    n_test_iters = args.iter
    n_cv_folds = args.cv_fold
    test_size = args.test_percent / 100
    if test_size > 1:
        test_size = 0.3

    class_weight = None
    if args.balanced_weight:
        class_weight = "balanced"

    selected_features = args.features

    # build scoring function
    if args.scorer == "ber":  # BER-based scoring function
        cv_scorer = metrics.make_scorer(balanced_error_rate, greater_is_better=False)
    else:
        cv_scorer = args.scorer
        # cv_scorer = metrics.make_scorer(metrics.fbeta_score, beta=10.0)

    # load features
    x_data, y_data, x_info = load_data(n_jobs)
    print("Summary:\n", "# of segments:", len(x_data), "# of VT/Vf:", np.sum(y_data), len(x_info))

    # only select the specified feature
    if selected_features:
        x_data = x_data[:, selected_features]

    # scale the features
    preprocessing.scale(x_data, copy=False)
    x_indicies = list(range(0, len(x_data)))

    # build estimators to test
    estimator_name = args.model
    estimator = None
    param_grid = None
    support_class_weight = False
    if estimator_name == "logistic_regression":
        estimator = linear_model.LogisticRegression(class_weight=class_weight)
        param_grid = {
            "C": np.logspace(-4, 4, 10)
        }
        support_class_weight = True
    elif estimator_name == "random_forest":
        estimator = ensemble.RandomForestClassifier(class_weight=class_weight)
        param_grid = {
            "n_estimators": list(range(10, 110, 10))
        }
        support_class_weight = True
    elif estimator_name == "gradient_boosting":
        estimator = ensemble.GradientBoostingClassifier(learning_rate=0.1)
        param_grid = {
            "n_estimators": list(range(150, 250, 10)),
            "max_depth": list(range(3, 8))
        }
    elif estimator_name == "adaboost":
        estimator = ensemble.AdaBoostClassifier()
        param_grid = {
            "n_estimators": list(range(30, 150, 10)),
            "learning_rate": np.logspace(-1, 0, 2)
        }
    elif estimator_name == "svc":
        estimator = svm.SVC(shrinking=False,
                            cache_size=2048,
                            verbose=False,
                            probability=True,
                            class_weight=class_weight)
        param_grid = {
            "C": np.logspace(0, 1, 2),
            "gamma": np.logspace(-2, -1, 2)
        }
        support_class_weight = True

    # Run the selected test
    csv_fields = ["se", "sp", "ppv", "acc", "se(sp95)", "se(sp97)", "se(sp99)", "tp", "tn", "fp", "fn"]
    csv_fields.extend(sorted(param_grid.keys()))
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        # perform the test for many times
        for it in range(n_test_iters):
            print(estimator_name, it)
            row = {}
            # Here we split the indicies of the rows rather than the data array itself.
            x_train_idx, x_test_idx, y_train, y_test = cross_validation.train_test_split(x_indicies,
                                                                                         y_data,
                                                                                         test_size=test_size,
                                                                                         stratify=y_data)
            x_train = x_data[x_train_idx]
            x_test = x_data[x_test_idx]
            x_test_info = x_info[x_test_idx]

            fit_params = None
            # try to balance class weighting
            if args.balanced_weight and not support_class_weight:
                # perform sample weighting instead if the estimator does not support class weighting
                n_vf = np.sum(y_data)
                sample_ratio = (len(y_data) - n_vf) / n_vf  # non-vf/vf ratio
                fit_params = {
                    "sample_weight": np.array([sample_ratio if y == 1 else 1.0 for y in y_train])
                }

            grid = grid_search.GridSearchCV(estimator,
                                            param_grid,
                                            fit_params=fit_params,
                                            scoring=cv_scorer,
                                            n_jobs=n_jobs,
                                            cv=n_cv_folds,
                                            verbose=0)

            # perform the classification test
            grid.fit(x_train, y_train)
            y_predict = grid.predict(x_test)
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
                y_predict_scores = grid.predict_proba(x_test)[:, 1]
                false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(y_test, y_predict_scores)
                # find sensitivity at 95% specificity
                x = np.searchsorted(false_pos_rate, 0.05)
                row["se(sp95)"] = true_pos_rate[x]

                x = np.searchsorted(false_pos_rate, 0.03)
                row["se(sp97)"] = true_pos_rate[x]

                x = np.searchsorted(false_pos_rate, 0.01)
                row["se(sp99)"] = true_pos_rate[x]

            # best parameters of grid search
            row.update(grid.best_params_)

            print("  ", ", ".join(["{0}={1}".format(field, row[field]) for field in csv_fields]))
            writer.writerow(row)


if __name__ == "__main__":
    main()
