#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search
import vf_data
from vf_features import load_features
from vf_eval import *
import multiprocessing as mp
import csv
import argparse


# label the segment according to different problem definition
label_methods_desc = """
label methods:
0: binary => terminated with VF: 1, others: 0
1: binary => terminated with VF or VFL: 1, others: 0
2: binary => terminated with VF or VFL or VT: 1, others: 0
------------------------------------------
3: binary => has VF: 1, others: 0
4: binary => has VF or VFL: 1, others: 0
5: binary => has VF or VFL or VT: 1, others: 0
------------------------------------------
TODO (not implemented):
6: multi-class => VF+VFL: 1,VT: 2, others: 0
7: multi-class => VF: 1, VFL+VT: 2, others: 0
8: multi-class => VF: 1, VFL: 2, VT: 3, others: 0
"""


def make_label(info, label_method):
    label = 0
    if label_method == 0:
        label = 1 if info.terminating_rhythm == vf_data.RHYTHM_VF else 0
    elif label_method == 1:
        label = 1 if (info.terminating_rhythm == vf_data.RHYTHM_VF or info.terminating_rhythm == vf_data.RHYTHM_VFL) else 0
    elif label_method == 2:
        label = 1 if (info.terminating_rhythm == vf_data.RHYTHM_VF or info.terminating_rhythm == vf_data.RHYTHM_VFL or info.terminating_rhythm == vf_data.RHYTHM_VT) else 0
    if label_method == 3:
        label = 1 if info.has_vf else 0
    elif label_method == 4:
        label = 1 if (info.has_vf or info.has_vfl) else 0
    elif label_method == 5:
        label = 1 if (info.has_vf or info.has_vfl or info.has_vt) else 0
    """
    elif label_method == 3:
        if info.terminating_rhythm == vf_data.RHYTHM_VF or info.terminating_rhythm == vf_data.RHYTHM_VFL:
            label = 1
        elif info.terminating_rhythm == vf_data.RHYTHM_VT:
            label = 2
    elif label_method == 4:
        if info.terminating_rhythm == vf_data.RHYTHM_VF:
            label = 1
        elif info.terminating_rhythm == vf_data.RHYTHM_VFL or info.terminating_rhythm == vf_data.RHYTHM_VT:
            label = 2
    elif label_method == 5:
        if info.terminating_rhythm == vf_data.RHYTHM_VF:
            label = 1
        elif info.terminating_rhythm == vf_data.RHYTHM_VFL:
            label = 2
        elif info.terminating_rhythm == vf_data.RHYTHM_VT:
            label = 3
    """
    return label


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    # known estimators
    estimator_names = ("logistic_regression", "random_forest", "adaboost", "gradient_boosting", "svc", "mlp")
    parser.add_argument("-m", "--model", type=str, required=True, choices=estimator_names)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-j", "--jobs", type=int, default=-1)
    parser.add_argument("-t", "--iter", type=int, default=1)
    parser.add_argument("-s", "--scorer", type=str, choices=("ber", "f1", "accuracy", "precision"), default="ber")
    parser.add_argument("-c", "--cv-fold", type=int, default=10)  # 10 fold CV by default
    parser.add_argument("-p", "--test-percent", type=int, default=30)  # 30% test set size
    parser.add_argument("-b", "--balanced-weight", action="store_true")  # used balanced class weighting
    parser.add_argument("-f", "--features", type=int, nargs="+")  # feature selection
    parser.add_argument("-l", "--label-method", type=int, default=0, help=label_methods_desc)
    parser.add_argument("-x", "--exclude-noise", action="store_true", default=False)
    all_db_names = ("mitdb", "vfdb", "cudb")
    parser.add_argument("-d", "--db-names", type=str, nargs="+", choices=all_db_names, default=None)
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
    x_data, x_info = load_features(args.input)

    # exclude segments with noises if needed
    if args.exclude_noise:
        no_artifact_idx = np.array([i for i, info in enumerate(x_info) if not info.has_artifact])
        x_data = x_data[no_artifact_idx, :]
        x_info = x_info[no_artifact_idx]

    # if we only want data from some specified databases
    if args.db_names:
        include_idx = [i for i, info in enumerate(x_info) if info.record.split("/")[0] in args.db_names]
        x_data = x_data[include_idx, :]
        x_info = x_info[include_idx]

    # label the data
    y_data = np.array([make_label(info, args.label_method) for info in x_info])
    print("Summary:\n", "# of segments:", len(x_data), "# of VT/Vf:", np.sum(y_data), len(x_info))

    # only select the specified feature
    if selected_features:
        x_data = x_data[:, selected_features]

    x_indicies = list(range(len(x_data)))

    # build estimators to test
    estimator_name = args.model
    estimator = None
    param_grid = None
    support_class_weight = False
    if estimator_name == "logistic_regression":
        from sklearn import linear_model
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
        from sklearn import svm
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
    elif estimator_name == "mlp":  # multiple layer perceptron neural network
        from sknn import mlp
        layers = [
            mlp.Layer(type="Tanh", name="hidden0", units=100),
            mlp.Layer("Softmax")
        ]
        estimator = mlp.Classifier(layers=layers,
                                   n_iter=25)
        param_grid = {
            "learning_rate": np.logspace(-4, -3, 2),
            "regularize": ("l2", "dropout"),
            "n_iter": [25],
            "hidden0__units": list(range(10, 80, 10)),
            "hidden0__type": ("Rectifier", "Sigmoid", "Tanh")
        }

    # Run the selected test
    csv_fields = ["se", "sp", "ppv", "acc", "se(sp95)", "se(sp97)", "se(sp99)", "tp", "tn", "fp", "fn"]
    csv_fields.extend(sorted(param_grid.keys()))
    with open(args.output, "w", newline="", buffering=1) as f:  # buffering=1 means line buffering
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

            # scale the features (NOTE: training and testing sets should be scaled separately.)
            preprocessing.scale(x_train, copy=False)
            preprocessing.scale(x_test, copy=False)

            fit_params = None
            # try to balance class weighting
            if args.balanced_weight and not support_class_weight:
                # perform sample weighting instead if the estimator does not support class weighting
                n_vf = np.sum(y_data)
                sample_ratio = (len(y_data) - n_vf) / n_vf  # non-vf/vf ratio
                weight_arg = "sample_weight" if estimator_name != "mlp" else "w"
                fit_params = {
                    weight_arg: np.array([sample_ratio if y == 1 else 1.0 for y in y_train])
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
            if estimator_name == "mlp":  # sknn has different format of output and it needs to be flatten into a 1d array.
                y_predict = y_predict.flatten()

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
                false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(y_test, y_predict_scores, pos_label=1)
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
