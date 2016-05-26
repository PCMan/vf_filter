#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search
import vf_data
from vf_features import load_features, feature_names
from vf_eval import *
import multiprocessing as mp
import csv
import argparse
from array import array

# initial classification to detect possibly shockable rhythm
SAFE_RHYTHM = 0  # others
DANGEROUS_RHYTHM = 1  # VF or VT
binary_class_names = ["safe", "dangerous"]

# AHA clasases
NON_SHOCKABLE = 0
SHOCKABLE = 1
INTERMEDIATE = 2
aha_classes = (NON_SHOCKABLE, SHOCKABLE, INTERMEDIATE)
aha_classe_names = ["non-shockable", "shockable", "intermediate"]
shockable_rhythms = ("(VF", "(VT", "(VFL")

# Use threshold value: 180 BPM to define rapid VT
# Reference: Nishiyama et al. 2015. Diagnosis of Automated External Defibrillators (JAHA)
RAPID_VT_RATE = 180

# 0.2 mV is suggested by AHA
COARSE_VF_THRESHOLD = 0.2

def create_aha_labels(x_data, x_data_info):
    y_data = np.zeros(len(x_data_info), dtype="int")
    for i in range(len(x_data)):
        x = x_data[i]
        info = x_data_info[i]
        rhythm = info.rhythm
        # distinguish subtypes of VT and VF
        # References for the definition of "coarse":
        # 1. Foundations of Respiratory Care. by Kenneth A. Wyka，Paul J. Mathews，John Rutkowski
        #    Chapter 19. p.537
        #    Quote: "Coarse VF exists when wave amplitude is more than 3 mm."
        # 2. ECGs Made Easy by Barbara J Aehlert
        #    p.203
        #    Quote: "Coarse VF is 3 mm or more in amplitude. Fine VF is less than 3 mm in amplitude."
        # 3. In AHA recommendations for AED, a peak-to-peak amplitude of 0.2 mV is suggested.
        if rhythm == "(VF":
            if info.amplitude > COARSE_VF_THRESHOLD:  # coarse VF
                y_data[i] = SHOCKABLE
            else:  # fine VF
                y_data[i] = INTERMEDIATE
        elif rhythm in ("(VT", "(VFL"):
            # VFL is VF with HR > 240 BPM, so it's kind of rapid VT
            # However, in the dataset we found segments with slower heart rate
            # marked as VFL. So let's double check here
            hr = info.get_heart_rate()
            if hr >= RAPID_VT_RATE:
                y_data[i] = SHOCKABLE
            elif hr > 0:
                y_data[i] = INTERMEDIATE
            else:  # no heart rate information
                y_data[i] = SHOCKABLE if rhythm == "(VFL" else INTERMEDIATE
    return y_data


def create_binary_labels(x_data_info):
    y_data = np.zeros(len(x_data_info), dtype="int")
    for i, info in enumerate(x_data_info):
        if info.rhythm in shockable_rhythms:
            y_data[i] = DANGEROUS_RHYTHM 
    return y_data


def exclude_rhythms(x_data, x_data_info, excluded_rhythms):
    excluded_idx = np.array([i for i, info in enumerate(x_data_info) if info.rhythm in excluded_rhythms])
    x_data = np.delete(x_data, excluded_idx, axis=0)
    x_data_info = np.delete(x_data_info, excluded_idx, axis=0)
    return x_data, x_data_info


def get_sample_weights(y_data):
    classes = np.unique(y_data)
    n_classes = [np.sum([y_data == k]) for k in classes]
    n_total = len(y_data)
    weights = np.zeros(y_data.shape)
    for k, n in zip(classes, n_classes):
        weights[y_data == k] = (n_total / n)
    return weights


def create_estimator(estimator_name, class_weight):
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
        import xgboost.sklearn as xgb
        estimator = xgb.XGBClassifier(learning_rate=0.1)
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
    elif estimator_name == "mlp1" or estimator_name == "mlp2":  # multiple layer perceptron neural network
        from sknn import mlp
        param_grid = {
            "learning_rate": [0.0001],
            "regularize": ["l2"],  # , "dropout"],
            "weight_decay": np.logspace(-6, -5, 2),  # parameter for L2 regularizer
            "hidden0__type": ["Tanh"]  # "Rectifier", "Sigmoid"
        }

        layers = [mlp.Layer(type="Tanh", name="hidden0")]
        # add the second hidden layer as needed
        if estimator_name == "mlp2":  # 2 hidden layer
            layers.append(mlp.Layer(type="Tanh", name="hidden1"))
            param_grid["hidden0__units"] = list(range(2, 5, 1))
            param_grid["hidden1__units"] = list(range(2, 5, 1))
            param_grid["hidden1__type"] = ["Tanh"]  # "Rectifier", "Sigmoid"
        else:
            param_grid["hidden0__units"] = list(range(5, 26, 1))
        # add the output layer
        layers.append(mlp.Layer("Softmax"))
        estimator = mlp.Classifier(layers=layers, batch_size=150)

    return estimator, param_grid, support_class_weight


def create_csv_fields(label_encoder):
    csv_fields = ["iter"]
    for class_name in binary_class_names:
        csv_fields.extend(["{0}[{1}]".format(field, class_name) for field in ("Se", "Sp", "precision")])
    for class_name in aha_classe_names:
        csv_fields.extend(["{0}[{1}]".format(field, class_name) for field in ("Se", "Sp", "precision")])
    for class_name in label_encoder.classes_:
        if class_name in shockable_rhythms:  # shockable rhythms
            csv_fields.append("Se[{0}]".format(class_name))
        else:  # other non-shockable rhythms
            csv_fields.append("Sp[{0}]".format(class_name))
    return csv_fields


# final classification based on AHA requirements
def aha_classifier(x_test, x_test_info, binary_y_predict):
    aha_y_predict = np.zeros(len(x_test), dtype="int")
    for i in range(len(x_test)):
        # Check if this rhythm is one of VF, VT, or VFL (dangerous rhythms)
        if binary_y_predict[i] == DANGEROUS_RHYTHM:
            info = x_test_info[i]
            # This rhythm can be VF, VFL, or VT, but we don't know
            # TODO: we may use some simple features to distinguish them if needed
            amplitude = info.amplitude
            # perform QRS detection to calculate heart rate
            # here we get the stored QRS detection result done previously for speed up.
            beats = info.detected_beats
            if beats:  # heart beats are detected
                # This can be VF (shockable), slow VT (intermediate), or other misclassified "safe" rhythms
                # FIXME: find a method to distinguish them
                hr = (len(beats) / info.get_duration()) * 60  # average HR (BPM)
                rr_average, rr_std, abnormal_beats = beat_statistics(info.detected_beats)
                rr_cv = rr_std / rr_average if rr_average else 0.0
                print(info.rhythm, "HR:", hr, "RR:", rr_average, "RR_std", rr_std, "RR_CV:", rr_cv,
                      "abnormal ratio:", abnormal_beats,
                      "IMF1:", x_test[i][-2], "IMF5:", x_test[i][-1])
                print("\t", info.detected_beats)
                if hr >= RAPID_VT_RATE:  # this is either rapid VT or VF (both are shockable, no need to distinguish them)
                    y = SHOCKABLE
                else:  # this rhythm is slower than 180 BPM
                    y = INTERMEDIATE
            else:  # no QRS complex was found, this must be VF or asystole
                if amplitude >= COARSE_VF_THRESHOLD:
                    y = SHOCKABLE
                else:
                    y = INTERMEDIATE
            aha_y_predict[i] = y
    return aha_y_predict


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    # known estimators
    estimator_names = ("logistic_regression", "random_forest", "adaboost", "gradient_boosting", "svc", "mlp1", "mlp2")
    parser.add_argument("-m", "--model", type=str, required=True, choices=estimator_names)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-e", "--error-log", type=str, help="filename of the error log")
    parser.add_argument("-j", "--jobs", type=int, default=-1)
    parser.add_argument("-t", "--iter", type=int, default=1)
    parser.add_argument("-s", "--scorer", type=str, choices=("ber", "f1", "accuracy", "precision", "f1_weighted"), default="f1_weighted")
    parser.add_argument("-c", "--cv-fold", type=int, default=5)  # 5 fold CV by default
    parser.add_argument("-p", "--test-percent", type=int, default=30)  # 30% test set size
    parser.add_argument("-b", "--balanced-weight", action="store_true")  # used balanced class weighting
    parser.add_argument("-f", "--features", type=str, nargs="+", choices=feature_names, default=[])  # feature selection
    parser.add_argument("-l", "--label-method", type=str, default="aha")
    parser.add_argument("-x", "--exclude-rhythms", type=str, nargs="+", default=["(ASYS"])  # exclude some rhythms from the test
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
    estimator_name = args.model

    class_weight = None
    if args.balanced_weight:
        class_weight = "balanced"

    selected_features = [feature_names.index(name) for name in args.features]  # convert feature names to indices

    # build scoring function
    if args.scorer == "ber":  # BER-based scoring function
        cv_scorer = metrics.make_scorer(balanced_error_rate, greater_is_better=False)
    else:
        cv_scorer = args.scorer
        # cv_scorer = metrics.make_scorer(metrics.fbeta_score, beta=10.0)

    # load features
    x_data, x_data_info = load_features(args.input)
    # only select the specified feature
    if selected_features:
        x_data = x_data[:, selected_features]

    # "X" is used internally by us (in label correction file) to mark some broken samples to exclude from the test
    excluded_rhythms = ["X"]
    if args.exclude_rhythms:
        excluded_rhythms.extend(args.exclude_rhythms)
    # exclude samples with some rhythms from the test
    x_data, x_data_info = exclude_rhythms(x_data, x_data_info, excluded_rhythms)

    # encode differnt types of rhythm names into numeric codes for stratified sampling later
    y_rhythm_names = [info.rhythm for info in x_data_info]
    label_encoder = preprocessing.LabelEncoder()
    x_rhythm_types = label_encoder.fit_transform(y_rhythm_names)

    # label the samples
    binary_y_data = create_binary_labels(x_data_info)
    aha_y_data = create_aha_labels(x_data, x_data_info)

    # try to classify the samples using a beat detector and simple heuristics
    # y_basic_predict = basic_classify(x_data, x_data_info)

    # How many samples cannot be detected with QRS detector??
    n_no_qrs = 0
    for info in x_data_info:
        if not info.detected_beats:
            print("No QRS!!", info.rhythm, info.record_name, info.begin_time)
            n_no_qrs += 1
    print(n_no_qrs, "samples has no detected QRS.")

    # build estimator to test
    estimator, param_grid, support_class_weight = create_estimator(estimator_name, class_weight)

    csv_fields = create_csv_fields(label_encoder)  # generate field names for the output csv file

    # prepare a matrix to store the error states of each sample during test iterations
    predict_results = None
    if args.error_log:  # output error log file
        # 2D table to store test results: initialize all fields with -1
        predict_results = np.full(shape=(len(x_data), n_test_iters), fill_value=-1, dtype=int)

    # also report the optimal parameters after tuning with CV to the csv file
    csv_fields.extend(sorted(param_grid.keys()))
    with open(args.output, "w", newline="", buffering=1) as f:  # buffering=1 means line buffering
        rows = []
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        # perform the test for many times
        for it in range(1, n_test_iters + 1):
            print(estimator_name, it)
            row = {"iter" : it}
            # generate test for this iteration
            # Here we split the indicies of the rows rather than the data array itself.
            x_indicies = list(range(len(x_data)))
            x_train_idx, x_test_idx, y_train, y_test = cross_validation.train_test_split(x_indicies,
                                                                                         binary_y_data,
                                                                                         test_size=test_size,
                                                                                         stratify=aha_y_data)  # stratify=x_rhythm_types does not work due to low number of some classes. :-(
            # training dataset
            x_train = x_data[x_train_idx]
            # testing dataset
            x_test = x_data[x_test_idx]
            x_test_info = x_data_info[x_test_idx]

            # scale the features (NOTE: training and testing sets should be scaled separately.)
            preprocessing.scale(x_train, copy=False)
            preprocessing.scale(x_test, copy=False)

            fit_params = None
            # try to balance class weighting
            if args.balanced_weight and not support_class_weight:
                # perform sample weighting instead if the estimator does not support class weighting
                weight_arg = "w" if estimator_name.startswith("mlp") else "sample_weight"
                fit_params = {
                    weight_arg: np.array(get_sample_weights(y_train))
                }

            # find best parameters using grid search + cross validation
            grid = grid_search.GridSearchCV(estimator,
                                            param_grid,
                                            fit_params=fit_params,
                                            scoring=cv_scorer,
                                            n_jobs=n_jobs,
                                            cv=n_cv_folds,
                                            verbose=0)
            grid.fit(x_train, y_train)  # perform the classification training
            y_predict = grid.predict(x_test)

            if args.error_log:  # calculate error statistics for each sample
                included_idx = np.array(x_test_idx)  # samples included in this test iteration
                predict_results[included_idx, it - 1] = y_predict  # remember prediction results of all included samples

            # output the result of classification to csv file
            results = MultiClassificationResult(y_test, y_predict, classes=range(len(binary_class_names))).results
            for class_name, result in zip(binary_class_names, results):
                row["Se[{0}]".format(class_name)] = result.sensitivity
                row["Sp[{0}]".format(class_name)] = result.specificity
                row["precision[{0}]".format(class_name)] = result.precision

            # perform final classification based on AHA classification scheme
            y_train = aha_y_data[x_train_idx]
            y_test = aha_y_data[x_test_idx]  # the actual AHA class
            grid.fit(x_train, y_train)  # perform the classification training
            y_predict = grid.predict(x_test)
            # y_predict = aha_classifier(x_test, x_test_info, y_predict)  # the predicted AHA class

            # report the performance of final AHA classification
            results = MultiClassificationResult(y_test, y_predict, classes=aha_classes).results
            for class_name, result in zip(aha_classe_names, results):
                row["Se[{0}]".format(class_name)] = result.sensitivity
                row["Sp[{0}]".format(class_name)] = result.specificity
                row["precision[{0}]".format(class_name)] = result.precision

            # report performance for each rhythm type (suggested by AHA guideline for AED development)
            for rhythm_id, rhythm_name in enumerate(label_encoder.classes_):
                y_test_rhythm_types = x_rhythm_types[x_test_idx]
                idx = (y_test_rhythm_types == rhythm_id)
                rhythm_y_test = y_test[idx]
                rhythm_y_predict = y_predict[idx]
                # convert to binary classification for each type of arrythmia
                bin_y_test = np.zeros((len(rhythm_y_test), 1))
                bin_y_predict = np.zeros((len(rhythm_y_predict), 1))
                if rhythm_name in shockable_rhythms:  # for shockable rhythms, report sensitivity (TPR)
                    if rhythm_name == "(VF,coarse" or rhythm_name == "(VT,rapid":
                        target = SHOCKABLE
                    else:
                        target = INTERMEDIATE
                    bin_y_test[rhythm_y_test == target] = 1
                    bin_y_predict[rhythm_y_predict == target] = 1
                    result = BinaryClassificationResult(bin_y_test, bin_y_predict)
                    row["Se[{0}]".format(rhythm_name)] = result.sensitivity
                else:  # for non-shockable rhythms, report specificity (TNR)
                    bin_y_test[rhythm_y_test == NON_SHOCKABLE] = 1
                    bin_y_predict[rhythm_y_predict == NON_SHOCKABLE] = 1
                    result = BinaryClassificationResult(bin_y_test, bin_y_predict)
                    row["Sp[{0}]".format(rhythm_name)] = result.sensitivity

            # best parameters of grid search
            row.update(grid.best_params_)
            rows.append(row)  # remember each row so we can calculate average for them later
            print("\n".join(["\t{0} = {1}".format(field, row.get(field, 0.0)) for field in csv_fields[1:]]))
            writer.writerow(row)
            print("-" * 80)

        # calculate average for all iterations automatically and write to csv
        n_params = len(param_grid)
        fields = csv_fields[1:-n_params]
        avg = {"iter": "average"}
        for field in fields:
            col = [row[field] for row in rows]
            avg[field] = np.mean(col)
        writer.writerow(avg)

        # log prediction errors of each sample during the test iterations in a csv file.
        if args.error_log:
            with open(args.error_log, "w", newline="") as f:
                writer = csv.writer(f)
                fields = ["sample", "record", "begin", "rhythm"] + [str(i) for i in range(1, n_test_iters + 1)] + ["tested", "errors", "error rate", "class", "predict"]
                writer.writerow(fields)  # write header for csv
                for i, info in enumerate(x_data_info):
                    label = aha_y_data[i]
                    x_predict_results = predict_results[i, :]  # prediction results of all iterations for this sample x
                    row = [(i + 1), info.record_name, info.begin_time, info.rhythm]
                    n_included = 0
                    n_errors = 0
                    for y_predict in x_predict_results:
                        if y_predict == -1:  # not included in this iteration of test
                            row.append("")
                        else:
                            n_included += 1
                            row.append(y_predict)
                            if y_predict != label:  # prediction error in this iteration
                                n_errors += 1
                    row.append(n_included)  # number of test iterations in which this sample x is included
                    row.append(n_errors)  # number of errors
                    row.append(n_errors / n_included if n_included > 0 else "N/A")  # error rate for this sample
                    row.append(label)  # correct class of the sample
                    # most frequently predicted class
                    if n_included > 0:
                        classes, freq = np.unique(x_predict_results[x_predict_results != -1], return_counts=True)
                        # find the prediction result with highest frequency
                        most_frequent_predict = classes[np.argmax(freq)]
                    else:  # the sample is excluded from all of the test iterations.
                        most_frequent_predict = "N/A"
                    row.append(most_frequent_predict)
                    writer.writerow(row)

if __name__ == "__main__":
    main()
