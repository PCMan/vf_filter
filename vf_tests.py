#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import feature_selection
from vf_features import load_features, feature_names
import vf_eval
import csv
import argparse
import vf_classify


def create_csv_fields(estimator_name, select_feature_names, label_encoder, param_grid):
    csv_fields = ["iter"]
    csv_fields.extend(["{0}[{1}]".format(field, "dangerous") for field in ("Se", "Sp", "precision")])
    for class_name in vf_classify.aha_classe_names:
        csv_fields.extend(["{0}[{1}]".format(field, class_name) for field in ("Se", "Sp", "precision")])
    for class_name in label_encoder.classes_:
        if class_name in vf_classify.shockable_rhythms:  # shockable rhythms
            csv_fields.append("Se[{0}]".format(class_name))
        else:  # other non-shockable rhythms
            csv_fields.append("Sp[{0}]".format(class_name))

    if estimator_name in ("random_forest", "adaboost", "gradient_boosting"):
        # feature scores, if applicable
        csv_fields.extend(select_feature_names)
    elif estimator_name in ("logistic_regression", "svc_linear"):
        # coefficients for regression, if applicable
        for class_id in range(0, 3):
            csv_fields.extend(["{0}[{1}]".format(feature, class_id) for feature in select_feature_names])

    # remember best parameters
    csv_fields.extend(sorted(param_grid.keys()))
    return csv_fields


def output_binary_result(row, y_test, y_predict):
    result = vf_eval.BinaryClassificationResult(y_test, y_predict)
    row["Se[dangerous]"] = result.sensitivity
    row["Sp[dangerous]"] = result.specificity
    row["precision[dangerous]"] = result.precision


def output_aha_result(row, x_test_idx, y_test, y_predict, label_encoder, x_rhythm_types):
    results = vf_eval.MultiClassificationResult(y_test, y_predict, classes=vf_classify.aha_classes).results
    for class_name, result in zip(vf_classify.aha_classe_names, results):
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
        if rhythm_name in vf_classify.shockable_rhythms:  # for shockable rhythms, report sensitivity (TPR)
            if rhythm_name == "(VF,coarse" or rhythm_name == "(VT,rapid":
                target = vf_classify.SHOCKABLE
            else:
                target = vf_classify.INTERMEDIATE
            bin_y_test[rhythm_y_test == target] = 1
            bin_y_predict[rhythm_y_predict == target] = 1
            result = vf_eval.BinaryClassificationResult(bin_y_test, bin_y_predict)
            row["Se[{0}]".format(rhythm_name)] = result.sensitivity
        else:  # for non-shockable rhythms, report specificity (TNR)
            bin_y_test[rhythm_y_test == vf_classify.NON_SHOCKABLE] = 1
            bin_y_predict[rhythm_y_predict == vf_classify.NON_SHOCKABLE] = 1
            result = vf_eval.BinaryClassificationResult(bin_y_test, bin_y_predict)
            row["Sp[{0}]".format(rhythm_name)] = result.sensitivity


def output_best_params(row, grid_search_cv):
    for key, val in grid_search_cv.best_params_.items():
        if key.startswith("estimator__"):
            key = key[11:]
        row[key] = val
        # row.update(grid_search_cv.best_params_)


def output_feature_scores(row, estimator, selected_feature_names):
    # feature importance
    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
        # normalize the scores
        for i, score in enumerate(scores):
            name = selected_feature_names[i]
            row[name] = score
    else:
        try:
            if hasattr(estimator, "coef_"):
                coef = estimator.coef_  # coefficients for each class
                for class_id in range(0, 3):
                    weights = coef[class_id, :]
                    for i, weight in enumerate(weights):
                        row["{0}[{1}]".format(selected_feature_names[i], class_id)] = weight
        except ValueError:  # non-linear estimators do not support coef_
            pass


def output_errors(log_filename, predict_results, x_data_info, aha_y_data):
    n_test_iters = predict_results.shape[1]
    with open(log_filename, "w", newline="") as f:
        writer = csv.writer(f)
        fields = ["sample", "record", "begin", "rhythm"]
        fields.extend([str(i) for i in range(1, n_test_iters + 1)])
        fields.extend(["tested", "errors", "error rate", "class", "predict"])
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


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, choices=vf_classify.estimator_names)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-e", "--error-log", type=str, help="filename of the error log")
    parser.add_argument("-j", "--jobs", type=int, default=-1)
    parser.add_argument("-t", "--iter", type=int, default=1)
    parser.add_argument("-s", "--scorer", type=str, choices=vf_eval.scorer_names, default="f1_weighted")
    parser.add_argument("-c", "--cv-fold", type=int, default=5)  # 5 fold CV by default
    parser.add_argument("-p", "--test-percent", type=int, default=30)  # 30% test set size
    parser.add_argument("-w", "--unbalanced-weight", action="store_true")  # avoid balanced class weighting
    parser.add_argument("-f", "--features", type=str, nargs="+", choices=feature_names)  # feature selection
    parser.add_argument("-x", "--exclude-rhythms", type=str, nargs="+", default=["(ASYS"])  # exclude some rhythms from the test
    parser.add_argument("-r", "--rfe-n-features", type=int, default=None)  # recursive feature elimination
    args = parser.parse_args()
    print(args)

    # setup testing parameters
    n_jobs = args.jobs
    n_test_iters = args.iter
    n_cv_folds = args.cv_fold
    test_size = args.test_percent / 100
    if test_size > 1:
        test_size = 0.3
    estimator_name = args.model

    class_weight = "balanced"
    if args.unbalanced_weight:
        class_weight = None

    # build scoring function
    cv_scorer = vf_eval.get_scorer(args.scorer)

    # load features
    x_data, x_data_info = load_features(args.input)

    # only select the specified feature
    if args.features:
        selected_features = [feature_names.index(name) for name in args.features]  # convert feature names to indices
        selected_feature_names = args.features
        x_data = x_data[:, selected_features]
    else:
        selected_features = list(range(len(feature_names)))
        selected_feature_names = feature_names

    if args.exclude_rhythms:  # exclude samples with some rhythms from the test
        x_data, x_data_info = vf_classify.exclude_rhythms(x_data, x_data_info, args.exclude_rhythms)

    # label the samples for AHA AED recommendation based multiclass scheme
    aha_y_data = vf_classify.initialize_aha_labels(x_data, x_data_info)
    # simplify the multi-class scheme to binary classification
    binary_y_data = (aha_y_data != vf_classify.NON_SHOCKABLE).astype("int")

    # encode differnt types of rhythm names into numeric codes for stratified sampling later
    y_rhythm_names = [info.rhythm for info in x_data_info]
    label_encoder = preprocessing.LabelEncoder()
    x_rhythm_types = label_encoder.fit_transform(y_rhythm_names)

    # build estimator to test
    estimator, param_grid, support_class_weight = vf_classify.create_estimator(estimator_name, class_weight)

    # prepare a matrix to store the error states of each sample during test iterations
    predict_results = None
    if args.error_log:  # output error log file
        # 2D table to store test results: initialize all fields with -1
        predict_results = np.full(shape=(len(x_data), n_test_iters), fill_value=-1, dtype=int)

    # also report the optimal parameters after tuning with CV to the csv file
    csv_fields = create_csv_fields(estimator_name, selected_feature_names, label_encoder, param_grid)  # generate field names for the output csv file

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
            if class_weight == "balanced" and not support_class_weight:
                # perform sample weighting instead if the estimator does not support class weighting
                weight_arg = "w" if estimator_name.startswith("mlp") else "sample_weight"
                fit_params = {
                    weight_arg: np.array(vf_classify.get_balanced_sample_weights(y_train))
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
            # output the result of classification to csv file
            output_binary_result(row, y_test, y_predict)

            # perform multiclass classification based on AHA classification scheme
            y_train = aha_y_data[x_train_idx]
            surviving_features = list(range(x_train.shape[1]))
            feature_ranking = []
            while True:  # loop for recursive feature elimination (RFE)
                # Because of a bug in joblib, we see a lot of warnings here.
                # https://github.com/scikit-learn/scikit-learn/issues/6370
                # Use the workaround to turn off the warnings
                import warnings
                warnings.filterwarnings("ignore")

                x_train_selected = x_train[:, surviving_features] # only select a subset of features for training
                grid.fit(x_train_selected, y_train)  # perform the classification training
                best_estimator = grid.best_estimator_  # now the estimator is trained and optimized
                if args.rfe_n_features:  # we want to perform RFE
                    if len(surviving_features) <= args.rfe_n_features:  # we already eliminated all of the features
                        break

                    # find the worst feature in this round (lowest score/coefficient)
                    if hasattr(best_estimator, 'coef_'):
                        coefs = best_estimator.coef_
                    elif hasattr(best_estimator, 'feature_importances_'):
                        coefs = best_estimator.feature_importances_
                    else:  # no feature importance scores for ranking.
                        break
                    ranking_scores = coefs ** 2
                    if coefs.ndim > 1:
                        ranking_scores = ranking_scores.sum(axis=0)
                    i_min_score = np.argmin(ranking_scores)  # find worst feature
                    worst_feature = surviving_features[i_min_score]  # find worst feature
                    del surviving_features[i_min_score] # eliminate the worst feature in this round
                    feature_ranking.append(worst_feature)
                    print("worst feature:", selected_feature_names[worst_feature], ", CV score:", grid.best_score_, grid.best_params_)
                    print("\tsurviviing_features:", [selected_feature_names[i] for i in surviving_features])
                else:  # we don't want RFE, quit the loop directly
                    break
            # perform prediction with the selected features
            y_test = aha_y_data[x_test_idx]  # the actual AHA class
            y_predict = best_estimator.predict(x_test[:, surviving_features])

            if args.error_log:  # calculate error statistics for each sample
                included_idx = np.array(x_test_idx)  # samples included in this test iteration
                predict_results[included_idx, it - 1] = y_predict  # remember prediction results of all included samples

            # report the performance of final AHA classification
            output_aha_result(row, x_test_idx, y_test, y_predict, label_encoder, x_rhythm_types)

            # store best parameters of grid search
            output_best_params(row, grid)

            output_feature_scores(row, grid.best_estimator_, selected_feature_names)

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
        print("\n".join(["\t{0} = {1}".format(field, avg.get(field, 0.0)) for field in csv_fields[1:]]))

        # log prediction errors of each sample during the test iterations in a csv file.
        if args.error_log:
            output_errors(args.error_log, predict_results, x_data_info, aha_y_data)


if __name__ == "__main__":
    main()
