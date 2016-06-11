#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.base import clone
from vf_features import load_features, feature_names
import vf_eval
import csv
import argparse
import vf_classify


def has_coefficients(estimator_name):
    return estimator_name in ("logistic_regression", "svc_linear")


def has_feature_importances(estimator_name):
    return estimator_name in ("random_forest", "adaboost", "gradient_boosting")


def create_csv_fields(estimator_name, select_feature_names, label_encoder, param_grid, feature_ranks):
    csv_fields = ["iter"]

    # fields for reporting AHA performance results
    csv_fields.extend(["AHA_Se[shockable]", "AHA_Sp[non_shockable]", "AHA_precision[shockable]"])
    for class_name in label_encoder.classes_:
        if class_name in vf_classify.shockable_rhythms:  # shockable rhythms
            csv_fields.append("AHA_Se[{0}]".format(class_name))
        elif class_name in vf_classify.intermediate_rhythms:  # intermediate rhythms
            pass  # FIXME: how to report the results for intermediate class?
        else:  # other non-shockable rhythms, report specificity
            csv_fields.append("AHA_Sp[{0}]".format(class_name))

    # fields for detailed multi-class results
    for class_name in vf_classify.aha_classe_names:
        csv_fields.extend(["{0}[{1}]".format(field, class_name) for field in ("Se", "Sp", "precision")])
    for class_name in label_encoder.classes_:
        # report sensitivity for shockable rhythms or intermediate rhythms
        if class_name in vf_classify.shockable_rhythms or class_name in vf_classify.intermediate_rhythms:
            csv_fields.append("Se[{0}]".format(class_name))
        else:  # other non-shockable rhythms
            csv_fields.append("Sp[{0}]".format(class_name))

    # fields for feature importance/ranking
    if has_feature_importances(estimator_name):
        csv_fields.extend(select_feature_names)
    elif has_coefficients(estimator_name):
        for class_id in range(0, 3):
            csv_fields.extend(["{0}[{1}]".format(feature, class_id) for feature in select_feature_names])
    if feature_ranks:
        csv_fields.extend(["rank[{0}]".format(feature) for feature in select_feature_names])

    # remember best parameters and CV score
    csv_fields.append("cv_score")
    csv_fields.append("testing_score")
    csv_fields.extend(sorted(param_grid.keys()))
    return csv_fields


def output_multiclass_result(row, x_test_idx, y_test, y_predict, label_encoder, x_rhythm_types):
    results = vf_eval.MultiClassificationResult(y_test, y_predict, classes=vf_classify.aha_classes).results
    for class_name, result in zip(vf_classify.aha_classe_names, results):
        print("class:", class_name, "tp=", result.tp, "tn=", result.tn, "fp=", result.fp, "fn=", result.fn, "total=", len(y_test))
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
        # for shockable rhythms, report sensitivity (TPR)
        if rhythm_name in vf_classify.shockable_rhythms or rhythm_name in vf_classify.intermediate_rhythms:
            if rhythm_name in vf_classify.shockable_rhythms:
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


def output_aha_result(row, x_test_idx, y_test, y_predict, label_encoder, x_rhythm_types):
    # get actual rhythm types for the test samples
    y_test_rhythm_types = x_rhythm_types[x_test_idx]

    # AHA asks for high Se for shockable rhythms (coarse VF and rapid VT) and
    # high Sp for non-shockable rhythms.
    # Only shockable and non-shockable rhythms are included in the performance evaluation.
    # The intermediate group excluded, but its performance may still be reported.
    exclude_intermediate_mask = (y_test != vf_classify.INTERMEDIATE)
    # now we only have SHOCKABLE and NON_SHOCKABLE in the test and predicted data
    binary_y_test = y_test[exclude_intermediate_mask]
    binary_y_predict = y_predict[exclude_intermediate_mask]
    # When evaluating AHA binary classification, the labels are: "shock" and "no shock"
    # So when a sample is predicted as intermediate, encode it to "no shock"
    binary_y_predict[binary_y_predict == vf_classify.INTERMEDIATE] = vf_classify.NON_SHOCKABLE
    results = vf_eval.BinaryClassificationResult(binary_y_test, binary_y_predict)
    print("AHA:", "tp=", results.tp, "tn=", results.tn, "fp=", results.fp, "fn=", results.fn, "total=", len(binary_y_test))
    row["AHA_Se[shockable]"] = results.sensitivity
    row["AHA_Sp[non_shockable]"] = results.specificity
    row["AHA_precision[shockable]"] = results.precision
    binary_y_test_rhythm_types = y_test_rhythm_types[exclude_intermediate_mask]  # exclude the intermediate class

    # FIXME: How to report performance for intermediate class?
    # report performance for each rhythm type (suggested by AHA guideline for AED development)
    for rhythm_id, rhythm_name in enumerate(label_encoder.classes_):
        # only evaluate test samples having this rhythm type
        rhythm_mask = (binary_y_test_rhythm_types == rhythm_id)
        rhythm_y_test = binary_y_test[rhythm_mask]
        rhythm_y_predict = binary_y_predict[rhythm_mask]
        results = vf_eval.BinaryClassificationResult(rhythm_y_test, rhythm_y_predict)
        # convert to binary classification for each type of arrythmia
        if rhythm_name in vf_classify.shockable_rhythms:  # for shockable rhythms, report sensitivity (TPR)
            row["AHA_Se[{0}]".format(rhythm_name)] = results.sensitivity
        elif rhythm_name in vf_classify.intermediate_rhythms:
            pass  # FIXME: how to report performanc for fine VF and slow VT?
        else:  # for non-shockable rhythms, report specificity (TNR)
            row["AHA_Sp[{0}]".format(rhythm_name)] = results.specificity


def output_best_params(row, best_params):
    for key, val in best_params.items():
        if key.startswith("estimator__"):
            key = key[11:]
        row[key] = val


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


# estimator must be a fitted estimator which contains coef_ property.
def find_worst_feature(estimator, feature_ids):
    # find the worst feature in this round (lowest score/coefficient)
    if hasattr(estimator, 'coef_'):
        coefs = estimator.coef_
    elif hasattr(estimator, 'feature_importances_'):
        coefs = estimator.feature_importances_
    else:  # no feature importance scores for ranking.
        return -1
    ranking_scores = coefs ** 2
    if coefs.ndim > 1:
        ranking_scores = ranking_scores.sum(axis=0)
    i_min_score = np.argmin(ranking_scores)  # find worst feature
    worst_feature = feature_ids[i_min_score]  # find worst feature
    return worst_feature, i_min_score


def output_feature_ranks(row, eliminated_features, selected_feature_names):
    for order, feature_id in enumerate(reversed(eliminated_features)):
        name = "rank[{0}]".format(selected_feature_names[feature_id])
        row[name] = order + 1


def calculate_average(rows, csv_fields, param_grid):
    n_params = len(param_grid)
    fields = csv_fields[1:-n_params]
    avg = {"iter": "average"}
    for field in fields:
        col = [row.get(field, 0.0) for row in rows]
        mean = np.mean(col)
        if field.startswith("Se") or field.startswith("Sp") or field.startswith("precision"):
            mean = "{0:.2f}%".format(mean * 100)
        avg[field] = mean
    return avg


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, choices=vf_classify.estimator_names)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-e", "--error-log", type=str, help="filename of the error log")
    parser.add_argument("-j", "--jobs", type=int, default=-1)
    parser.add_argument("-t", "--iter", type=int, default=1)
    parser.add_argument("-s", "--scorer", type=str, choices=vf_eval.scorer_names, default="f1_macro")
    parser.add_argument("-c", "--cv-fold", type=int, default=5)  # 5 fold CV by default
    parser.add_argument("-p", "--test-percent", type=int, default=30)  # 30% test set size
    parser.add_argument("-w", "--unbalanced-weight", action="store_true")  # avoid balanced class weighting
    parser.add_argument("-f", "--features", type=str, nargs="+", choices=feature_names)  # feature selection
    parser.add_argument("-x", "--exclude-rhythms", type=str, nargs="+", default=["(ASYS"])  # exclude some rhythms from the test
    parser.add_argument("-r", "--perform-rfe", action="store_true")  # recursive feature elimination
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

    if args.perform_rfe:  # we want to perform RFE
        if not has_coefficients(estimator_name) and not has_feature_importances(estimator_name):
            print("Recursive feature elimination is not supported for this model.")
            args.perform_rfe = False

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

    # generate field names for the output csv file
    csv_fields = create_csv_fields(estimator_name, selected_feature_names, label_encoder, param_grid, args.perform_rfe)
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

            # scale the features (NOTE: training and testing sets should be scaled by the same factor.)
            # scale to [-1, 1] (or scale to [0, 1]. scaling is especially needed by SVM)
            data_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
            data_scaler.fit_transform(x_train)
            data_scaler.transform(x_test)  # the test dataset should be scaled by the same factor

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

            # multiclass classification based on AHA classification scheme (non-shockable, shockable, intermediate)
            y_train = aha_y_data[x_train_idx]  # labels of the training set
            y_test = aha_y_data[x_test_idx]  # labels of the testing set
            surviving_features = list(range(x_train.shape[1]))
            eliminated_features = []
            best_estimator = None
            best_params = None
            best_cv_score = 0.0
            best_test_score = 0.0
            while True:  # loop for recursive feature elimination (RFE)
                # Because of a bug in joblib, we see a lot of warnings here.
                # https://github.com/scikit-learn/scikit-learn/issues/6370
                # Use the workaround to turn off the warnings
                import warnings
                warnings.filterwarnings("ignore")

                x_train_selected = x_train[:, surviving_features]  # only select a subset of features for training
                grid.fit(x_train_selected, y_train)  # perform the classification training
                if not best_estimator:  # train the estimator for the first time
                    best_estimator = grid.best_estimator_  # now the estimator is trained and optimized
                    best_cv_score = grid.best_score_
                    best_test_score = grid.score(x_test, y_test)
                    y_predict = best_estimator.predict(x_test)  # predict the test data set

                rfe_estimator = grid.best_estimator_  # estimator used for RFE
                cv_score = grid.best_score_  # cross-validation score
                best_params = grid.best_params_  # best parameter found using grid search

                # testing with currently selected features
                test_score = rfe_estimator.score(x_test[:, surviving_features], y_test)
                rfe_y_predict = rfe_estimator.predict(x_test[:, surviving_features])
                rfe_row = {}
                output_aha_result(rfe_row, x_test_idx, y_test, rfe_y_predict, label_encoder, x_rhythm_types)
                output_multiclass_result(rfe_row, x_test_idx, y_test, rfe_y_predict, label_encoder, x_rhythm_types)
                print("cv_score:", cv_score, ", test_score:", test_score)
                print("\n".join(["\t{0} = {1}".format(field, rfe_row.get(field, 0.0)) for field in csv_fields[1:]]))

                if args.perform_rfe and surviving_features:  # we want to perform RFE
                    # eliminate the worst feature in this round (lowest score/coefficient)
                    worst_feature, i_worst_feature = find_worst_feature(rfe_estimator, surviving_features)
                    del surviving_features[i_worst_feature]  # eliminate the worst feature in this round
                    eliminated_features.append(worst_feature)
                    print(selected_feature_names[worst_feature], [selected_feature_names[i] for i in surviving_features])
                    if not surviving_features:  # we already eliminated all of the features
                        break
                else:  # we don't want RFE at all, quit the loop directly
                    break

            # generate ranking for features based on the order of elimination
            if args.perform_rfe:
                output_feature_ranks(row, eliminated_features, selected_feature_names)

            if args.error_log:  # calculate error statistics for each sample
                included_idx = np.array(x_test_idx)  # samples included in this test iteration
                predict_results[included_idx, it - 1] = y_predict  # remember prediction results of all included samples

            # report the performance of final AHA classification
            output_aha_result(row, x_test_idx, y_test, y_predict, label_encoder, x_rhythm_types)
            output_multiclass_result(row, x_test_idx, y_test, y_predict, label_encoder, x_rhythm_types)
            # best scores
            row["cv_score"] = best_cv_score
            row["testing_score"] = best_test_score
            # store best parameters of grid search
            output_best_params(row, best_params)
            # feature importance / ranking
            output_feature_scores(row, best_estimator, selected_feature_names)

            rows.append(row)  # remember each row so we can calculate average for them later
            print("\n".join(["\t{0} = {1}".format(field, row.get(field, 0.0)) for field in csv_fields[1:]]))
            writer.writerow(row)
            print("-" * 80)

        # calculate average for all iterations automatically and write to csv
        avg = calculate_average(rows, csv_fields, param_grid)
        writer.writerow(avg)
        print("\n".join(["\t{0} = {1}".format(field, avg.get(field, 0.0)) for field in csv_fields[1:]]))

        # log prediction errors of each sample during the test iterations in a csv file.
        if args.error_log:
            output_errors(args.error_log, predict_results, x_data_info, aha_y_data)


if __name__ == "__main__":
    main()
