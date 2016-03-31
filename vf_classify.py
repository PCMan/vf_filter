#!/usr/bin/env python2
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


N_JOBS = int(mp.cpu_count() / 2) if mp.cpu_count() > 1 else 1
N_CV_FOLDS = 10


def balanced_error_rate(y_true, y_predict):
    incorrect = (y_true != y_predict)
    fp = np.sum(np.logical_and(y_predict, incorrect))
    pred_negative = np.logical_not(y_predict)
    fn = np.sum(np.logical_and(pred_negative, incorrect))
    n_positive = np.sum(y_true)
    n_negative = len(y_true) - n_positive
    return  0.5 * (fn / n_positive + fp / n_negative)


def classification_report(y_true, y_predict, x_test_info=None):
    correct = (y_true == y_predict)
    incorrect = np.logical_not(correct)
    pred_negative = np.logical_not(y_predict)
    tp = np.sum(np.logical_and(y_predict, correct))
    fp = np.sum(np.logical_and(y_predict, incorrect))
    tn = np.sum(np.logical_and(pred_negative, correct))
    fn = np.sum(np.logical_and(pred_negative, incorrect))
    print "tp =%d, fp = %d, tn = %d, fn = %d" % (tp, fp, tn, fn)
    print "sensitivity:", float(tp) / (tp + fn), "specificity:", float(tn) / (tn + fp), "precision:", float(tp) / (tp + fp)


def list_classification_errors(y_true, y_predict, x_info):
    incorrect = (y_true != y_predict)
    print "list segments with errors:"
    for info in x_info[incorrect]:
        (record, begin_time) = info
        print "  ", record, begin_time


def output_errors(y_true, y_predict, x_indicies, filename):
    error_idx = sorted(np.array(x_indicies)[y_true != y_predict])
    with open(filename, "w") as f:
        for i in error_idx:
            f.write("{0}\n".format(i))


def main():

    # load features
    x_data, y_data, x_info = load_data(N_JOBS)
    print "Summary:\n", "# of segments:", len(x_data), "# of VT/Vf:", np.sum(y_data), len(x_info)
    # normalize the features
    preprocessing.normalize(x_data)
    x_indicies = range(0, len(x_data))

    # Here we split the indicies of the rows rather than the data array itself.
    x_train_idx, x_test_idx, y_train, y_test = cross_validation.train_test_split(x_indicies, y_data, test_size=0.3, stratify=y_data)
    x_train = x_data[x_train_idx]
    x_test = x_data[x_test_idx]
    x_test_info = x_info[x_test_idx]

    # BER-based scoring function
    cv_scorer = metrics.make_scorer(balanced_error_rate, greater_is_better=False)

    # Logistic regression
    estimator = linear_model.LogisticRegressionCV(scoring=cv_scorer)
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    # print "Logistic regression: error:", float(np.sum(y_predict != y_test) * 100) / len(y_test), "%"
    print "Logistic regression: precision:\n", classification_report(y_test, y_predict), estimator.scores_, "\n"
    output_errors(y_test, y_predict, x_indicies=x_test_idx, filename="log_reg_errors.txt")


    # Random forest
    estimator = ensemble.RandomForestClassifier()
    grid = grid_search.RandomizedSearchCV(estimator, {
                                        "n_estimators": range(10, 110, 10)
                                    },
                                    n_iter=10,
                                    scoring=cv_scorer,
                                    n_jobs=N_JOBS, cv=N_CV_FOLDS, verbose=1)
    grid.fit(x_train, y_train)
    y_predict = grid.predict(x_test)
    print "RandomForest:\n", classification_report(y_test, y_predict), grid.best_params_, grid.best_score_, "\n"
    output_errors(y_test, y_predict, x_indicies=x_test_idx, filename="rf_errors.txt")


    # SVC with RBF kernel
    estimator = svm.SVC(shrinking=False, cache_size=1024, verbose=False)
    grid = grid_search.RandomizedSearchCV(estimator, {
                                        "C": np.logspace(-2, 2, 5),
                                        "gamma": np.logspace(-2, 2, 5)
                                    },
                                    scoring=cv_scorer,
                                    n_jobs=N_JOBS, cv=N_CV_FOLDS, verbose=1)
    grid.fit(x_train, y_train)
    y_predict = grid.predict(x_test)
    print "SVC:\n", classification_report(y_test, y_predict), grid.best_params_, grid.best_score_, "\n"
    output_errors(y_test, y_predict, x_indicies=x_test_idx, filename="svc_errors.txt")


    '''
    # AdaBoost decision tree
    estimator = ensemble.AdaBoostClassifier()
    grid = grid_search.RandomizedSearchCV(estimator, {
                                        "n_estimators": range(10, 110, 10),
                                        "learning_rate": np.logspace(-2, 1, 4)
                                    },
                                    n_iter=20,
                                    scoring=cv_scorer,
                                    n_jobs=N_JOBS, cv=N_CV_FOLDS, verbose=1)
    grid.fit(x_train, y_train)
    y_predict = grid.predict(x_test)
    print "AdaBoost:\n", classification_report(y_test, y_predict), grid.best_params_, grid.best_score_, "\n"
    '''

    # Gradient boosting
    estimator = ensemble.GradientBoostingClassifier()
    grid = grid_search.RandomizedSearchCV(estimator, {
                                        "learning_rate": np.logspace(-2, 1, 4),
                                        "n_estimators": range(50, 210, 10),
                                        "max_depth": range(3, 10)
                                    },
                                    n_iter=20,
                                    scoring=cv_scorer,
                                    n_jobs=N_JOBS, cv=N_CV_FOLDS, verbose=1)
    grid.fit(x_train, y_train)
    y_predict = grid.predict(x_test)
    print "Gradient Boosting:\n", classification_report(y_test, y_predict), grid.best_params_, grid.best_score_, "\n"
    # list_classification_errors(y_test, y_predict, x_info=x_test_info)
    # debugging:
    # inspect segments with errors
    output_errors(y_test, y_predict, x_indicies=x_test_idx, filename="gb_errors.txt")


if __name__ == "__main__":
    main()
