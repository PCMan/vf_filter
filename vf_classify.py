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


N_JOBS = 6
N_CV_FOLDS = 10
CV_SCORING = "accuracy"


def classification_report(y_true, y_predict):
    correct = (y_true == y_predict)
    incorrect = np.logical_not(correct)
    pred_negative = np.logical_not(y_predict)
    tp = np.sum(np.logical_and(y_predict, correct))
    fp = np.sum(np.logical_and(y_predict, incorrect))
    tn = np.sum(np.logical_and(pred_negative, correct))
    fn = np.sum(np.logical_and(pred_negative, incorrect))
    print "tp =%d, fp = %d, tn = %d, fn = %d" % (tp, fp, tn, fn)
    print "sensitivity:", float(tp) / (tp + fn), "specificity:", float(tn) / (tn + fp), "precision:", float(tp) / (tp + fp)


def main():

    # load features
    x_data, y_data = load_data(N_JOBS)
    print "Summary:\n", "# of segments:", len(x_data), "# of VT/Vf:", np.sum(y_data)
    # normalize the features
    preprocessing.normalize(x_data)
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=0.3, stratify=y_data)

    # Logistic regression
    estimator = linear_model.LogisticRegressionCV(scoring=CV_SCORING)
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    # print "Logistic regression: error:", float(np.sum(y_predict != y_test) * 100) / len(y_test), "%"
    print "Logistic regression: precision:\n", classification_report(y_test, y_predict), estimator.scores_, "\n"

    # Random forest
    estimator = ensemble.RandomForestClassifier()
    grid = grid_search.RandomizedSearchCV(estimator, {
                                        "n_estimators": range(10, 110, 10)
                                    },
                                    n_iter=10,
                                    scoring=CV_SCORING,
                                    n_jobs=N_JOBS, cv=N_CV_FOLDS, verbose=1)
    grid.fit(x_train, y_train)
    y_predict = grid.predict(x_test)
    print "RandomForest:\n", classification_report(y_test, y_predict), grid.best_params_, grid.best_score_, "\n"

    # SVC with RBF kernel
    estimator = svm.SVC(shrinking=False, cache_size=1024, verbose=False)
    grid = grid_search.RandomizedSearchCV(estimator, {
                                        "C": np.logspace(-2, 2, 5),
                                        "gamma": np.logspace(-2, 2, 5)
                                    },
                                    scoring=CV_SCORING,
                                    n_jobs=N_JOBS, cv=N_CV_FOLDS, verbose=1)
    grid.fit(x_train, y_train)
    y_predict = grid.predict(x_test)
    print "SVC:\n", classification_report(y_test, y_predict), grid.best_params_, grid.best_score_, "\n"

    # AdaBoost decision tree
    estimator = ensemble.AdaBoostClassifier()
    grid = grid_search.RandomizedSearchCV(estimator, {
                                        "n_estimators": range(10, 110, 10),
                                        "learning_rate": np.logspace(-2, 1, 4)
                                    },
                                    n_iter=20,
                                    scoring=CV_SCORING,
                                    n_jobs=N_JOBS, cv=N_CV_FOLDS, verbose=1)
    grid.fit(x_train, y_train)
    y_predict = grid.predict(x_test)
    print "AdaBoost:\n", classification_report(y_test, y_predict), grid.best_params_, grid.best_score_, "\n"

    # Gradient boosting
    estimator = ensemble.GradientBoostingClassifier()
    grid = grid_search.RandomizedSearchCV(estimator, {
                                        "learning_rate": np.logspace(-2, 1, 4),
                                        "n_estimators": range(50, 210, 10),
                                        "max_depth": range(3, 10)
                                    },
                                    n_iter=20,
                                    scoring=CV_SCORING,
                                    n_jobs=N_JOBS, cv=N_CV_FOLDS, verbose=1)
    grid.fit(x_train, y_train)
    y_predict = grid.predict(x_test)
    print "Gradient Boosting:\n", classification_report(y_test, y_predict), grid.best_params_, grid.best_score_, "\n"


if __name__ == "__main__":
    main()
