#!/usr/bin/env python2
# coding: utf-8
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import ensemble
from sklearn.feature_selection import SelectFromModel
from vf_data import load_data
import multiprocessing as mp


N_JOBS = (mp.cpu_count() - 1) if mp.cpu_count() > 1 else 1


def main():

    # load features
    x_data, y_data, x_info = load_data(N_JOBS)
    print("Summary:\n", "# of segments:", len(x_data), "# of VT/Vf:", np.sum(y_data), len(x_info))
    # normalize the features
    preprocessing.normalize(x_data)

    # stability feature selection using randomized logistic regression
    estimator = linear_model.RandomizedLogisticRegression(n_resampling=500, verbose=True)
    estimator.fit(x_data, y_data)

    feature_names = ("TCSC", "TCI", "STE", "MEA", "PSR", "VF", "SPEC", "LZ", "SpEn")

    print("feature scores (stability selection):")
    for name, score in zip(feature_names, estimator.scores_):
        print(name, ":", score)

    print("")

    estimator = ensemble.RandomForestClassifier()
    estimator.fit(x_data, y_data)
    print("feature scores (tree selection):")
    for name, score in zip(feature_names, estimator.feature_importances_):
        print(name, ":", score)


if __name__ == "__main__":
    main()
