#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import svm
from vf_data import load_data
from vf_features import feature_names
import multiprocessing as mp
from scipy.stats import pearsonr, spearmanr


N_JOBS = (mp.cpu_count() - 1) if mp.cpu_count() > 1 else 1


def main():

    # load features
    x_data, y_data, x_info = load_data(N_JOBS)
    print("Summary:\n", "# of segments:", len(x_data), "# of VT/Vf:", np.sum(y_data), len(x_info))

    '''
    print("variance test:")
    fs = feature_selection.VarianceThreshold()
    fs.fit(x_data)
    for i, var in enumerate(fs.variances_):
        print(feature_names[i], "\t", var)
    print("")
    '''

    print("Pearson's correlation:")
    for i, name in enumerate(feature_names):
        correlation, p = pearsonr(x_data[:, i], y_data)
        print(name, "\t", correlation, "\tp =", p)
    print("")

    print("Spearman's correlation:")
    for i, name in enumerate(feature_names):
        correlation, p = spearmanr(x_data[:, i], y_data)
        print(name, "\t", correlation, "\tp =", p)
    print("")

    preprocessing.scale(x_data, copy=False)
    estimator = ensemble.RandomForestRegressor()
    print("Tree-based test for each feature:")
    for i, name in enumerate(feature_names):
        scores = cross_validation.cross_val_score(estimator,
                                                  x_data[:, i:i+1], y_data,
                                                  scoring="r2",
                                                  cv=cross_validation.ShuffleSplit(len(x_data), 3, .3))
        score = np.mean(scores)
        print(name, ":", score)
    print("")
    

if __name__ == "__main__":
    main()
