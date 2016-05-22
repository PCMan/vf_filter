#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import svm
from vf_features import feature_names, load_features
import multiprocessing as mp
from scipy.stats import pearsonr, spearmanr
import argparse
from array import array


N_JOBS = (mp.cpu_count() - 1) if mp.cpu_count() > 1 else 1


NON_SHOCKABLE = 0
SHOCKABLE = 1
INTERMEDIATE = 2
aha_classes = (NON_SHOCKABLE, SHOCKABLE, INTERMEDIATE)
aha_classe_names = ["non-shockable", "shockable", "intermediate"]
shockable_rhythms = ("(VF", "(VT", "(VFL", "(VF,coarse", "(VF,fine", "(VT,rapid", "(VT,slow")

# Use threshold value: 180 BPM to define rapid VT
# Reference: Nishiyama et al. 2015. Diagnosis of Automated External Defibrillators (JAHA)
RAPID_VT_RATE = 180

# 0.2 mV is suggested by AHA
COARSE_VF_THRESHOLD = 0.2


def make_aha_classes(x_data_info):
    for info in x_data_info:
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
                info.rhythm += ",coarse"
            else:  # fine VF
                info.rhythm += ",fine"
        elif rhythm == "(VT":
            hr = info.get_heart_rate()
            if hr >= RAPID_VT_RATE:
                info.rhythm += ",rapid"
            elif hr > 0:
                info.rhythm += ",slow"
        elif rhythm == "(VFL":  # VFL is VF with HR > 240 BPM, so it's kind of rapid VT
            info.rhythm = "(VT,rapid"



def make_labels(x_data_info, label_method):
    y_data = array('I')
    for info in x_data_info:
        rhythm = info.rhythm
        label = 0
        if label_method == "aha":
            # distinguish subtypes of VT and VF
            if rhythm == "(VF,coarse":
                label = SHOCKABLE
            elif rhythm == "(VF,fine":  # fine VF
                label = INTERMEDIATE
            elif rhythm == "(VT,rapid":
                label = SHOCKABLE
            elif rhythm == "(VT,slow":
                label = INTERMEDIATE
            else:
                label = NON_SHOCKABLE
        elif label_method == "0":
            label = 1 if rhythm == "(VF" else 0
        elif label_method == "1":
            label = 1 if rhythm in ("(VF", "(VFL") else 0
        elif label_method == "2":
            label = 1 if rhythm in ("(VF", "(VFL", "(VT") else 0
        elif label_method == "3":  # multi-class: VF, VFL/VT, others
            if rhythm == "(VF":
                label = 1
            elif rhythm in ("(VT", "(VFL"):  # We define VFL as rapid VT here.
                # VT at 240-300 beats/min is often termed ventricular flutter.
                # http://emedicine.medscape.com/article/159075-overview
                label = 2
            else:  # others
                label = 0
        y_data.append(label)
    return np.array(y_data)


def main():
    parser = argparse.ArgumentParser()
    # known estimators
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-l", "--label-method", type=str, default="1")
    args = parser.parse_args()

    # load features
    x_data, x_data_info = load_features(args.input)
    if args.label_method == "aha":  # distinguish subtypes of VT and VF
        make_aha_classes(x_data_info)

    # label the samples
    y_data = make_labels(x_data_info, args.label_method)

    for i in range(10):
        print(x_data[i], y_data[i])

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
