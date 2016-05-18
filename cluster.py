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
from array import array
from sklearn import cluster


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

# label the segment according to different problem definition
label_methods_desc = """
label methods:
aha: multi-class based on AHA guideline for AED:
    shockable (coarse VF + rapid VT): 1
    intermediate (fine VF + slow VT) :2
    non-shockable (others): 0
0: binary => VF: 1, others: 0
1: binary => VF or VFL: 1, others: 0
2: binary => VF or VFL or VT: 1, others: 0
3: multi-class:
    VF: 1
    VFL/VT: 2
    others: 0
"""

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


def correct_annotations(x_data_info, amendment_file):
    correction = {}
    with open(amendment_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            record_name, begin_time, correct_label = parts
            correction[(record_name, int(begin_time))] = correct_label
    if correction:
        for i, info in enumerate(x_data_info):
            replace = correction.get((info.record_name, info.begin_time), None)
            if replace and replace != "C":  # found an entry for the sample
                # C means correct and confirmed so there is no need to fix it if the mark is "C".
                print("Fix", info.record_name, info.begin_time, replace)
                # fix the incorrect rhythm annotation for this sample
                info.rhythm = replace
                if replace == "(VF" or replace == "(VT":
                    # currently, we don't know the amplitude of the segment so we cannot determine if its coarse or
                    # fine VF. So, let's exclude the sample for now. We don't know the rate of (VT either.
                    # TODO: calculate amplitude and rate for all segments?
                    info.rhythm = "X"


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


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    # known estimators
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-j", "--jobs", type=int, default=-1)
    parser.add_argument("-t", "--iter", type=int, default=1)
    parser.add_argument("-f", "--features", type=int, nargs="+")  # feature selection
    parser.add_argument("-x", "--exclude-rhythms", type=str, nargs="+", default=["(ASYS"])  # exclude some rhythms from the test
    parser.add_argument("-a", "--amendment-file", type=str, help="Override the incorrect labels of the original dataset.")
    args = parser.parse_args()

    # setup testing parameters
    print(args)
    selected_features = args.features

    # load features
    x_data, x_data_info = load_features(args.input)
    # only select the specified feature
    if selected_features:
        x_data = x_data[:, selected_features]

    # Override the incorrect annotations of the original dataset by an amendment file.
    if args.amendment_file:
        correct_annotations(x_data_info, args.amendment_file)

    # "X" is used internally by us (in amendment file) to mark some broken samples to exclude from the test
    excluded_rhythms = args.exclude_rhythms + ["X"] if args.exclude_rhythms else ["X"]
    # exclude samples with some rhythms from the test
    x_data, x_data_info = exclude_rhythms(x_data, x_data_info, excluded_rhythms)

    y_rhythm_names = np.array([info.rhythm for info in x_data_info])

    preprocessing.scale(x_data, copy=False)
    preprocessing.normalize(x_data, copy=False)

    n_clusters = 3
    centroid, label, inertia = cluster.k_means(x_data, n_clusters)
    class_idx = [(label == i) for i in range(0, n_clusters)]
    for idx in class_idx:
        rhythms = y_rhythm_names[idx]
        names, counts = np.unique(rhythms, return_counts=True)
        # print(names[np.argmax(counts)])
        print(sorted([item for item in zip(names, counts)], key=lambda item: item[1], reverse=True))

    '''
    dbscan = cluster.DBSCAN(eps=1.0)
    result = dbscan.fit_predict(x_data)
    print(np.unique(dbscan.labels_), np.sum(dbscan.labels_ == -1))
    '''

if __name__ == "__main__":
    main()
