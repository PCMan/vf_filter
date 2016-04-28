#!/usr/bin/env python3
# import pyximport; pyximport.install()
# evaluation tools for classification
import numpy as np
from array import array
from sklearn import cross_validation


def balanced_error_rate(y_true, y_predict):
    incorrect = (y_true != y_predict)
    fp = np.sum(np.logical_and(y_predict, incorrect))
    pred_negative = np.logical_not(y_predict)
    fn = np.sum(np.logical_and(pred_negative, incorrect))
    n_positive = np.sum(y_true)
    n_negative = len(y_true) - n_positive
    return 0.5 * (fn / n_positive + fp / n_negative)


class ClassificationResult:
    def __init__(self, y_true, y_predict):
        correct = (y_true == y_predict)
        incorrect = np.logical_not(correct)
        pred_negative = np.logical_not(y_predict)
        tp = np.sum(np.logical_and(y_predict, correct))
        fp = np.sum(np.logical_and(y_predict, incorrect))
        tn = np.sum(np.logical_and(pred_negative, correct))
        fn = np.sum(np.logical_and(pred_negative, incorrect))
        self.sensitivity = 0.0 if tp == 0 else tp / (tp + fn)
        self.specificity = 0.0 if tn == 0 else tn / (tn + fp)
        self.precision = 0.0 if tp == 0 else tp / (tp + fp)
        self.accuracy = (tp + tn) / len(y_true)
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn


NON_SHOCKABLE = 0
SHOCKABLE = 1
INTERMEDIATE = 2
EXCLUDED = 3

RAPID_VT_RATE = 100

# Test the classifiers with the settings suggested by AHA for AEDs
class AHATest:

    def __init__(self, x_data, x_data_info):
        # prepare the data for AHA test procedure for AED
        self.coarse_vf_idx = array('i')
        self.fine_vf_idx = array('i')
        self.rapid_vt_idx = array('i')
        self.slow_vt_idx = array('i')
        self.nsr_idx = array('i')
        self.asystole_idx = array('i')
        self.others_idx = array('i')
        y_data = np.zeros(len(x_data))

        for i, info in enumerate(x_data_info):  # examine the info of each ECG segment
            last_rhythm = info.get_last_rhythm()
            if last_rhythm:
                name = last_rhythm.name
                if name == "(VF":
                    if last_rhythm.is_coarse:
                        self.coarse_vf_idx.append(i)
                        y_data[i] = SHOCKABLE
                    else:
                        self.fine_vf_idx.append(i)
                        y_data[i] = INTERMEDIATE
                elif name == "(VT":
                    hr = last_rhythm.get_heart_rate()
                    if hr > RAPID_VT_RATE:
                        self.rapid_vt_idx.append(i)
                        y_data[i] = SHOCKABLE
                    elif hr > 0:
                        self.slow_vt_idx.append(i)
                        y_data[i] = INTERMEDIATE
                    else:
                        y_data[i] = EXCLUDED
                    # rhythms with HR = 0 BPM are those for which HR is unknwon.
                elif name == "(VFL":
                    self.rapid_vt_idx.append(i)
                    y_data[i] = SHOCKABLE
                elif name == "(N":
                    # nearly all rythms other than VF are annotated as NSR in cudb
                    # so it's unreliable. Drop NSR beats from cudb.
                    if info.record.startswith("cudb/"):
                        y_data[i] = EXCLUDED
                    else:
                        self.nsr_idx.append(i)
                        y_data[i] = NON_SHOCKABLE
                elif name == "(ASYS":
                    self.asystole_idx.append(i)
                    y_data[i] = NON_SHOCKABLE
                else:
                    self.others_idx.append(i)
                    y_data[i] = NON_SHOCKABLE
        self.y_data = y_data

    # randomly generate train and test set based on AHA guideline for AED
    def train_test_split(self, test_size=0.3):
        """
        Test dataset composition suggested by AHA for AED evaluation
        1: shockable:
            coarse VF:      200
            rapid VT:       50
        2: intermediate:
            fine VF         25
            other VT:       25
        0: non - shockable: others(especially asystole)
            NSR:            100
            asystole:       100
            others:         70 (AF, SB, SVT, IVR, PVC)
        """
        subtypes = [
            self.coarse_vf_idx, self.rapid_vt_idx,  # shockable
            self.fine_vf_idx, self.slow_vt_idx,  # intermediate
            self.nsr_idx, self.asystole_idx, self.others_idx  # non-shockable
        ]

        # desired total sample number for each kind of arrhythmia
        n_desired = np.round(np.array([200, 50, 25, 25, 100, 100, 70]) / test_size)

        # numbers for each kind actually available
        n_avail = np.array([len(subtype_idx) for subtype_idx in subtypes])

        # numbers of samples that are going to be selected
        # we need to perform stratified sampling for each kind of arrhythmia
        scale = n_avail / n_desired
        n_actual = (n_desired * np.min(scale[scale > 1.0])).astype("int")

        x_train_idx = []
        x_test_idx = []
        y_train = []
        y_test = []
        for subtype_idx, n_subtype in zip(subtypes, n_actual):
            if n_subtype > len(subtype_idx):
                n_subtype = len(subtype_idx)
            if n_subtype == 1:
                print("warning: number of samples in subtype is not enough")
                continue
            n_test = int(np.floor(n_subtype * test_size))  # test set
            n_train = n_subtype - n_test  # training set
            sub_y_data = self.y_data[subtype_idx]
            sub_x_train_idx, sub_x_test_idx, sub_y_train, sub_y_test = cross_validation.train_test_split(subtype_idx,
                                                                                                         sub_y_data,
                                                                                                         test_size=n_test,
                                                                                                         train_size=n_train)
            x_train_idx.append(sub_x_train_idx)
            x_test_idx.append(sub_x_test_idx)
            y_train.append(sub_y_train)
            y_test.append(sub_y_test)
        return np.concatenate(x_train_idx), np.concatenate(x_test_idx), np.concatenate(y_train), np.concatenate(y_test)
