#!/usr/bin/env python3
# import pyximport; pyximport.install()
# evaluation tools for classification
import numpy as np
from array import array
from sklearn import cross_validation
from sklearn.preprocessing import label_binarize
from vf_eval import *

__all__ = ["AHATest"]

NON_SHOCKABLE = 0
SHOCKABLE = 1
INTERMEDIATE = 2
EXCLUDED = 3

RAPID_VT_RATE = 100


# Test the classifiers with the settings suggested by AHA for AEDs
class AHATest:

    def __init__(self, x_data, x_data_info):
        # prepare the data for AHA test procedure for AED
        coarse_vf_idx = array('i')
        fine_vf_idx = array('i')
        rapid_vt_idx = array('i')
        slow_vt_idx = array('i')
        nsr_idx = array('i')
        asystole_idx = array('i')
        others_idx = array('i')
        y_data = np.zeros(len(x_data))

        for i, info in enumerate(x_data_info):  # examine the info of each ECG segment
            last_rhythm = info.get_last_rhythm()
            if last_rhythm:
                name = last_rhythm.name
                if name == "(VF":
                    if last_rhythm.is_coarse:
                        coarse_vf_idx.append(i)
                        y_data[i] = SHOCKABLE
                    else:
                        fine_vf_idx.append(i)
                        y_data[i] = INTERMEDIATE
                elif name == "(VT":
                    hr = last_rhythm.get_heart_rate()
                    if hr > RAPID_VT_RATE:
                        rapid_vt_idx.append(i)
                        y_data[i] = SHOCKABLE
                    elif hr > 0:
                        slow_vt_idx.append(i)
                        y_data[i] = INTERMEDIATE
                    else:
                        y_data[i] = EXCLUDED
                    # rhythms with HR = 0 BPM are those for which HR is unknwon.
                elif name == "(VFL":
                    rapid_vt_idx.append(i)
                    y_data[i] = SHOCKABLE
                elif name == "(N":
                    # nearly all rhythms other than VF are annotated as NSR in cudb
                    # so it's unreliable. Drop NSR beats from cudb.
                    # edb is good, but it contains too many NSR samples, slowing down training. So drop it.
                    if info.record.startswith("cudb/") or info.record.startswith("edb/"):
                        y_data[i] = EXCLUDED
                    else:
                        nsr_idx.append(i)
                        y_data[i] = NON_SHOCKABLE
                elif name == "(ASYS":
                    asystole_idx.append(i)
                    y_data[i] = NON_SHOCKABLE
                else:
                    others_idx.append(i)
                    y_data[i] = NON_SHOCKABLE

        self.coarse_vf_idx = np.array(coarse_vf_idx)
        self.fine_vf_idx = np.array(fine_vf_idx)
        self.rapid_vt_idx = np.array(rapid_vt_idx)
        self.slow_vt_idx = np.array(slow_vt_idx)
        self.nsr_idx = np.array(nsr_idx)
        self.asystole_idx = np.array(asystole_idx)
        self.others_idx = np.array(others_idx)

        self.y_data = y_data

    def summary(self):
        n_shock = (len(self.coarse_vf_idx) + len(self.rapid_vt_idx))
        n_intermediate = (len(self.fine_vf_idx) + len(self.slow_vt_idx))
        n_nonshock = (len(self.asystole_idx) + len(self.nsr_idx) + len(self.others_idx))
        n_total_included = n_shock + n_nonshock + n_intermediate
        summary_text = """
total (the whole dataset): {total}
total included: {total_included}
shockable: {shock} ({shock_p} %)
    coarse VF: {cvf}
    rapid VT: {rvt}
intermediate: {inter} ({inter_p} %)
    fine VF: {fvf}
    slow VT: {svt}
non-shockable: {nonshock} ({nonshock_p} %)
    asystole: {asys}
    nsr: {nsr}
    others: {others}
        """.format(
            total=len(self.y_data),
            total_included=n_total_included,
            shock=n_shock,
            shock_p=(n_shock * 100 / n_total_included),
            cvf=len(self.coarse_vf_idx),
            rvt=len(self.rapid_vt_idx),
            inter=n_intermediate,
            inter_p=(n_intermediate * 100 / n_total_included),
            fvf=len(self.fine_vf_idx),
            svt=len(self.slow_vt_idx),
            nonshock=n_nonshock,
            nonshock_p=(n_nonshock * 100 / n_total_included),
            asys=len(self.asystole_idx),
            nsr=len(self.nsr_idx),
            others=len(self.others_idx)
        )
        return summary_text

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

        # stratified random sampling for each subtype of arrhtyhmia
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

    @staticmethod
    def classification_report(y_true, y_predict):
        result = {}
        # convert multi-class result to several binary tests
        classes = [NON_SHOCKABLE, SHOCKABLE, INTERMEDIATE]
        bin_true = label_binarize(y_true, classes=classes)
        bin_predict = label_binarize(y_predict, classes=classes)
        for i in range(len(classes)):
            bin_result = BinaryClassificationResult(bin_true[:, i], bin_predict[:, i])
            result[i] = bin_result
        return result

    @staticmethod
    def get_classes():
        return [NON_SHOCKABLE, SHOCKABLE, INTERMEDIATE]
