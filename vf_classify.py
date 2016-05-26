#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from vf_eval import *


# initial classification to detect possibly shockable rhythm
SAFE_RHYTHM = 0  # others
DANGEROUS_RHYTHM = 1  # VF or VT
binary_class_names = ["safe", "dangerous"]

# AHA clasases
NON_SHOCKABLE = 0
SHOCKABLE = 1
INTERMEDIATE = 2
aha_classes = (NON_SHOCKABLE, SHOCKABLE, INTERMEDIATE)
aha_classe_names = ["non-shockable", "shockable", "intermediate"]
shockable_rhythms = ("(VF", "(VT", "(VFL")

# Use threshold value: 180 BPM to define rapid VT
# Reference: Nishiyama et al. 2015. Diagnosis of Automated External Defibrillators (JAHA)
RAPID_VT_RATE = 180

# 0.2 mV is suggested by AHA
COARSE_VF_THRESHOLD = 0.2


def create_aha_labels(x_data, x_data_info):
    y_data = np.zeros(len(x_data_info), dtype="int")
    for i in range(len(x_data)):
        x = x_data[i]
        info = x_data_info[i]
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
                y_data[i] = SHOCKABLE
            else:  # fine VF
                y_data[i] = INTERMEDIATE
        elif rhythm in ("(VT", "(VFL"):
            # VFL is VF with HR > 240 BPM, so it's kind of rapid VT
            # However, in the dataset we found segments with slower heart rate
            # marked as VFL. So let's double check here
            hr = info.get_heart_rate()
            if hr >= RAPID_VT_RATE:
                y_data[i] = SHOCKABLE
            elif hr > 0:
                y_data[i] = INTERMEDIATE
            else:  # no heart rate information
                y_data[i] = SHOCKABLE if rhythm == "(VFL" else INTERMEDIATE
    return y_data


def create_binary_labels(x_data_info):
    y_data = np.zeros(len(x_data_info), dtype="int")
    for i, info in enumerate(x_data_info):
        if info.rhythm in shockable_rhythms:
            y_data[i] = DANGEROUS_RHYTHM
    return y_data


def exclude_rhythms(x_data, x_data_info, excluded_rhythms):
    excluded_idx = np.array([i for i, info in enumerate(x_data_info) if info.rhythm in excluded_rhythms])
    x_data = np.delete(x_data, excluded_idx, axis=0)
    x_data_info = np.delete(x_data_info, excluded_idx, axis=0)
    return x_data, x_data_info


def get_balanced_sample_weights(y_data):
    classes = np.unique(y_data)
    n_classes = [np.sum([y_data == k]) for k in classes]
    n_total = len(y_data)
    weights = np.zeros(y_data.shape)
    for k, n in zip(classes, n_classes):
        weights[y_data == k] = (n_total / n)
    return weights


def create_estimator(estimator_name, class_weight):
    estimator = None
    param_grid = None
    support_class_weight = False

    if estimator_name == "logistic_regression":
        from sklearn import linear_model
        estimator = linear_model.LogisticRegression(class_weight=class_weight)
        param_grid = {
            "C": np.logspace(-2, 4, 10)
        }
        support_class_weight = True
    elif estimator_name == "random_forest":
        estimator = ensemble.RandomForestClassifier(class_weight=class_weight)
        param_grid = {
            "n_estimators": list(range(10, 110, 10))
        }
        support_class_weight = True
        # support_class_weight = False
    elif estimator_name == "gradient_boosting":
        import xgboost.sklearn as xgb
        estimator = xgb.XGBClassifier(learning_rate=0.1)
        param_grid = {
            "n_estimators": list(range(150, 250, 10)),
            "max_depth": list(range(3, 8))
        }
    elif estimator_name == "adaboost":
        estimator = ensemble.AdaBoostClassifier()
        param_grid = {
            "n_estimators": list(range(30, 150, 10)),
            "learning_rate": np.logspace(-1, 0, 2)
        }
    elif estimator_name == "svc":
        from sklearn import svm
        estimator = svm.SVC(shrinking=False,
                            cache_size=2048,
                            verbose=False,
                            probability=True,
                            class_weight=class_weight)
        param_grid = {
            "C": np.logspace(0, 1, 2),
            "gamma": np.logspace(-2, -1, 2)
        }
        support_class_weight = True
    elif estimator_name == "mlp1" or estimator_name == "mlp2":  # multiple layer perceptron neural network
        from sknn import mlp
        param_grid = {
            "learning_rate": [0.0001],
            "regularize": ["l2"],  # , "dropout"],
            "weight_decay": np.logspace(-6, -5, 2),  # parameter for L2 regularizer
            "hidden0__type": ["Tanh"]  # "Rectifier", "Sigmoid"
        }

        layers = [mlp.Layer(type="Tanh", name="hidden0")]
        # add the second hidden layer as needed
        if estimator_name == "mlp2":  # 2 hidden layer
            layers.append(mlp.Layer(type="Tanh", name="hidden1"))
            param_grid["hidden0__units"] = list(range(2, 5, 1))
            param_grid["hidden1__units"] = list(range(2, 5, 1))
            param_grid["hidden1__type"] = ["Tanh"]  # "Rectifier", "Sigmoid"
        else:
            param_grid["hidden0__units"] = list(range(5, 26, 1))
        # add the output layer
        layers.append(mlp.Layer("Softmax"))
        estimator = mlp.Classifier(layers=layers, batch_size=150)

    return estimator, param_grid, support_class_weight


"""
    # final classification based on AHA requirements
    def aha_classifier(x_test, x_test_info, binary_y_predict):
        aha_y_predict = np.zeros(len(x_test), dtype="int")
        for i in range(len(x_test)):
            # Check if this rhythm is one of VF, VT, or VFL (dangerous rhythms)
            if binary_y_predict[i] == DANGEROUS_RHYTHM:
                info = x_test_info[i]
                # This rhythm can be VF, VFL, or VT, but we don't know
                # TODO: we may use some simple features to distinguish them if needed
                amplitude = info.amplitude
                # perform QRS detection to calculate heart rate
                # here we get the stored QRS detection result done previously for speed up.
                beats = info.detected_beats
                if beats:  # heart beats are detected
                    # This can be VF (shockable), slow VT (intermediate), or other misclassified "safe" rhythms
                    # FIXME: find a method to distinguish them
                    if hr >= RAPID_VT_RATE:  # this is either rapid VT or VF (both are shockable, no need to distinguish them)
                        y = SHOCKABLE
                    else:  # this rhythm is slower than 180 BPM
                        y = INTERMEDIATE
                else:  # no QRS complex was found, this must be VF or asystole
                    if amplitude >= COARSE_VF_THRESHOLD:
                        y = SHOCKABLE
                    else:
                        y = INTERMEDIATE
                aha_y_predict[i] = y
        return aha_y_predict
"""
