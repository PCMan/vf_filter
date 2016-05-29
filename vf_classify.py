#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import vf_eval


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
shockable_rhythms = ("(VF,coarse", "(VF,fine", "(VT,rapid", "(VT,slow")

# Use threshold value: 180 BPM to define rapid VT
# Reference: Nishiyama et al. 2015. Diagnosis of Automated External Defibrillators (JAHA)
RAPID_VT_RATE = 180

# 0.2 mV is suggested by AHA
COARSE_VF_THRESHOLD = 0.2

estimator_names = ("logistic_regression", "random_forest", "adaboost", "gradient_boosting", "svc_linear", "svc_poly", "svc_rbf", "mlp1", "mlp2")


# Generate y labels for each input data point according to AHA classification scheme
# After calling this function, the "rhythm" fields of some x_data_info elements will be modified.
def initialize_aha_labels(x_data, x_data_info):
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
                info.rhythm = "(VF,coarse"
            else:  # fine VF
                y_data[i] = INTERMEDIATE
                info.rhythm = "(VF,fine"
        elif rhythm in ("(VT", "(VFL"):
            # VFL is VF with HR > 240 BPM, so it's kind of rapid VT
            # However, in the dataset we found segments with slower heart rate
            # marked as VFL. So let's double check here
            hr = info.get_heart_rate()
            if hr >= RAPID_VT_RATE:
                y_data[i] = SHOCKABLE
                info.rhythm = "(VT,rapid"
            elif hr > 0:
                y_data[i] = INTERMEDIATE
                info.rhythm = "(VT,slow"
            else:  # no heart rate information
                if rhythm == "(VFL":
                    y_data[i] = SHOCKABLE
                    info.rhythm = "(VT,rapid"
                else:
                    y_data[i] = INTERMEDIATE
                    info.rhythm = "(VT,slow"
    return y_data


def exclude_rhythms(x_data, x_data_info, excluded_rhythms):
    # "X" is used internally by us (in label correction file) to mark some broken samples to exclude from the test
    excluded_idx = np.array([i for i, info in enumerate(x_data_info) if info.rhythm in excluded_rhythms or info.rhythm == "X"])
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


def create_estimator(estimator_name, class_weight, n_features):
    estimator = None
    param_grid = None
    support_class_weight = False

    if estimator_name == "logistic_regression":
        from sklearn import linear_model
        estimator = linear_model.LogisticRegression(class_weight=class_weight)
        param_grid = {
            "C": np.logspace(-3, 4, 20)
        }
        support_class_weight = True
    elif estimator_name == "random_forest":
        estimator = ensemble.RandomForestClassifier(class_weight=class_weight)
        param_grid = {
            "n_estimators": list(range(10, 110, 10)),
            "max_features": ("auto", 0.5, 0.8, None)
            # "max_features": np.arange(int(np.sqrt(n_features)), n_features, step=4)
        }
        support_class_weight = True
        # support_class_weight = False
    elif estimator_name == "gradient_boosting":
        """
        import xgboost.sklearn as xgb
        estimator = xgb.XGBClassifier(learning_rate=0.1)
        param_grid = {
            # "n_estimators": list(range(150, 250, 10)),
            # "max_depth": list(range(3, 8))
        }
        """
        # for some unknown reason, XGBoost does not perform well on my machine and hangs sometimes
        # fallback to use the less efficient implementation in sklearn.
        estimator = ensemble.GradientBoostingClassifier(learning_rate=0.1, warm_start=True)
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
    elif estimator_name.startswith("svc_"):
        subtype = estimator_name[4:]
        from sklearn import svm
        if subtype == "linear":  # linear SVC uses liblinear insteaed of libsvm internally, which is more efficient
            param_grid = {
                "C": np.logspace(-5, 2, 50),
            }
            estimator = svm.LinearSVC(dual=False,  # dual=False when n_samples > n_features according to the API doc.
                                      class_weight=class_weight)
        else:
            estimator = svm.SVC(shrinking=False,
                                cache_size=2048,
                                verbose=False,
                                probability=False,  # use True when predict_proba() is needed
                                class_weight=class_weight)
            if subtype == "rbf":
                estimator.set_params(kernel="rbf")
                param_grid = {
                    "C": np.logspace(-2, 2, 20),
                    "gamma": np.logspace(-2, -1, 3)
                }
            else:  # poly
                estimator.set_params(kernel="poly")
                param_grid = {
                    "degree": [2],
                    "C": np.logspace(-3, 1, 20)
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
