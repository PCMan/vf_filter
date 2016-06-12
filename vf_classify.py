#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import grid_search
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
shockable_rhythms = ("(VF,coarse", "(VT,rapid")
intermediate_rhythms = ("(VF,fine", "(VT,slow")

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


def create_estimator(estimator_name, class_weight):
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
                "C": np.logspace(-6, 2, 50),
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


class VfClassifier:
    def __init__(self, estimator_name="svc_rbf", n_cv_folds=5, scorer="f1_macro",
                 class_weight="balanced", filter_fs_order=None, n_rfe_iters=0, n_jobs=-1):
        self.estimator_name = estimator_name
        self.n_cv_folds = n_cv_folds
        self.scorer = scorer
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        estimator, param_grid, support_class_weight = create_estimator(estimator_name, class_weight)
        self.estimator = estimator
        self.param_grid = param_grid
        self.support_class_weight = support_class_weight

        self.filter_fs_order = filter_fs_order  # list features for filter type feature selection in order of elimination
        if filter_fs_order:  # RFE and filter type FS cannot be performed at the same time
            n_rfe_iters = 0
        self.n_rfe_iters = n_rfe_iters  # perform recursive feature elimination (RFE) and get feature ranks

        # results of each iterations
        self.data_scalers = []
        self.estimators = []
        self.params = []
        self.cv_scores = []
        self.best_iter = -1
        self.selected_feature_masks = []  # masks for selected features in each iteration
        self.eliminated_features = []

    def reset(self):
        if self.best_iter != -1:  # clear previous trained results:
            self.estimators = []
            self.params = []
            self.cv_scores = []
            self.best_iter = -1
            self.selected_feature_masks = []  # masks for selected features in each iteration
            self.eliminated_features = []
            self.data_scalers = []

    def set_filter_fs_order(self, filter_fs_order):
        self.reset()
        self.filter_fs_order = filter_fs_order

    def train(self, x_train, y_train):
        self.reset()

        fit_params = None
        # try to balance class weighting
        if self.class_weight == "balanced" and not self.support_class_weight:
            # perform sample weighting instead if the estimator does not support class weighting
            weight_arg = "w" if self.estimator_name.startswith("mlp") else "sample_weight"
            fit_params = {
                weight_arg: np.array(get_balanced_sample_weights(y_train))
            }

        n_features = x_train.shape[1]
        selected_features_mask = np.ones(n_features, dtype=np.bool)
        # number of iterations
        if self.n_rfe_iters:
            n_iters = self.n_rfe_iters
        elif self.filter_fs_order:
            n_iters = len(self.filter_fs_order)
        else:
            n_iters = 1

        # find best parameters using grid search + cross validation
        grid = grid_search.GridSearchCV(self.estimator,
                                        self.param_grid,
                                        fit_params=fit_params,
                                        scoring=self.scorer,
                                        n_jobs=self.n_jobs,
                                        cv=self.n_cv_folds,
                                        verbose=0)
        best_score = 0.0
        # Because of a bug in joblib, we see a lot of warnings here.
        # https://github.com/scikit-learn/scikit-learn/issues/6370
        # Use the workaround to turn off the warnings
        import warnings
        warnings.filterwarnings("ignore")

        for it in range(n_iters):
            x_train_selected = x_train[:, selected_features_mask]
            print("n_selected_features:", np.sum(selected_features_mask))
            # scale the features (NOTE: training and testing sets should be scaled by the same factor.)
            # scale to [-1, 1] (or scale to [0, 1]. scaling is especially needed by SVM)
            data_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
            x_train_selected = data_scaler.fit_transform(x_train_selected)
            self.data_scalers.append(data_scaler)
            grid.fit(x_train_selected, y_train)  # training the model
            estimator = grid.best_estimator_
            self.estimators.append(estimator)
            self.cv_scores.append(grid.best_score_)
            self.params.append(grid.best_params_)
            self.selected_feature_masks.append(np.copy(selected_features_mask))  # need to copy the array
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                self.best_iter = it

            eliminated_feature_idx = -1
            if self.n_rfe_iters:  # perform recursive feature elimination (eliminate the least important feature)
                # find the worst feature in this round (lowest score/coefficient)
                if hasattr(estimator, 'coef_'):
                    coefs = estimator.coef_
                elif hasattr(estimator, 'feature_importances_'):
                    coefs = estimator.feature_importances_
                else:  # no feature importance scores for ranking.
                    break  # FIXME: we should raise an exception here
                feature_importance = coefs ** 2
                if coefs.ndim > 1:
                    feature_importance = feature_importance.sum(axis=0)
                i_min_score = np.argmin(feature_importance)  # find worst feature
                feature_ids = np.flatnonzero(selected_features_mask)
                print("rfe:", it, feature_ids, i_min_score)
                eliminated_feature_idx = feature_ids[i_min_score]  # find worst feature
            elif self.filter_fs_order:  # perform filter type feature selection
                eliminated_feature_idx = self.filter_fs_order[it]
                print("filter_fs:", it, eliminated_feature_idx)
            if eliminated_feature_idx != -1:
                selected_features_mask[eliminated_feature_idx] = False
                self.eliminated_features.append(eliminated_feature_idx)

    def predict(self, x_test, y_test=None, i_estimator=-1, predict=True, score=True):
        if i_estimator == -1:  # the estimator to use is not specified
            i_estimator = self.best_iter  # use the best one automatically
        if i_estimator == -1:  # do not have any trained estimators, need to call train() first.
            return None
        selected_features_mask = self.selected_feature_masks[i_estimator]
        x_test_selected = x_test[:, selected_features_mask]  # only select some dimensions
        estimator = self.estimators[i_estimator]
        data_scaler = self.data_scalers[i_estimator]

        # the test dataset should be scaled by the same factor
        x_test_selected = data_scaler.transform(x_test_selected)
        if predict:
            y_predict = estimator.predict(x_test_selected)
        else:
            y_predict = None

        if score:
            test_score = estimator.score(x_test_selected, y_test)
        else:
            test_score = 0.0
        return y_predict, test_score

    def has_coefficients(self):
        return self.estimator_name in ("logistic_regression", "svc_linear")

    def has_feature_importances(self):
        return self.estimator_name in ("random_forest", "adaboost", "gradient_boosting")
