# Estimators
# Linear models

import optuna

# Gradient boosting frameworks
from catboost import CatBoostClassifier, CatBoostRegressor

# Imbalanced-learn ensemble
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from ray import air, tune
from ray.air import session

# Ensemble methods
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDRegressor,
)

# Other classifiers and regressors
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

# for gridsearchcv, randomsearchcv
# Define hyperparameters for each model
lr_params = {"penalty": ["l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100]}
rf_params = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
xgb_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [6, 10, 15],
    "learning_rate": [0.001, 0.01, 0.1],
}
cb_params = {
    "iterations": [50, 100, 200],
    "depth": [4, 6, 8],
    "learning_rate": [0.001, 0.01, 0.1],
}
lgbm_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [-1, 10, 20],
    "learning_rate": [0.001, 0.01, 0.1],
}
brf_params = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
mlp_params = {
    "hidden_layer_sizes": [(50,), (100,)],
    "activation": ["relu", "tanh", "logistic"],
}
gbc_params = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.001, 0.01, 0.1],
    "max_depth": [3, 5, 10],
}
abc_params = {"n_estimators": [50, 100, 200], "learning_rate": [0.001, 0.01, 0.1]}
svc_params = {"C": [0.1, 1, 10], "gamma": [0.001, 0.01, 0.1]}
knn_params = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
dtc_params = {"criterion": ["gini", "entropy"], "max_depth": [None, 5, 10]}
etc_params = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}
gnb_params = {}

# Put models and hyperparameters into a list of tuples
estimators_params_clfs = [
    (LogisticRegression, lr_params),
    (RandomForestClassifier, rf_params),
    (XGBClassifier, xgb_params),
    (CatBoostClassifier, cb_params),
    (LGBMClassifier, lgbm_params),
    (BalancedRandomForestClassifier, brf_params),
    (MLPClassifier, mlp_params),
    (GradientBoostingClassifier, gbc_params),
    (AdaBoostClassifier, abc_params),
    #                     (SVC, svc_params),
    #                     (KNeighborsClassifier, knn_params),
    #                     (DecisionTreeClassifier, dtc_params),
    #                     (ExtraTreesClassifier, etc_params),
    #                     (GaussianNB, gnb_params)
]

# Define hyperparameters for each model
lr_params_reg = {"fit_intercept": [True, False]}
ridge_params_reg = {"alpha": [0.5, 1.0]}
lasso_params_reg = {"alpha": [0.5, 1.0]}
elastic_params_reg = {"alpha": [0.5, 1.0], "l1_ratio": [0.5, 0.7]}
xgb_params_reg = {"max_depth": [3, 5], "learning_rate": [0.01, 0.1]}
lgbm_params_reg = {"max_depth": [3, 5], "learning_rate": [0.01, 0.1]}
cb_params_reg = {"depth": [3, 5], "learning_rate": [0.01, 0.1]}
svr_params_reg = {"kernel": ["linear", "poly", "rbf", "sigmoid"], "C": [0.5, 1.0]}
lsvr_params_reg = {"C": [0.5, 1.0]}
knr_params_reg = {
    "n_neighbors": [
        5,
        7,
        10,
        15,
        20,
    ],  # Increasing the number of neighbors can help in making the model more generalized.
    "weights": [
        "uniform",
        "distance",
    ],  # The 'distance' option can give more importance to closer instances, which may help reduce overfitting.
    "algorithm": [
        "auto",
        "ball_tree",
        "kd_tree",
        "brute",
    ],  # The algorithm used to compute the nearest neighbors can sometimes have an effect on overfitting, but it generally depends more on the dataset.
    "p": [
        1,
        2,
    ],  # This corresponds to the power parameter for the Minkowski metric. 1 is for manhattan_distance and 2 for euclidean_distance.
}
dtr_params_reg = {
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [1, 2, 5, 10],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": [None, "auto", "sqrt", "log2"],
    "max_leaf_nodes": [None, 10, 20, 30, 40],
}
rfr_params_reg = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
gbr_params_reg = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10],
    "learning_rate": [0.001, 0.01, 0.1],
}
etr_params_reg = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
abr_params_reg = {"n_estimators": [50, 100, 200], "learning_rate": [0.001, 0.01, 0.1]}
mlpr_params_reg = {"hidden_layer_sizes": [(100,)], "activation": ["relu", "tanh"]}
sgr_params_reg = {
    "loss": ["squared_loss", "huber", "epsilon_insensitive"],
    "penalty": ["l2", "l1", "elasticnet"],
    "alpha": [0.0001, 0.001, 0.01],
}

# Put models and hyperparameters into a list of tuples
estimators_params_regs = [
    (LinearRegression, lr_params_reg),
    (Ridge, ridge_params_reg),
    (Lasso, lasso_params_reg),
    (ElasticNet, elastic_params_reg),
    (XGBRegressor, xgb_params_reg),
    (LGBMRegressor, lgbm_params_reg),
    (CatBoostRegressor, cb_params_reg),
    # (SVR, svr_params_reg),
    (LinearSVR, lsvr_params_reg),
    # TODO overfit
    # (KNeighborsRegressor, knr_params_reg),
    # TODO overfit
    # (DecisionTreeRegressor, dtr_params_reg),
    (RandomForestRegressor, rfr_params_reg),
    (GradientBoostingRegressor, gbr_params_reg),
    # (ExtraTreesRegressor, etr_params_reg),
    (AdaBoostRegressor, abr_params_reg),
    # TODO underfit
    # (MLPRegressor, mlpr_params_reg),
    # (SGDRegressor, sgr_params_reg)
]


# Define hyperparameters for each model
lr_params_clf_optuna_search = {"penalty": ["l2"], "C": [0.001, 100.0]}
rf_params_clf_optuna_search = {"n_estimators": [50, 200], "max_depth": [10, 20]}
xgb_params_clf_optuna_search = {
    "n_estimators": [50, 200],
    "max_depth": [10, 15],
    "learning_rate": [0.001, 0.1],
}
cb_params_clf_optuna_search = {
    "iterations": [50, 200],
    "depth": [4, 8],
    "learning_rate": [0.001, 0.1],
}
lgbm_params_clf_optuna_search = {
    "n_estimators": [50, 200],
    "max_depth": [10, 20],
    "learning_rate": [0.001, 0.1],
}
brf_params_clf_optuna_search = {"n_estimators": [50, 200], "max_depth": [10, 20]}
mlp_params_clf_optuna_search = {
    "hidden_layer_sizes": [(50,), (100,)],
    "activation": ["relu", "tanh", "logistic"],
}
gbc_params_clf_optuna_search = {
    "n_estimators": [50, 200],
    "learning_rate": [0.001, 0.1],
    "max_depth": [3, 10],
}
abc_params_clf_optuna_search = {
    "n_estimators": [50, 200],
    "learning_rate": [0.001, 0.1],
}
svc_params_clf_optuna_search = {"C": [0.1, 10.0], "gamma": [0.001, 0.1]}
knn_params_clf_optuna_search = {
    "n_neighbors": [3, 7],
    "weights": ["uniform", "distance"],
}
dtc_params_clf_optuna_search = {"criterion": ["gini", "entropy"], "max_depth": [5, 10]}
etc_params_clf_optuna_search = {"n_estimators": [50, 200], "max_depth": [5, 10]}


# Put models and hyperparameters into a list of tuples
estimators_params_optuna_clfs = [
    (LogisticRegression, lr_params_clf_optuna_search),
    (RandomForestClassifier, rf_params_clf_optuna_search),
    (XGBClassifier, xgb_params_clf_optuna_search),
    (CatBoostClassifier, cb_params_clf_optuna_search),
    (LGBMClassifier, lgbm_params_clf_optuna_search),
    (BalancedRandomForestClassifier, brf_params_clf_optuna_search),
    (GradientBoostingClassifier, gbc_params_clf_optuna_search),
    (AdaBoostClassifier, abc_params_clf_optuna_search),
    # (SVC, svc_params_clf_optuna_search),
    # (KNeighborsClassifier, knn_params_clf_optuna_search),
    # (DecisionTreeClassifier, dtc_params_clf_optuna_search),
    # (ExtraTreesClassifier, etc_params_clf_optuna_search),
]


# Define hyperparameters for each model
lr_params_reg_optuna_search = {"fit_intercept": [True, False]}
ridge_params_reg_optuna_search = {"alpha": [0.5, 1.0]}
lasso_params_reg_optuna_search = {"alpha": [0.5, 1.0]}
elastic_params_reg_optuna_search = {"alpha": [0.5, 1.0], "l1_ratio": [0.5, 0.7]}
xgb_params_reg_optuna_search = {"max_depth": [3, 5], "learning_rate": [0.01, 0.1]}
lgbm_params_reg_optuna_search = {"max_depth": [3, 5], "learning_rate": [0.01, 0.1]}
cb_params_reg_optuna_search = {"depth": [3, 5], "learning_rate": [0.01, 0.1]}
svr_params_reg_optuna_search = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "C": [0.5, 1.0],
}
lsvr_params_reg_optuna_search = {"C": [0.5, 1.0]}
knr_params_reg_optuna_search = {
    "n_neighbors": [
        5,
        7,
        10,
        15,
        20,
    ],  # Increasing the number of neighbors can help in making the model more generalized.
    "weights": [
        "uniform",
        "distance",
    ],  # The 'distance' option can give more importance to closer instances, which may help reduce overfitting.
    "algorithm": [
        "auto",
        "ball_tree",
        "kd_tree",
        "brute",
    ],  # The algorithm used to compute the nearest neighbors can sometimes have an effect on overfitting, but it generally depends more on the dataset.
    "p": [
        1,
        2,
    ],  # This corresponds to the power parameter for the Minkowski metric. 1 is for manhattan_distance and 2 for euclidean_distance.
}
dtr_params_reg_optuna_search = {
    "max_depth": [5, 10],
    "min_samples_split": [3, 5],
    "min_samples_leaf": [2, 3],
}
rfr_params_reg_optuna_search = {"n_estimators": [50, 200], "max_depth": [10, 20]}
gbr_params_reg_optuna_search = {
    "n_estimators": [50, 200],
    "max_depth": [3, 10],
    "learning_rate": [0.01, 0.1],
}
etr_params_reg_optuna_search = {"n_estimators": [50, 200], "max_depth": [10, 20]}
abr_params_reg_optuna_search = {"n_estimators": [50, 200], "learning_rate": [0.01, 0.1]}
mlpr_params_reg_optuna_search = {
    "hidden_layer_sizes": [(100,)],
    "activation": ["relu", "tanh"],
}
sgr_params_reg_optuna_search = {
    "loss": ["huber", "epsilon_insensitive"],
    "penalty": ["l2", "l1", "elasticnet"],
    "alpha": [0.001, 0.01],
}

# Put models and hyperparameters into a list of tuples
estimators_params_optuna_regs = [
    (LinearRegression, lr_params_reg_optuna_search),
    (Ridge, ridge_params_reg_optuna_search),
    (Lasso, lasso_params_reg_optuna_search),
    (ElasticNet, elastic_params_reg_optuna_search),
    (XGBRegressor, xgb_params_reg_optuna_search),
    (LGBMRegressor, lgbm_params_reg_optuna_search),
    (CatBoostRegressor, cb_params_reg_optuna_search),
    # TODO underfit
    # (SVR, svr_params_reg_optuna_search),
    # (LinearSVR, lsvr_params_reg_optuna_search),
    # TODO overfit
    # (KNeighborsRegressor, knr_params_reg_optuna_search),
    # TODO underfit
    # (DecisionTreeRegressor, dtr_params_reg_optuna_search),
    (RandomForestRegressor, rfr_params_reg_optuna_search),
    # (GradientBoostingRegressor, gbr_params_reg_optuna_search),
    # (ExtraTreesRegressor, etr_params_reg_optuna_search),
    (AdaBoostRegressor, abr_params_reg_optuna_search),
    # (SGDRegressor, sgr_params_reg_optuna_search)
]


# Define hyperparameters for each model for optunasearchcv clfs
lr_params_optunasearchcv_clf = {
    "C": optuna.distributions.FloatDistribution(0.001, 10.0)
}
rf_params_optunasearchcv_clf = {
    "n_estimators": optuna.distributions.IntDistribution(100, 200),
    "max_depth": optuna.distributions.IntDistribution(3, 20),
}
xgb_params_optunasearchcv_clf = {
    "n_estimators": optuna.distributions.IntDistribution(50, 200),
    "max_depth": optuna.distributions.IntDistribution(3, 15),
    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.1),
}
cb_params_optunasearchcv_clf = {
    "iterations": optuna.distributions.IntDistribution(50, 200),
    "depth": optuna.distributions.IntDistribution(4, 8),
    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.1),
}
lgbm_params_optunasearchcv_clf = {
    "n_estimators": optuna.distributions.IntDistribution(50, 200),
    "max_depth": optuna.distributions.IntDistribution(10, 20),
    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.1),
}
brf_params_optunasearchcv_clf = {
    "n_estimators": optuna.distributions.IntDistribution(50, 200),
    "max_depth": optuna.distributions.IntDistribution(10, 20),
}
gbc_params_optunasearchcv_clf = {
    "n_estimators": optuna.distributions.IntDistribution(50, 200),
    "learning_rate": optuna.distributions.FloatDistribution(0.001, 0.1),
    "max_depth": optuna.distributions.IntDistribution(3, 10),
}
abc_params_optunasearchcv_clf = {
    "n_estimators": optuna.distributions.IntDistribution(50, 200),
    "learning_rate": optuna.distributions.FloatDistribution(0.001, 0.1),
}
svc_params_optunasearchcv_clf = {
    "C": optuna.distributions.FloatDistribution(0.1, 10),
    "gamma": optuna.distributions.FloatDistribution(0.001, 0.1),
}
knn_params_optunasearchcv_clf = {
    "n_neighbors": optuna.distributions.IntDistribution(3, 7),
    "weights": optuna.distributions.CategoricalDistribution(
        choices=("uniform", "distance")
    ),
}
dtc_params_optunasearchcv_clf = {
    "criterion": optuna.distributions.CategoricalDistribution(
        choices=("gini", "entropy")
    ),
    "max_depth": optuna.distributions.IntDistribution(5, 10),
}
etc_params_optunasearchcv_clf = {
    "n_estimators": optuna.distributions.IntDistribution(50, 200),
    "max_depth": optuna.distributions.IntDistribution(5, 10),
}
gnb_params_optunasearchcv_clf = {}

# Put models and hyperparameters into a list of tuples
estimators_params_optunasearchcv_clfs = [
    (LogisticRegression, lr_params_optunasearchcv_clf),
    (RandomForestClassifier, rf_params_optunasearchcv_clf),
    (XGBClassifier, xgb_params_optunasearchcv_clf),
    (CatBoostClassifier, cb_params_optunasearchcv_clf),
    (LGBMClassifier, lgbm_params_optunasearchcv_clf),
    (BalancedRandomForestClassifier, brf_params_optunasearchcv_clf),
    # (GradientBoostingClassifier, gbc_params_optunasearchcv_clf),
    (AdaBoostClassifier, abc_params_optunasearchcv_clf),
    # (SVC, svc_params_optunasearchcv_clf),
    # (KNeighborsClassifier, knn_params_optunasearchcv_clf),
    # (DecisionTreeClassifier, dtc_params_optunasearchcv_clf),
    # (ExtraTreesClassifier, etc_params_optunasearchcv_clf),
    # (GaussianNB, gnb_params_optunasearchcv_clf)
]


# Define hyperparameters for each model for optunasearchcv regs
lr_params_optunasearchcv_reg = {
    "fit_intercept": optuna.distributions.CategoricalDistribution(choices=(True, False))
}
ridge_params_optunasearchcv_reg = {
    "alpha": optuna.distributions.FloatDistribution(0.5, 1.0)
}
lasso_params_optunasearchcv_reg = {
    "alpha": optuna.distributions.FloatDistribution(0.5, 1.0)
}
elastic_params_optunasearchcv_reg = {
    "alpha": optuna.distributions.FloatDistribution(0.5, 1.0),
    "l1_ratio": optuna.distributions.FloatDistribution(0.5, 0.7),
}
xgb_params_optunasearchcv_reg = {
    "max_depth": optuna.distributions.IntDistribution(3, 5),
    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.1),
}
lgbm_params_optunasearchcv_reg = {
    "max_depth": optuna.distributions.IntDistribution(3, 5),
    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.1),
}
cb_params_optunasearchcv_reg = {
    "depth": optuna.distributions.IntDistribution(3, 5),
    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.10),
}
svr_params_optunasearchcv_reg = {
    "kernel": optuna.distributions.CategoricalDistribution(
        choices=("linear", "poly", "rbf", "sigmoid")
    ),
    "C": optuna.distributions.FloatDistribution(0.1, 1.0),
}
lsvr_params_optunasearchcv_reg = {"C": optuna.distributions.FloatDistribution(0.5, 1.0)}
knr_params_optunasearchcv_reg = {
    "n_neighbors": optuna.distributions.IntDistribution(
        30, 50
    ),  # Increasing the number of neighbors can help in making the model more generalized.
    "weights": optuna.distributions.CategoricalDistribution(
        choices=("uniform", "distance")
    ),  # The 'distance' option can give more importance to closer instances, which may help reduce overfitting.
    "algorithm": optuna.distributions.CategoricalDistribution(
        choices=("auto", "ball_tree", "kd_tree", "brute")
    ),  # The algorithm used to compute the nearest neighbors can sometimes have an effect on overfitting, but it generally depends more on the dataset.
    "p": optuna.distributions.IntDistribution(
        1, 2
    ),  # This corresponds to the power parameter for the Minkowski metric. 1 is for manhattan_distance and 2 for euclidean_distance.
}
dtr_params_optunasearchcv_reg = {
    "max_depth": optuna.distributions.IntDistribution(3, 5),
    "min_samples_split": optuna.distributions.IntDistribution(2, 5),
    "min_samples_leaf": optuna.distributions.IntDistribution(2, 5),
    "max_features": optuna.distributions.CategoricalDistribution(
        choices=("auto", "sqrt", "log2")
    ),
    "max_leaf_nodes": optuna.distributions.IntDistribution(5, 10),
}
rfr_params_optunasearchcv_reg = {
    "n_estimators": optuna.distributions.IntDistribution(50, 200),
    "max_depth": optuna.distributions.IntDistribution(10, 20),
}
gbr_params_optunasearchcv_reg = {
    "n_estimators": optuna.distributions.IntDistribution(50, 200),
    "max_depth": optuna.distributions.IntDistribution(3, 10),
    "learning_rate": optuna.distributions.FloatDistribution(0.001, 0.1),
}
etr_params_optunasearchcv_reg = {
    "n_estimators": optuna.distributions.IntDistribution(50, 200),
    "max_depth": optuna.distributions.IntDistribution(10, 20),
}
abr_params_optunasearchcv_reg = {
    "n_estimators": optuna.distributions.IntDistribution(50, 200),
    "learning_rate": optuna.distributions.FloatDistribution(0.001, 0.1),
}
sgr_params_optunasearchcv_reg = {
    "loss": optuna.distributions.CategoricalDistribution(
        choices=("squared_loss", "huber", "epsilon_insensitive")
    ),
    "penalty": optuna.distributions.CategoricalDistribution(
        choices=("l2", "l1", "elasticnet")
    ),
    "alpha": optuna.distributions.FloatDistribution(0.0001, 0.01),
}

# Put models and hyperparameters into a list of tuples
estimators_params_optunasearchcv_regs = [
    (LinearRegression, lr_params_optunasearchcv_reg),
    (Ridge, ridge_params_optunasearchcv_reg),
    (Lasso, lasso_params_optunasearchcv_reg),
    (ElasticNet, elastic_params_optunasearchcv_reg),
    (XGBRegressor, xgb_params_optunasearchcv_reg),
    (LGBMRegressor, lgbm_params_optunasearchcv_reg),
    (CatBoostRegressor, cb_params_optunasearchcv_reg),
    # TODO underfit
    # (SVR, svr_params_optunasearchcv_reg),
    # (LinearSVR, lsvr_params_optunasearchcv_reg),
    # TODO overfit
    # (KNeighborsRegressor, knr_params_optunasearchcv_reg),
    # TODO underfit
    # (DecisionTreeRegressor, dtr_params_optunasearchcv_reg),
    (RandomForestRegressor, rfr_params_optunasearchcv_reg),
    # (GradientBoostingRegressor, gbr_params_optunasearchcv_reg),
    # (ExtraTreesRegressor, etr_params_optunasearchcv_reg),
    (AdaBoostRegressor, abr_params_optunasearchcv_reg),
    # (SGDRegressor, sgr_params_optunasearchcv_reg)
]

# Define hyperparameters for tunesearch classification
lr_params_tunesearch_clfs = {"C": tune.uniform(0.001, 10.0)}
rf_params_tunesearch_clfs = {
    "n_estimators": tune.randint(100, 200),
    "max_depth": tune.randint(3, 20),
}
xgb_params_tunesearch_clfs = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.randint(3, 15),
    "learning_rate": tune.uniform(0.01, 0.1),
}
cb_params_tunesearch_clfs = {
    "iterations": tune.randint(50, 200),
    "depth": tune.randint(4, 8),
    "learning_rate": tune.uniform(0.01, 0.1),
}
lgbm_params_tunesearch_clfs = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.randint(10, 20),
    "learning_rate": tune.uniform(0.01, 0.1),
}
brf_params_tunesearch_clfs = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.randint(10, 20),
}
gbc_params_tunesearch_clfs = {
    "n_estimators": tune.randint(50, 200),
    "learning_rate": tune.uniform(0.001, 0.1),
    "max_depth": tune.randint(3, 10),
}
abc_params_tunesearch_clfs = {
    "n_estimators": tune.randint(50, 200),
    "learning_rate": tune.uniform(0.001, 0.1),
}
svc_params_tunesearch_clfs = {
    "C": tune.uniform(0.1, 10),
    "gamma": tune.uniform(0.001, 0.1),
}
knn_params_tunesearch_clfs = {
    "n_neighbors": tune.randint(3, 7),
    "weights": tune.choice(["uniform", "distance"]),
}
dtc_params_tunesearch_clfs = {
    "criterion": tune.choice(["gini", "entropy"]),
    "max_depth": tune.randint(5, 10),
}
etc_params_tunesearch_clfs = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.randint(5, 10),
}
# gnb_params_tunesearch_clfs = {}

# Put models and hyperparameters into a list of tuples
estimators_params_tunesearch_clfs = [
    (LogisticRegression, lr_params_tunesearch_clfs),
    (RandomForestClassifier, rf_params_tunesearch_clfs),
    (XGBClassifier, xgb_params_tunesearch_clfs),
    (CatBoostClassifier, cb_params_tunesearch_clfs),
    (LGBMClassifier, lgbm_params_tunesearch_clfs),
    (BalancedRandomForestClassifier, brf_params_tunesearch_clfs),
    (GradientBoostingClassifier, gbc_params_tunesearch_clfs),
    (AdaBoostClassifier, abc_params_tunesearch_clfs),
    (SVC, svc_params_tunesearch_clfs),
    # (KNeighborsClassifier, knn_params_tunesearch_clfs),
    (DecisionTreeClassifier, dtc_params_tunesearch_clfs),
    (ExtraTreesClassifier, etc_params_tunesearch_clfs),
    # (GaussianNB, gnb_params_tunesearch_clfs),
]

# Define hyperparameters for tune regression
lr_params_tunesearch_regs = {"fit_intercept": tune.choice([True, False])}
ridge_params_tunesearch_regs = {"alpha": tune.uniform(0.5, 1.0)}
lasso_params_tunesearch_regs = {"alpha": tune.uniform(0.5, 1.0)}
elastic_params_tunesearch_regs = {
    "alpha": tune.uniform(0.5, 1.0),
    "l1_ratio": tune.uniform(0.5, 0.7),
}
xgb_params_tunesearch_regs = {
    "max_depth": tune.randint(3, 5),
    "learning_rate": tune.uniform(0.01, 0.1),
}
lgbm_params_tunesearch_regs = {
    "max_depth": tune.randint(3, 5),
    "learning_rate": tune.uniform(0.01, 0.1),
}
cb_params_tunesearch_regs = {
    "depth": tune.randint(3, 5),
    "learning_rate": tune.uniform(0.01, 0.10),
}
svr_params_tunesearch_regs = {
    "kernel": tune.choice(["linear", "poly", "rbf", "sigmoid"]),
    "C": tune.uniform(0.1, 1.0),
}
lsvr_params_tunesearch_regs = {"C": tune.uniform(0.5, 1.0)}
knr_params_tunesearch_regs = {
    "n_neighbors": tune.randint(
        30, 50
    ),  # Increasing the number of neighbors can help in making the model more generalized.
    "weights": tune.choice(
        ["uniform", "distance"]
    ),  # The 'distance' option can give more importance to closer instances, which may help reduce overfitting.
    "algorithm": tune.choice(
        ["auto", "ball_tree", "kd_tree", "brute"]
    ),  # The algorithm used to compute the nearest neighbors can sometimes have an effect on overfitting, but it generally depends more on the dataset.
    "p": tune.randint(
        1, 2
    ),  # This corresponds to the power parameter for the Minkowski metric. 1 is for manhattan_distance and 2 for euclidean_distance.
}
dtr_params_tunesearch_regs = {
    "max_depth": tune.randint(3, 5),
    "min_samples_split": tune.randint(2, 5),
    "min_samples_leaf": tune.randint(2, 5),
    "max_features": tune.choice(["auto", "sqrt", "log2"]),
    "max_leaf_nodes": tune.randint(5, 10),
}
rfr_params_tunesearch_regs = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.randint(10, 20),
}
gbr_params_tunesearch_regs = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.randint(3, 10),
    "learning_rate": tune.uniform(0.001, 0.1),
}
etr_params_tunesearch_regs = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.randint(10, 20),
}
abr_params_tunesearch_regs = {
    "n_estimators": tune.randint(50, 200),
    "learning_rate": tune.uniform(0.001, 0.1),
}
sgr_params_tunesearch_regs = {
    "loss": tune.choice(["squared_loss", "huber", "epsilon_insensitive"]),
    "penalty": tune.choice({"l2", "l1", "elasticnet"}),
    "alpha": tune.uniform(0.0001, 0.01),
}

# Put models and hyperparameters into a list of tuples
estimators_params_tunesearch_regs = [
    (LinearRegression, lr_params_tunesearch_regs),
    (Ridge, ridge_params_tunesearch_regs),
    (Lasso, lasso_params_tunesearch_regs),
    (ElasticNet, elastic_params_tunesearch_regs),
    (XGBRegressor, xgb_params_tunesearch_regs),
    (LGBMRegressor, lgbm_params_tunesearch_regs),
    (CatBoostRegressor, cb_params_tunesearch_regs),
    # TODO underfit
    # (SVR, svr_params_tunesearch_regs),
    # (LinearSVR, lsvr_params_tunesearch_regs),
    # TODO overfit
    # (KNeighborsRegressor, knr_params_tunesearch_regs),
    # TODO underfit
    # (DecisionTreeRegressor, dtr_params_tunesearch_regs),
    (RandomForestRegressor, rfr_params_tunesearch_regs),
    # (GradientBoostingRegressor, gbr_params_tunesearch_regs),
    (ExtraTreesRegressor, etr_params_tunesearch_regs),
    (AdaBoostRegressor, abr_params_tunesearch_regs),
    # (SGDRegressor, sgr_params_tunesearch_regs),
]
