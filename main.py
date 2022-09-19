# impelemnt various examples for testing purposes

import pandas as pd
import numpy as np
import xgboost
import catboost
from optuna.pruners import HyperbandPruner
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.model_selection import KFold, train_test_split
from lohrasb.best_estimator import BaseModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import lightgbm
from lohrasb.project_conf import ROOT_PROJECT
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
import optuna

# SFC_XGBREG_OPTUNA = BaseModel(
#     estimator=xgboost.XGBRegressor(),
#     estimator_params={
#         "max_depth": [4, 5],
#         # "min_child_weight": [0.1, 0.9],
#         # "gamma": [1, 9],
#     },
#     hyper_parameter_optimization_method="optuna",
#     measure_of_accuracy="r2",
#     test_size=0.33,
#     cv=KFold(n_splits=3, random_state=42, shuffle=True),
#     with_stratified=False,
#     verbose=3,
#     random_state=42,
#     n_jobs=-1,
#     n_iter=100,
#     eval_metric="rmse",
#     number_of_trials=10,
#     sampler=TPESampler(),
#     pruner=HyperbandPruner(),
# )

SFC_XGBCLS_OPTUNA = BaseModel(
    estimator=xgboost.XGBRegressor(),
        estimator_params={
            "eval_metric" :['auc'],
            "max_depth": [4, 5],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
            "booster": ["gbtree", "gblinear", "dart"],
        },
         measure_of_accuracy="mean_squared_error",
        hyper_parameter_optimization_method="optuna",
        test_size=0.33,
        with_stratified=True,
        verbose=3,
        n_jobs=-1,
        random_state=42,
        n_iter=100,
        cv=KFold(n_splits=3,random_state=42,shuffle=True),

        # optuna params
        # optuna study init params
        study=optuna.create_study(
            storage=None,
            sampler=TPESampler(),
            pruner=HyperbandPruner(),
            study_name="TEST XGBClassifier",
            direction="maximize",
            load_if_exists=False,
            directions=None,
        ),
        # optuna optimization params
        study_optimize_objective=None,
        study_optimize_objective_n_trials=15,
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs=-1,
        study_optimize_catch=(),
        study_optimize_callbacks=None,
        study_optimize_gc_after_trial=False,
        study_optimize_show_progress_bar=False,
)

# SFC_RFREG_OPTUNA = BaseModel(
#     estimator=RandomForestRegressor(),
#     estimator_params={"max_depth": [4, 5], "verbose": [0]},
#     hyper_parameter_optimization_method="optuna",
#     measure_of_accuracy="r2",
#     test_size=0.33,
#     cv=KFold(n_splits=3, random_state=42, shuffle=True),
#     with_stratified=False,
#     verbose=3,
#     random_state=42,
#     n_jobs=-1,
#     n_iter=100,
#     eval_metric="no",
#     number_of_trials=10,
#     sampler=TPESampler(),
#     pruner=HyperbandPruner(),
# )

# SFC_LRREG_OPTUNA = BaseModel(
#     estimator=LinearRegression(),
#     estimator_params={
#         "positive": [True,False],
#          },
#     hyper_parameter_optimization_method="optuna",
#     measure_of_accuracy="r2",
#     test_size=0.33,
#     cv=KFold(n_splits=3, random_state=42, shuffle=True),
#     with_stratified=False,
#     verbose=3,
#     random_state=42,
#     n_jobs=-1,
#     n_iter=100,
#     eval_metric="no",
#     number_of_trials=10,
#     sampler=TPESampler(),
#     pruner=HyperbandPruner(),
# )

# SFC_LGBREG_OPTUNA = BaseModel(
#     estimator=lightgbm.LGBMRegressor(),
#     estimator_params={"max_depth": [4, 5]},
#     hyper_parameter_optimization_method="optuna",
#     measure_of_accuracy="r2",
#     test_size=0.33,
#     cv=KFold(n_splits=3, random_state=42, shuffle=True),
#     with_stratified=False,
#     verbose=3,
#     random_state=42,
#     n_jobs=-1,
#     n_iter=100,
#     eval_metric="l1",
#     number_of_trials=10,
#     sampler=TPESampler(),
#     pruner=HyperbandPruner(),
# )

# SFC_RFCLS_OPTUNA = BaseModel(
#     estimator=RandomForestClassifier(),
#     estimator_params={"max_depth": [1, 12]},
#     hyper_parameter_optimization_method="optuna",
#     measure_of_accuracy="f1",
#     test_size=0.33,
#     cv=KFold(n_splits=3, random_state=42, shuffle=True),
#     with_stratified=True,
#     verbose=3,
#     random_state=42,
#     n_jobs=-1,
#     n_iter=100,
#     eval_metric="auc",
#     number_of_trials=10,
#     sampler=TPESampler(),
#     pruner=HyperbandPruner(),
# )
# SFC_LGBCLS_OPTUNA = BaseModel(
#     estimator=lightgbm.LGBMClassifier(),
#     estimator_params={"max_depth": [1, 12]},
#     hyper_parameter_optimization_method="optuna",
#     measure_of_accuracy="f1",
#     test_size=0.33,
#     cv=KFold(n_splits=3, random_state=42, shuffle=True),
#     with_stratified=True,
#     verbose=3,
#     random_state=42,
#     n_jobs=-1,
#     n_iter=100,
#     eval_metric="auc",
#     number_of_trials=10,
#     sampler=TPESampler(),
#     pruner=HyperbandPruner(),
# )

# SFC_BRFCLS_OPTUNA = BaseModel(
#     estimator=BalancedRandomForestClassifier(),
#     estimator_params={"max_depth": [1, 12]},
#     hyper_parameter_optimization_method="optuna",
#     measure_of_accuracy="f1",
#     test_size=0.33,
#     cv=KFold(n_splits=3, random_state=42, shuffle=True),
#     with_stratified=True,
#     verbose=3,
#     random_state=42,
#     n_jobs=-1,
#     n_iter=100,
#     eval_metric="auc",
#     number_of_trials=10,
#     sampler=TPESampler(),
#     pruner=HyperbandPruner(),
# )

# SFC_LGCLS_OPTUNA = BaseModel(
#     estimator=LogisticRegression(),
#     estimator_params={
#         "C": [1, 12]
#         },
#     hyper_parameter_optimization_method="optuna",
#     measure_of_accuracy="f1",
#     test_size=0.33,
#     cv=KFold(n_splits=3, random_state=42, shuffle=True),
#     with_stratified=True,
#     verbose=3,
#     random_state=42,
#     n_jobs=-1,
#     n_iter=100,
#     eval_metric="auc",
#     number_of_trials=10,
#     sampler=TPESampler(),
#     pruner=HyperbandPruner(),
# )


# SFC_SVCCLS_OPTUNA = BaseModel(
#     estimator=SVC(),
#     estimator_params={
#         "C": [1, 2] ,
#         "kernel" : ["rbf"]
#     },
#     hyper_parameter_optimization_method="optuna",
#     measure_of_accuracy="f1",
#     test_size=0.33,
#     cv=KFold(n_splits=3, random_state=42, shuffle=True),
#     with_stratified=True,
#     verbose=3,
#     random_state=42,
#     n_jobs=-1,
#     n_iter=10,
#     eval_metric="auc",
#     number_of_trials=5,
#     sampler=TPESampler(),
#     pruner=HyperbandPruner(),
# )


# SFC_CAT_OPTUNA = BaseModel(
#     estimator=catboost.CatBoostClassifier(),
#     estimator_params={
#         # "objective": ["Logloss", "CrossEntropy"],
#         "depth": [1, 12],
#         "boosting_type": ["Ordered", "Plain"],
#         "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"],
#     },
#     hyper_parameter_optimization_method="optuna",
#     measure_of_accuracy="f1",
#     test_size=0.33,
#     cv=KFold(n_splits=3, random_state=42, shuffle=True),
#     with_stratified=True,
#     verbose=0,
#     random_state=42,
#     n_jobs=-1,
#     n_iter=100,
#     eval_metric="AUC",
#     number_of_trials=10,
#     sampler=TPESampler(),
#     pruner=HyperbandPruner(),
# )

try: 
    print(ROOT_PROJECT / "lohrasb" / "data" / "data.csv")
    data = pd.read_csv(ROOT_PROJECT / "lohrasb" / "data" / "data.csv")
except Exception as e:
    print(ROOT_PROJECT / "lohrasb" / "data" / "data.csv")
    print(e)

print(data.columns.to_list())
X = data.loc[:, data.columns != "default payment next month"]
y = data.loc[:, data.columns == "default payment next month"]

X = X.select_dtypes(include=np.number)

print(data.columns.to_list())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)


# def run_xgboost_regressor():
#     SFC_XGBREG_OPTUNA.fit(X_train, y_train)
#     y_pred = SFC_XGBREG_OPTUNA.predict(X_test)
#     print(y_pred)

def run_xgboost_classifier():
    SFC_XGBCLS_OPTUNA.fit(X_train, y_train)
    y_preds = SFC_XGBCLS_OPTUNA.predict(X_test)
    pred_labels = np.rint(y_preds)
    print(pred_labels)

# def run_randomforest_regressor():
#     SFC_RFREG_OPTUNA.fit(X_train, y_train)
#     y_pred = SFC_RFREG_OPTUNA.predict(X_test)
#     print(y_pred)


# def run_randomforest_classifier():
#     SFC_RFCLS_OPTUNA.fit(X_train, y_train)
#     y_pred = SFC_RFCLS_OPTUNA.predict(X_test)
#     print(y_pred)


# def run_balancedrandomforest_classifier():
#     SFC_BRFCLS_OPTUNA.fit(X_train, y_train)
#     y_pred = SFC_BRFCLS_OPTUNA.predict(X_test)
#     print(y_pred)

# def run_svc_classifier():
#     SFC_SVCCLS_OPTUNA.fit(X_train, y_train)
#     y_pred = SFC_SVCCLS_OPTUNA.predict(X_test)
#     print(y_pred)

# def run_logistic_classifier():
#     SFC_LGCLS_OPTUNA.fit(X_train, y_train)
#     y_pred = SFC_LGCLS_OPTUNA.predict(X_test)
#     print(y_pred)

# def run_catboost_classifier():
#     SFC_CAT_OPTUNA.fit(X_train, y_train)
#     y_preds= SFC_CAT_OPTUNA.predict(X_test)
#     pred_labels = np.rint(y_preds)
#     print(pred_labels)


# def run_lgb_classifier():
#     SFC_LGBCLS_OPTUNA.fit(X_train, y_train)
#     y_pred = SFC_LGBCLS_OPTUNA.predict(X_test)
#     print(y_pred)


# def run_lgb_regressor():
#     SFC_LGBREG_OPTUNA.fit(X_train, y_train)
#     y_pred = SFC_LGBREG_OPTUNA.predict(X_test)
#     print(y_pred)

# def run_linear_regression():
#     SFC_LRREG_OPTUNA.fit(X_train, y_train)
#     y_pred = SFC_LRREG_OPTUNA.predict(X_test)
#     print(y_pred)

if __name__ == "__main__":
    # run random forest regressor on test data
    # run_randomforest_regressor() # OK
    # run random forest classifier on test data
    # run_randomforest_classifier() # OK
    # run balanced random forest classifier on test data
    # run_balancedrandomforest_classifier() # OK
    # run xgboost regressor on test data
    # run_xgboost_regressor() # OK
    run_xgboost_classifier() # OK
    # run_svc_classifier() # OK
    # run_logistic_classifier() # OK
    # run_linear_regression() # OK
    # run catboost classifier on test data
    # run_catboost_classifier() # OK
    # run lgb classifier on test data
    # run_lgb_classifier() # OK
    # run lgb regressor on test data
    # run_lgb_regressor() # OK
