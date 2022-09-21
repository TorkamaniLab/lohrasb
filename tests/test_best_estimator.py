import pandas as pd
import xgboost
import optuna
import catboost
from lohrasb.project_conf import ROOT_PROJECT
from optuna.pruners import HyperbandPruner
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.model_selection import KFold, train_test_split
from lohrasb.best_estimator import BaseModel
import numpy as np
from sklearn.linear_model import *
from sklearn.svm import *
from xgboost import *
from sklearn.linear_model import *
from catboost import *
from lightgbm import *
from sklearn.neural_network import *
from imblearn.ensemble import *
from sklearn.ensemble import *

def test_best_estimator():
    """Test feature scally selector add"""

    # SFC_XGB_OPTUNA = BaseModel(
    #     estimator=xgboost.XGBClassifier(),
    #     estimator_params={
    #         "max_depth": [4, 5],
    #         "min_child_weight": [0.1, 0.9],
    #         "gamma": [1, 9],
    #         "booster": ["gbtree"],
    #     },
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

    # SFC_XGBREG_OPTUNA = BaseModel(
    #     estimator=xgboost.XGBRegressor(),
    #     estimator_params={
    #         "max_depth": [4, 5],
    #         #"min_child_weight": [0.1, 0.9],
    #         #"gamma": [1, 9],
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


    # SFC_CAT_OPTUNA = BaseModel(
    #     estimator=catboost.CatBoostClassifier(),
    #     estimator_params={
    #         #"objective": ["Logloss", "CrossEntropy"],
    #         "depth": [1, 12],
    #         "boosting_type": ["Ordered", "Plain"],
    #         "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"]
    
    #     },
    #     hyper_parameter_optimization_method="optuna",
    #     measure_of_accuracy="f1",
    #     test_size=0.33,
    #     cv=KFold(n_splits=3, random_state=42, shuffle=True),
    #     with_stratified=True,
    #     verbose=3,
    #     random_state=42,
    #     n_jobs=-1,
    #     n_iter=100,
    #     eval_metric="AUC",
    #     number_of_trials=10,
    #     sampler=TPESampler(),
    #     pruner=HyperbandPruner(),
    # )


    # SFC_CATREG_OPTUNA = BaseModel(
    #     estimator=catboost.CatBoostRegressor(),
    #     estimator_params={
    #         "depth": [1, 12]
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
    #     eval_metric="RMSE",
    #     number_of_trials=10,
    #     sampler=TPESampler(),
    #     pruner=HyperbandPruner(),
    # )


    # SFC_GRID = BaseModel(
    #     estimator=xgboost.XGBClassifier(),
    #     estimator_params={
    #         "eval_metric":['auc'],
    #         "max_depth": [4, 5],
    #         "min_child_weight": [0.1, 0.9],
    #         "gamma": [1, 9],
    #     },
    #     measure_of_accuracy="f1",
    #     hyper_parameter_optimization_method="grid",
    #     test_size=0.33,
    #     with_stratified=True,
    #     verbose=3,
    #     n_jobs=-1,
    #     random_state=42,
    #     n_iter=100,
    #     cv=KFold(n_splits=3,random_state=42,shuffle=True),
    #     # optuna params
    #     # optuna study init params
    #     study=optuna.create_study(
    #         storage=None,
    #         sampler=TPESampler(),
    #         pruner=HyperbandPruner(),
    #         study_name=None,
    #         direction="maximize",
    #         load_if_exists=False,
    #         directions=None,
    #     ),
    #     # optuna optimization params
    #     study_optimize_objective=None,
    #     study_optimize_objective_n_trials=100,
    #     study_optimize_objective_timeout=600,
    #     study_optimize_n_jobs=-1,
    #     study_optimize_catch=(),
    #     study_optimize_callbacks=None,
    #     study_optimize_gc_after_trial=False,
    #     study_optimize_show_progress_bar=False,


    # )

    # SFC_RANDOM = BaseModel(
    #     estimator=xgboost.XGBClassifier(),
    #     estimator_params={
    #         "max_depth": [4, 5],
    #         "min_child_weight": [0.1, 0.9],
    #         "gamma": [1, 9],
    #         "booster": ["gbtree"],
    #     },
    #     measure_of_accuracy="f1",
    #     hyper_parameter_optimization_method="random",
    #     test_size=0.33,
    #     with_stratified=True,
    #     verbose=3,
    #     n_jobs=-1,
    #     random_state=42,
    #     n_iter=100,
    #     cv=KFold(n_splits=3,random_state=42,shuffle=True),

    #     # optuna params
    #     # optuna study init params
    #     study=optuna.create_study(
    #         storage=None,
    #         sampler=TPESampler(),
    #         pruner=HyperbandPruner(),
    #         study_name="TEST XGBClassifier",
    #         direction="maximize",
    #         load_if_exists=False,
    #         directions=None,
    #     ),
    #     # optuna optimization params
    #     study_optimize_objective=None,
    #     study_optimize_objective_n_trials=100,
    #     study_optimize_objective_timeout=600,
    #     study_optimize_n_jobs=-1,
    #     study_optimize_catch=(),
    #     study_optimize_callbacks=None,
    #     study_optimize_gc_after_trial=False,
    #     study_optimize_show_progress_bar=False,

    # )

    SFC_OPTUNA_XGB_CLASSIFIER = BaseModel(
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 5],
        },
         measure_of_accuracy="f1",
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
        study_optimize_objective_n_trials=100,
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs=-1,
        study_optimize_catch=(),
        study_optimize_callbacks=None,
        study_optimize_gc_after_trial=False,
        study_optimize_show_progress_bar=False,
    )


    try:
        data = pd.read_csv(ROOT_PROJECT / "lohrasb"  / "data" / "data.csv")
    except:
        data = pd.read_csv("/home/circleci/project/data/data.csv")
    print(data.columns.to_list())

    X = data.loc[:, data.columns != "default payment next month"]
    y = data.loc[:, data.columns == "default payment next month"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0
    )

    # SFC_CAT_OPTUNA.fit(X_train, y_train)
    # y_preds_catboost_classifier = SFC_CAT_OPTUNA.predict(X_test)
    # y_preds_catboost_classifier = np.rint(y_preds_catboost_classifier)
    # print(len(y_preds_catboost_classifier))


    #SFC_CATREG_OPTUNA.fit(X_train, y_train)
    #y_preds_catboost_regressor = SFC_CATREG_OPTUNA.predict(X_test)
    #print(len(y_preds_catboost_regressor))


    # SFC_XGB_OPTUNA.fit(X_train, y_train)
    # y_preds_xgboost_classifier = SFC_XGB_OPTUNA.predict(X_test)
    # y_preds_xgboost_classifier = np.rint(y_preds_xgboost_classifier)
    # print(len(y_preds_xgboost_classifier))


    # SFC_XGBREG_OPTUNA.fit(X_train, y_train)
    # y_preds_xgboost_regressor = SFC_XGBREG_OPTUNA.predict(X_test)
    # print(len(y_preds_xgboost_regressor))


    # SFC_GRID.fit(X_train, y_train)
    # y_preds_xgboost_classifier_grid = SFC_GRID.predict(X_test)
    # y_preds_xgboost_classifier_grid = np.rint(y_preds_xgboost_classifier_grid)
    # print(len(y_preds_xgboost_classifier_grid))

    # SFC_RANDOM.fit(X_train, y_train)
    # y_preds_xgboost_classifier_random = SFC_RANDOM.predict(X_test)
    # y_preds_xgboost_classifier_random = np.rint(y_preds_xgboost_classifier_random)
    # print(len(y_preds_xgboost_classifier_random))


    SFC_OPTUNA_XGB_CLASSIFIER.fit(X_train, y_train)
    y_preds_xgboost_classifier_optuna= SFC_OPTUNA_XGB_CLASSIFIER.predict(X_test)
    y_preds_xgboost_classifier_optuna = np.rint(y_preds_xgboost_classifier_optuna)
    print(len(y_preds_xgboost_classifier_optuna))


    #assert len(y_preds_catboost_regressor)==9900#['PAY_0', 'LIMIT_BAL', 'PAY_AMT2', 'BILL_AMT1']
    #assert len(y_preds_xgboost_regressor)==9900#['PAY_0', 'LIMIT_BAL', 'PAY_AMT2', 'BILL_AMT1']
    #assert len(y_preds_xgboost_regressor)==9900#['PAY_0', 'LIMIT_BAL', 'PAY_AMT2', 'BILL_AMT1']
    assert len(y_preds_xgboost_classifier_optuna)==9900#['PAY_0', 'LIMIT_BAL', 'PAY_AMT2', 'BILL_AMT1']
    # assert len(y_preds_xgboost_classifier_grid)==9900#['PAY_0', 'LIMIT_BAL', 'PAY_AMT2', 'BILL_AMT1']
    # assert len(y_preds_xgboost_classifier_random)==9900#['PAY_0', 'LIMIT_BAL', 'PAY_AMT2', 'BILL_AMT1']


    #assert len(pred_labels)==4#['PAY_0', 'LIMIT_BAL', 'PAY_AMT2', 'BILL_AMT1']
    #assert len(pred_labels)==4#['PAY_0', 'LIMIT_BAL', 'PAY_AMT2', 'BILL_AMT1']
    #assert len(pred_labels)==4#['PAY_0', 'LIMIT_BAL', 'PAY_AMT2', 'BILL_AMT1']
    #assert len(pred_labels) == 4


