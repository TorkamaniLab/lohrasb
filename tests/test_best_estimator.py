import pandas as pd
import optuna
from lohrasb.project_conf import ROOT_PROJECT
from optuna.pruners import HyperbandPruner
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.model_selection import KFold, train_test_split
from lohrasb.best_estimator import BaseModel
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.linear_model import *
from sklearn.svm import *
from xgboost import *
from sklearn.linear_model import *
from catboost import *
from lightgbm import *
from sklearn.neural_network import *
from imblearn.ensemble import *
from sklearn.ensemble import *
# prepare data for tests
try:
    print(ROOT_PROJECT / "lohrasb" / "data" / "data.csv")
    data = pd.read_csv(ROOT_PROJECT / "lohrasb" / "data" / "data.csv")
except Exception as e:
    print(ROOT_PROJECT / "lohrasb" / "data" / "data.csv")
    print(e)

X = data.loc[:, data.columns != "default payment next month"]
y = data.loc[:, data.columns == "default payment next month"]
y = y.values.ravel()

X = X.select_dtypes(include=np.number)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)

# function for classifications
def run_classifiers(obj, X_train, y_train, X_test, y_test):
    obj.fit(X_train, y_train)
    y_preds = obj.predict(X_test)
    pred_labels = np.rint(y_preds)
    return f1_score(y_test, pred_labels)
    
# function for regressions
def run_regressors(obj, X_train, y_train, X_test, y_test):
    obj.fit(X_train, y_train)
    y_preds = obj.predict(X_test)
    return mean_absolute_error(y_test, y_preds)


models_classifiers = {
    "XGBClassifier": {
        "eval_metric": ["auc"],
        "max_depth": [4, 5],
    },
    "LGBMClassifier": {
    "max_depth": [1, 12]
    },
    "CatBoostClassifier": {
        "depth": [5, 6],
        "logging_level": ["Silent"],

    },
    "MLPClassifier": {
        "activation": ["relu"],
        "alpha": [0.0001],
    },
    "BalancedRandomForestClassifier": {
        "n_estimators": [100, 200],
        "min_impurity_decrease": [0.0, 0.1],
    },
}


models_regressors = {
    "XGBRegressor": {
        "max_depth": [4, 5],
        "min_child_weight": [0.1, 0.9],
        "gamma": [1, 9],
    },
    "LinearRegression": {
        "fit_intercept": [True, False],
    },
    "RandomForestRegressor": {
        "max_depth": [4, 5],
    },
    "MLPRegressor": {
        "activation": ["logistic"],
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": [0.0001],
    },
}


def test_best_estimator():
    """Test feature scally selector add"""
    # functions for classifiers
    def run_gird_classifiers(pause_iteration=False):
        for model in models_classifiers:
            obj = BaseModel.bestmodel_factory.using_gridsearch(
                estimator=eval(model + "()"),
                estimator_params=models_classifiers[model],
                measure_of_accuracy="f1",
                verbose=3,
                n_jobs=-1,
                random_state=42,
                cv=KFold(2),
            )
            # run classifiers
            f1 = run_classifiers(obj, X_train, y_train, X_test, y_test)
            assert f1>= 0.0
    def run_random_classifiers(pause_iteration=False):
        for model in models_classifiers:
            obj = BaseModel.bestmodel_factory.using_randomsearch(
                estimator=eval(model + "()"),
                estimator_params=models_classifiers[model],
                measure_of_accuracy="f1",
                verbose=3,
                n_jobs=-1,
                random_state=42,
                cv=KFold(2),
                n_iter=1,
            )
            # run classifiers
            f1 = run_classifiers(obj, X_train, y_train, X_test, y_test)
            assert f1>= 0.0
    def run_optuna_classifiers(pause_iteration=False):
        for model in models_classifiers:
            obj = BaseModel.bestmodel_factory.using_optuna(
                estimator=eval(model + "()"),
                estimator_params=models_classifiers[model],
                measure_of_accuracy="f1",
                verbose=3,
                n_jobs=-1,
                random_state=42,
                # optuna params
                # optuna study init params
                study=optuna.create_study(
                    storage=None,
                    sampler=TPESampler(),
                    pruner=HyperbandPruner(),
                    study_name=None,
                    direction="maximize",
                    load_if_exists=False,
                    directions=None,
                ),
                # optuna optimization params
                study_optimize_objective=None,
                study_optimize_objective_n_trials=10,
                study_optimize_objective_timeout=600,
                study_optimize_n_jobs=-1,
                study_optimize_catch=(),
                study_optimize_callbacks=None,
                study_optimize_gc_after_trial=False,
                study_optimize_show_progress_bar=False,
            )
            # run classifiers
            f1 = run_classifiers(obj, X_train, y_train, X_test, y_test)
            assert f1>= 0.0
    
    # functions for regressors
    def run_gird_regressors(pause_iteration=False):
        for model in models_regressors:
            obj = BaseModel.bestmodel_factory.using_gridsearch(
                estimator=eval(model + "()"),
                estimator_params=models_regressors[model],
                measure_of_accuracy="mean_absolute_error",
                verbose=3,
                n_jobs=-1,
                random_state=42,
                cv=KFold(2),
            )
            # run regressors
            mean_absolute_error = run_regressors(obj, X_train, y_train, X_test, y_test)
            assert mean_absolute_error >= 0.0
    
    def run_random_regressors(pause_iteration=False):
        for model in models_regressors:
            obj = BaseModel.bestmodel_factory.using_randomsearch(
                estimator=eval(model + "()"),
                estimator_params=models_regressors[model],
                measure_of_accuracy="mean_absolute_error",
                verbose=3,
                n_jobs=-1,
                random_state=42,
                cv=KFold(2),
                n_iter=1,
            )
            # run regressors
            mean_absolute_error = run_regressors(obj, X_train, y_train, X_test, y_test)
            assert mean_absolute_error >= 0.0

    def run_optuna_regressors(pause_iteration=False):
        for model in models_regressors:
            obj = BaseModel.bestmodel_factory.using_optuna(
            estimator=eval(model + "()"),
            estimator_params=models_regressors[model],
            measure_of_accuracy="mean_absolute_error",
            verbose=3,
            n_jobs=-1,
            random_state=42,
            # optuna params
            # optuna study init params
            study=optuna.create_study(
                storage=None,
                sampler=TPESampler(),
                pruner=HyperbandPruner(),
                study_name=None,
                direction="minimize",
                load_if_exists=False,
                directions=None,
            ),
            # optuna optimization params
            study_optimize_objective=None,
            study_optimize_objective_n_trials=10,
            study_optimize_objective_timeout=600,
            study_optimize_n_jobs=-1,
            study_optimize_catch=(),
            study_optimize_callbacks=None,
            study_optimize_gc_after_trial=False,
            study_optimize_show_progress_bar=False,
            )
            # run regressors
            mean_absolute_error = run_regressors(obj, X_train, y_train, X_test, y_test)
            assert mean_absolute_error >= 0.0

    # run tests for classifiers
    run_gird_classifiers()
    run_random_classifiers()
    run_optuna_classifiers()

    # run tests for regressors
    run_gird_regressors()
    run_random_regressors()
    run_optuna_regressors()

# run all tests in once
test_best_estimator()


