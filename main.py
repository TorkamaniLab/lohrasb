# impelemnt various examples for testing purposes

import pandas as pd
import numpy as np
from optuna.pruners import HyperbandPruner
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.model_selection import KFold, train_test_split
from lohrasb.best_estimator import BaseModel
import optuna
from sklearn.metrics import f1_score, mean_absolute_error
from lohrasb.project_conf import ROOT_PROJECT
from sklearn.linear_model import *
from sklearn.svm import *
from xgboost import *
from sklearn.linear_model import *
from catboost import *
from lightgbm import *
from sklearn.neural_network import *
from imblearn.ensemble import *
from sklearn.ensemble import *
from lohrasb.utils.metrics import CalcMetrics

# initialize CalcMetrics
calc_metric = CalcMetrics(
    y_true=None,
    y_pred=None,
    metric=None,
)


# prepare data for tests
try:
    print(ROOT_PROJECT / "lohrasb" / "data" / "data.csv")
    data = pd.read_csv(ROOT_PROJECT / "lohrasb" / "data" / "data.csv")
except Exception as e:
    print(ROOT_PROJECT / "lohrasb" / "data" / "data.csv")
    print(e)

print(data.columns.to_list())
X = data.loc[:, data.columns != "default payment next month"]
y = data.loc[:, data.columns == "default payment next month"]
y = y.values.ravel()

X = X.select_dtypes(include=np.number)

print(data.columns.to_list())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)

print(y_train)
print(y_test)

# functions for classifications
def run_classifiers(obj, X_train, y_train, X_test, y_test, measure_of_accuracy):
    """
    A function to get best estimator fit it again, calculate predictions
    and calculate f1 score.

    Parameters
    ----------
    obj: Object
        Best estimator for classification
    X_train: pd.DataFrame
        Training dataframe
    y_train : pd.DataFrame
        Training target
    X_test: pd.DataFrame
        Testing dataframe
    y_test : pd.DataFrame
        Testing target
    measure_of_accuracy: function body
        Performance metirc
    Return
    ----------
        True

    """
    obj.fit(X_train, y_train)
    y_preds = obj.predict(X_test)
    pred_labels = np.rint(y_preds)
    print("model output : ")
    print(pred_labels)
    print(f"{measure_of_accuracy} score for classification :")
    print(calc_metric.get_simple_metric(measure_of_accuracy, y_test, pred_labels))
    return True


# functions for regressions
def run_regressors(obj, X_train, y_train, X_test, y_test, measure_of_accuracy):
    """
    A function to get best estimator fit it again, calculate predictions
    and calculate mean_absolute_error.

    Parameters
    ----------
    obj: Object
        Best estimator for regression
    X_train: pd.DataFrame
        Training dataframe
    y_train : pd.DataFrame
        Training target
    X_test: pd.DataFrame
        Testing dataframe
    y_test : pd.DataFrame
        Testing target
    measure_of_accuracy: function body
        Performance metirc
    Return
    ----------
        True

    """
    obj.fit(X_train, y_train)
    y_preds = obj.predict(X_test)
    print("model output : ")
    print(y_preds)
    print(f"{measure_of_accuracy} score for regression :")
    print(calc_metric.get_simple_metric(measure_of_accuracy, y_test, y_preds))


def run_classifiers_optuna(obj, X_train, y_train, X_test, y_test):
    """
    A function to get best estimator fit it again, calculate predictions
    and calculate f1 score.

    Parameters
    ----------
    obj: Object
        Best estimator for classification
    X_train: pd.DataFrame
        Training dataframe
    y_train : pd.DataFrame
        Training target
    X_test: pd.DataFrame
        Testing dataframe
    y_test : pd.DataFrame
        Testing target
    Return
    ----------
        True

    """
    obj.fit(X_train, y_train)
    y_preds = obj.predict(X_test)
    pred_labels = np.rint(y_preds)
    print("model output : ")
    print(pred_labels)
    print("f1_score score for classification (optuna):")
    print(f1_score(y_test, pred_labels))

    return True


# functions for regressions
def run_regressors_optuna(obj, X_train, y_train, X_test, y_test):
    """
    A function to get best estimator fit it again, calculate predictions
    and calculate mean_absolute_error.

    Parameters
    ----------
    obj: Object
        Best estimator for regression
    X_train: pd.DataFrame
        Training dataframe
    y_train : pd.DataFrame
        Training target
    X_test: pd.DataFrame
        Testing dataframe
    y_test : pd.DataFrame
        Testing target
    Return
    ----------
        True

    """
    obj.fit(X_train, y_train)
    y_preds = obj.predict(X_test)
    print("model output : ")
    print(y_preds)
    print("mean_absolute_error regression score for regression  (optuna) :")
    print(mean_absolute_error(y_test, y_preds))


# A dictonary of many classification predictive models and
# some of their parameters in some ranges.
models_classifiers = {
    "XGBClassifier": {
        "eval_metric": ["auc"],
        "max_depth": [4, 5],
    },
    "LGBMClassifier": {"max_depth": [1, 12]},
    "CatBoostClassifier": {
        "depth": [5, 6],
        "boosting_type": ["Ordered"],
        "bootstrap_type": ["Bayesian"],
        "logging_level": ["Silent"],
    },
    "SVC": {
        "C": [0.5, 1],
        "kernel": ["poly"],
    },
    "MLPClassifier": {
        "activation": ["identity"],
        "alpha": [0.0001, 0.001],
    },
    "BalancedRandomForestClassifier": {
        "n_estimators": [100, 200],
        "min_impurity_decrease": [0.0, 0.1],
    },
}

# A dictonary of many regression predictive models and
# some of their parameters in some ranges.
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

# check grid search on selected classification models
def run_gird_classifiers(pause_iteration=True):
    """
    Loop trough some of the classifiers that already
    created and to test if grid search works on them
    or not.
    Parameters
    ----------
    pause_iteration: boolean
        To pause the running of the function after each iteration.
    Return
    ----------
        None

    """
    for model in models_classifiers:
        measure_of_accuracy = "f1_plus_tp"
        obj = BaseModel.bestmodel_factory.using_gridsearch(
            estimator=eval(model + "()"),
            estimator_params=models_classifiers[model],
            measure_of_accuracy=measure_of_accuracy,
            verbose=3,
            n_jobs=-1,
            random_state=42,
            cv=KFold(2),
        )
        # run classifiers
        run_classifiers(obj, X_train, y_train, X_test, y_test, measure_of_accuracy)
        if pause_iteration:
            val = input(f"Enter confirmation of results for the model {model} Y/N: ")
            if val == "N":
                break


# check grid search on selected regression models
def run_gird_regressoros(pause_iteration=True):
    """
    Loop trough some of the regressors that already
    created and to test if grid search works on them
    or not.
    Parameters
    ----------
    pause_iteration: boolean
        To pause the running of the function after each iteration.
    Return
    ----------
        None

    """
    for model in models_regressors:
        measure_of_accuracy = "mean_absolute_error"
        obj = BaseModel.bestmodel_factory.using_gridsearch(
            estimator=eval(model + "()"),
            estimator_params=models_regressors[model],
            measure_of_accuracy=measure_of_accuracy,
            verbose=3,
            n_jobs=-1,
            random_state=42,
            cv=KFold(2),
        )
        # run classifiers
        run_regressors(obj, X_train, y_train, X_test, y_test, measure_of_accuracy)
        if pause_iteration:
            val = input(f"Enter confirmation of results for the model {model} Y/N: ")
            if val == "N":
                break


# check randomized search on selected classification models
def run_random_classifiers(pause_iteration=True):
    """
    Loop trough some of the classifiers that already
    created and to test if random search works on them
    or not.
    Parameters
    ----------
    pause_iteration: boolean
        To pause the running of the function after each iteration.
    Return
    ----------
        None

    """
    for model in models_classifiers:
        measure_of_accuracy = "f1_score"
        obj = BaseModel.bestmodel_factory.using_randomsearch(
            estimator=eval(model + "()"),
            estimator_params=models_classifiers[model],
            measure_of_accuracy=measure_of_accuracy,
            verbose=3,
            n_jobs=-1,
            random_state=42,
            cv=KFold(2),
            n_iter=50,
        )
        # run classifiers
        run_classifiers(obj, X_train, y_train, X_test, y_test, measure_of_accuracy)
        if pause_iteration:
            val = input(f"Enter confirmation of results for the model {model} Y/N: ")
            if val == "N":
                break


# check randomized search on selected regression models
def run_random_regressoros(pause_iteration=True):
    """
    Loop trough some of the regressors that already
    created and to test if random search works on them
    or not.
    Parameters
    ----------
    pause_iteration: boolean
        To pause the running of the function after each iteration.
    Return
    ----------
        None

    """
    for model in models_regressors:
        measure_of_accuracy = "mean_absolute_error"
        obj = BaseModel.bestmodel_factory.using_randomsearch(
            estimator=eval(model + "()"),
            estimator_params=models_regressors[model],
            measure_of_accuracy=measure_of_accuracy,
            verbose=3,
            n_jobs=-1,
            random_state=42,
            cv=KFold(2),
            n_iter=50,
        )
        # run classifiers
        run_regressors(obj, X_train, y_train, X_test, y_test, measure_of_accuracy)
        if pause_iteration:
            val = input(f"Enter confirmation of results for the model {model} Y/N: ")
            if val == "N":
                break


# check optuna search on selected classification models
def run_optuna_classifiers(pause_iteration=True):
    """
    Loop trough some of the classifiers that already
    created and to test if optuna works on them
    or not.
    Parameters
    ----------
    pause_iteration: boolean
        To pause the running of the function after each iteration.
    Return
    ----------
        None

    """
    for model in models_classifiers:
        obj = BaseModel.bestmodel_factory.using_optuna(
            estimator=eval(model + "()"),
            estimator_params=models_classifiers[model],
            measure_of_accuracy="f1_score",
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
        run_classifiers_optuna(obj, X_train, y_train, X_test, y_test)
        if pause_iteration:
            val = input(f"Enter confirmation of results for the model {model} Y/N: ")
            if val == "N":
                break


# check optuna search on selected regression models
def run_optuna_regressors(pause_iteration=True):
    """
    Loop trough some of the regressors that already
    created and to test if optuna works on them
    or not.
    Parameters
    ----------
    pause_iteration: boolean
        To pause the running of the function after each iteration.
    Return
    ----------
        None

    """
    for model in models_regressors:
        obj = BaseModel.bestmodel_factory.using_optuna(
            estimator=eval(model + "()"),
            estimator_params=models_regressors[model],
            measure_of_accuracy="mean_absolute_error",
            verbose=3,
            n_jobs=1,
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
        # run classifiers
        run_regressors_optuna(obj, X_train, y_train, X_test, y_test)
        if pause_iteration:
            val = input(f"Enter confirmation of results for the model {model} Y/N: ")
            if val == "N":
                break


def run_all(pause_iteration):
    """Run all tests cases

    Parameters
    ----------
    pause_iteration: boolean
        To pause the running of the function after each iteration.
    Return
    ----------
        None
    """
    run_gird_classifiers(pause_iteration)  # OK
    run_gird_regressoros(pause_iteration)  # OK
    run_random_classifiers(pause_iteration)  # OK
    run_random_regressoros(pause_iteration)  # OK
    run_optuna_classifiers(pause_iteration)  # OK
    run_optuna_regressors(pause_iteration)  # OK


if __name__ == "__main__":
    """run all tests in once"""
    run_all(False)  # OK
