import pandas as pd
import optuna
from lohrasb.project_conf import ROOT_PROJECT
from optuna.pruners import HyperbandPruner
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.model_selection import KFold, train_test_split
from lohrasb.best_estimator import BaseModel
import os
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error,make_scorer
from sklearn.linear_model import *
from sklearn.svm import *
from xgboost import *
from sklearn.linear_model import *
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

X = data.loc[:, data.columns != "default payment next month"]
y = data.loc[:, data.columns == "default payment next month"]
y = y.values.ravel()

X = X.select_dtypes(include=np.number)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)

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
    return measure_of_accuracy(y_test,y_preds)


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
    return measure_of_accuracy(y_test,y_preds)


# A dictonary of many classification predictive models and
# some of their parameters in some ranges.

models_classifiers = {
    "XGBClassifier": {
        "eval_metric": ["auc"],
        "max_depth": [4, 5],
    },
    "LGBMClassifier": {"max_depth": [1, 12]},
    # TODO catboost can not be installed with potry
    # "CatBoostClassifier": {
    #     "depth": [5, 6],
    #     "boosting_type": ["Ordered"],
    #     "bootstrap_type": ["Bayesian"],
    #     "logging_level": ["Silent"],
    # },
    # "SVC": {
    #     "C": [0.5],
    #     "kernel": ["poly"],
    # },
    "MLPClassifier": {
        "activation": ["identity"],
        "alpha": [ 0.001],
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
        "min_child_weight": [0.1],
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
        "solver": ["adam"],
        "alpha": [0.0001],
    },
}


def test_best_estimator():
    """Test feature scally selector add"""
    # functions for classifiers
    def run_gird_classifiers(pause_iteration=False):
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
            measure_of_accuracy=make_scorer(f1_score, greater_is_better=True)
            obj = BaseModel().optimize_by_gridsearchcv(
                estimator=eval(model + "()"),
                estimator_params=models_classifiers[model],
                fit_params=None,
                measure_of_accuracy=measure_of_accuracy,
                verbose=3,
                n_jobs=-1,
                random_state=42,
                cv=KFold(2),
            )
            # run classifiers
            f1 = run_classifiers(obj, X_train, y_train, X_test, y_test,f1_score)
            assert f1>= 0.0
    def run_random_classifiers(pause_iteration=False):
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
            measure_of_accuracy=make_scorer(f1_score, greater_is_better=True)
            obj = BaseModel().optimize_by_randomsearchcv(
                estimator=eval(model + "()"),
                estimator_params=models_classifiers[model],
                fit_params=None,
                measure_of_accuracy=measure_of_accuracy,
                verbose=3,
                n_jobs=-1,
                random_state=42,
                cv=KFold(2),
                n_iter=1,
            )
            # run classifiers
            f1 = run_classifiers(obj, X_train, y_train, X_test, y_test,f1_score)
            assert f1>= 0.0
    def run_optuna_classifiers(pause_iteration=False):
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
            obj = BaseModel().optimize_by_optuna(
                estimator=eval(model + "()"),
                estimator_params=models_classifiers[model],
                fit_params=None,
                measure_of_accuracy="f1_score(y_true, y_pred, average='micro')",
                verbose=3,
                n_jobs=-1,
                random_state=42,
                test_size = 0.33,
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
            f1 = run_classifiers(obj, X_train, y_train, X_test, y_test,f1_score)
            assert f1>= 0.0
    
    # functions for regressors
    def run_gird_regressors(pause_iteration=False):
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
            measure_of_accuracy=make_scorer(mean_absolute_error, greater_is_better=False)
            obj = BaseModel().optimize_by_gridsearchcv(
                estimator=eval(model + "()"),
                fit_params=None,
                estimator_params=models_regressors[model],
                measure_of_accuracy=measure_of_accuracy,
                verbose=3,
                n_jobs=-1,
                random_state=42,
                cv=KFold(2),
            )
            # run regressors
            mae = run_regressors(obj, X_train, y_train, X_test, y_test, mean_absolute_error)
            assert mae >= 0.0
    
    def run_random_regressors(pause_iteration=False):
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
            measure_of_accuracy=make_scorer(mean_absolute_error, greater_is_better=False)
            obj = BaseModel().optimize_by_randomsearchcv(
                estimator=eval(model + "()"),
                estimator_params=models_regressors[model],
                fit_params=None,
                measure_of_accuracy=measure_of_accuracy,
                verbose=3,
                n_jobs=-1,
                random_state=42,
                cv=KFold(2),
                n_iter=2,
            )
            # run regressors
            mae = run_regressors(obj, X_train, y_train, X_test, y_test, mean_absolute_error)
            assert mae >= 0.0

    def run_optuna_regressors(pause_iteration=False):
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
            obj = BaseModel().optimize_by_optuna(
            estimator=eval(model + "()"),
            estimator_params=models_regressors[model],
            fit_params=None,
            measure_of_accuracy="mean_absolute_error(y_true, y_pred, multioutput='raw_values')",
            verbose=3,
            n_jobs=-1,
            random_state=42,
            test_size = 0.33,
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
            mae = run_regressors(obj, X_train, y_train, X_test, y_test, mean_absolute_error)
            assert mae >= 0.0

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


