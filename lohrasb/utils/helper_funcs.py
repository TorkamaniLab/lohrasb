
import pip
import subprocess
import sys

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    make_scorer,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC

maping_mesurements = {
    "accuracy_score": accuracy_score,
    "explained_variance_score": explained_variance_score,
    "f1": f1_score,
    "f1_score": f1_score,
    "mean_absolute_error": mean_absolute_error,
    "mae": mean_absolute_error,
    "mean_absolute_percentage_error": mean_absolute_percentage_error,
    "mape": mean_absolute_percentage_error,
    "mean_squared_error": mean_squared_error,
    "mse": mean_squared_error,
    "median_absolute_error": median_absolute_error,
    "precision_score": precision_score,
    "precision": precision_score,
    "r2": r2_score,
    "r2_score": r2_score,
    "recall_score": recall_score,
    "recall": recall_score,
    "roc_auc_score": roc_auc_score,
    "roc": roc_auc_score,
    "roc_auc": roc_auc_score,
}


def _trail_param_retrive(trial, dict, keyword):
    """An internal function. Return a trial suggest using dict params of estimator and
    one keyword of it. Based on the keyword, it will return an
    Optuna.trial.suggest. The return will be trial.suggest_int(keyword, min(dict[keyword]), max(dict[keyword]))

    Example : _trail_param_retrive(trial, {
            "max_depth": [2, 3],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
             }, "gamma") --> will be trail.suggest_int for gamma using [1,9]

    Parameters
    ----------
    trial: Optuna trial
        A trial is a process of evaluating an objective function.
        For more info, visit
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
    dict: dict
        A dictionary of estimator params.
        e.g., {
            "max_depth": [2, 3],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
             }
    Keyword: str
        A keyword of estimator key params. e.g., "gamma"
    """
    if isinstance(dict[keyword][0] , str)  or dict[keyword][0] is None:
        return trial.suggest_categorical(keyword, dict[keyword])
    if isinstance(dict[keyword][0] , int):
        if len(dict[keyword]) >=2:
            if isinstance(dict[keyword][1] , int):
                return trial.suggest_int(keyword, min(dict[keyword]), max(dict[keyword]))
        else :
            return trial.suggest_float(keyword, min(dict[keyword]), max(dict[keyword]))
    if isinstance(dict[keyword][0] , float):
        return trial.suggest_float(keyword, min(dict[keyword]), max(dict[keyword]))


def _trail_params_retrive(trial, dict):
    """An internal function. Return a trial suggests using dict params of estimator.
    
    Example : _trail_param_retrive(trial, {
            "eval_metric": ["auc"],
            "max_depth": [2, 3],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
            "booster": ["gbtree", "gblinear", "dart"],
             }, "gamma") --> will return params where 
             
             parmas = {
                "eval_metric": trial.suggest_categorical("eval_metric", ["auc"]),
                "max_depth": trial.suggest_int("max_depth", 2,3),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 0.9),
                "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
             }
    Parameters
    ----------
    trial: Optuna trial
        A trial is a process of evaluating an objective function.
        For more info, visit
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
    dict: dict
        A dictionary of estimator params.
        e.g., {
            "eval_metric": ["auc"],
            "max_depth": [2, 3],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
            "booster": ["gbtree", "gblinear", "dart"],
             }
    """
    params = {}
    for keyword in dict.keys():
        if keyword not in params.keys():
            if isinstance(dict[keyword][0] , str)  or dict[keyword][0] is None:
                params[keyword] = trial.suggest_categorical(keyword, dict[keyword])
            if isinstance(dict[keyword][0] , int):
                if len(dict[keyword]) >=2:
                    if isinstance(dict[keyword][1] , int):
                        params[keyword] = trial.suggest_int(keyword, min(dict[keyword]), max(dict[keyword]))
                else :
                    params[keyword] = trial.suggest_float(keyword, min(dict[keyword]), max(dict[keyword]))
            if isinstance(dict[keyword][0] , float):
                params[keyword] = trial.suggest_float(keyword, min(dict[keyword]), max(dict[keyword]))
    return params

def calc_metric_for_multi_outputs_classification(
    multi_label, valid_y, preds, SCORE_TYPE
):
    """Internal function for calculating the performance of a multi-output
    classification estimator.

    Parameters
    ----------
    multi_label : Pandas DataFrame
        A multioutput Class label. This is a Pandas multioutput label data frame.
    valid_y : Pandas DataFrame or Pandas Series
        True labels
    preds : Pandas DataFrame Pandas Series
        predicted labels.
    SCORE_TYPE : str
        A string refers to the type of error measurement function.
        Supported values "f1_score", "accuracy_score", "precision_score",
        "recall_score", "roc_auc_score","tp","tn"
    """
    sum_errors = 0

    for i, l in enumerate(multi_label):
        f1 = f1_score(valid_y[l], preds[:, i])
        acc = accuracy_score(valid_y[l], preds[:, i])
        pr = precision_score(valid_y[l], preds[:, i])
        recall = recall_score(valid_y[l], preds[:, i])
        roc = roc_auc_score(valid_y[l], preds[:, i])
        tn, fp, fn, tp = confusion_matrix(
            valid_y[l], preds[:, i], labels=[0, 1]
        ).ravel()

        if SCORE_TYPE == "f1" or SCORE_TYPE == "f1_score":
            sum_errors = sum_errors + f1
        if (
            SCORE_TYPE == "acc"
            or SCORE_TYPE == "accuracy_score"
            or SCORE_TYPE == "accuracy"
        ):
            sum_errors = sum_errors + acc
        if (
            SCORE_TYPE == "pr"
            or SCORE_TYPE == "precision_score"
            or SCORE_TYPE == "precision"
        ):
            sum_errors = sum_errors + pr
        if (
            SCORE_TYPE == "recall"
            or SCORE_TYPE == "recall_score"
            or SCORE_TYPE == "recall"
        ):
            sum_errors = sum_errors + recall
        if (
            SCORE_TYPE == "roc"
            or SCORE_TYPE == "roc_auc_score"
            or SCORE_TYPE == "roc_auc"
        ):
            sum_errors = sum_errors + roc

        # other metrics - not often use

        if SCORE_TYPE == "tp" or SCORE_TYPE == "true possitive":
            sum_errors = sum_errors + tp
        if SCORE_TYPE == "tn" or SCORE_TYPE == "true negative":
            sum_errors = sum_errors + tn

    return sum_errors


def _calc_metric_for_single_output_classification(valid_y, pred_labels, SCORE_TYPE):
    """Internal function for calculating the performance of a
    classification estimator.

    Parameters
    ----------
    valid_y : Pandas DataFrame or Pandas Series
        True labels
    preds : Pandas DataFrame Pandas Series
        predicted labels.
    SCORE_TYPE : str
        A string refers to the type of error measurement function.
        Supported values "f1_score", "accuracy_score", "precision_score",
        "recall_score", "roc_auc_score","tp","tn"

    """

    sum_errors = 0
    f1 = f1_score(valid_y, pred_labels)
    acc = accuracy_score(valid_y, pred_labels)
    pr = precision_score(valid_y, pred_labels)
    recall = recall_score(valid_y, pred_labels)
    roc = roc_auc_score(valid_y, pred_labels)

    tn, _, _, tp = confusion_matrix(valid_y, pred_labels, labels=[0, 1]).ravel()
    if SCORE_TYPE == "f1" or SCORE_TYPE == "f1_score":
        sum_errors = sum_errors + f1
    if (
        SCORE_TYPE == "acc"
        or SCORE_TYPE == "accuracy_score"
        or SCORE_TYPE == "accuracy"
    ):
        sum_errors = sum_errors + acc
    if (
        SCORE_TYPE == "pr"
        or SCORE_TYPE == "precision_score"
        or SCORE_TYPE == "precision"
    ):
        sum_errors = sum_errors + pr
    if SCORE_TYPE == "recall" or SCORE_TYPE == "recall_score" or SCORE_TYPE == "recall":
        sum_errors = sum_errors + recall
    if SCORE_TYPE == "roc" or SCORE_TYPE == "roc_auc_score" or SCORE_TYPE == "roc_auc":
        sum_errors = sum_errors + roc

    # other metrics - not often use

    if SCORE_TYPE == "tp" or SCORE_TYPE == "true possitive":
        sum_errors = sum_errors + tp
    if SCORE_TYPE == "tn" or SCORE_TYPE == "true negative":
        sum_errors = sum_errors + tn

    return sum_errors


def _calc_metric_for_single_output_regression(valid_y, pred_labels, SCORE_TYPE):
    """Internal function for calculating the performance of a
    regression estimator.

    Parameters
    ----------
    valid_y : Pandas DataFrame or Pandas Series
        True values
    preds : Pandas DataFrame Pandas Series
        predicted values.
    SCORE_TYPE : str
        A string refers to the type of error measurement function.
        Supported values "r2_score", "explained_variance_score", "max_error",
        "mean_absolute_error", "mean_squared_error","median_absolute_error",
        "mean_absolute_percentage_error"

    """

    r2 = r2_score(valid_y, pred_labels)
    explained_variance_score_sr = explained_variance_score(valid_y, pred_labels)

    max_error_err = max_error(valid_y, pred_labels)
    mean_absolute_error_err = mean_absolute_error(valid_y, pred_labels)
    mean_squared_error_err = mean_squared_error(valid_y, pred_labels)
    median_absolute_error_err = median_absolute_error(valid_y, pred_labels)
    mean_absolute_percentage_error_err = mean_absolute_percentage_error(
        valid_y, pred_labels
    )

    if SCORE_TYPE == "r2" or SCORE_TYPE == "r2_score":
        return r2
    if SCORE_TYPE == "explained_variance_score":
        return explained_variance_score_sr

    if SCORE_TYPE == "max_error":
        return max_error_err
    if SCORE_TYPE == "mean_absolute_error":
        return mean_absolute_error_err
    if SCORE_TYPE == "mean_squared_error":
        return mean_squared_error_err
    if SCORE_TYPE == "median_absolute_error":
        return median_absolute_error_err
    if SCORE_TYPE == "mean_absolute_percentage_error":
        return mean_absolute_percentage_error_err


def install_and_import(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)
