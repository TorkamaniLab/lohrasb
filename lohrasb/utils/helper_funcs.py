from lohrasb import logger


def _trail_param_retrive(trial, params_dict, keyword):
    """
    An internal function that returns a trial suggestion using the dictionary of estimator parameters and a specific keyword.
    Based on the keyword, it will return an Optuna.trial.suggest value.
    The returned value will be trial.suggest_int(keyword, min(params_dict[keyword]), max(params_dict[keyword])).

    Example:
        _trail_param_retrive(trial, {
            "max_depth": [2, 3],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
        }, "gamma") --> returns trial.suggest_int for gamma using [1, 9].

    Parameters:
        trial (Optuna trial): A trial is a process of evaluating an objective function.
                              For more information, visit:
                              https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
        params_dict (dict): A dictionary of estimator parameters.
                            Example: {
                                "max_depth": [2, 3],
                                "min_child_weight": [0.1, 0.9],
                                "gamma": [1, 9],
                            }
        keyword (str): A keyword of the estimator's parameter.
                       Example: "gamma"
    """
    if not isinstance(params_dict, dict):
        raise TypeError("The params_dict argument must be a dictionary.")
    if not isinstance(keyword, str):
        raise TypeError("The keyword argument must be a string.")

    if keyword not in params_dict:
        raise ValueError(f"The keyword '{keyword}' is not found in the params_dict.")

    param_values = params_dict[keyword]
    if not isinstance(param_values, list):
        raise TypeError(f"The value for the keyword '{keyword}' must be a list.")

    if not param_values:
        raise ValueError(f"The list for the keyword '{keyword}' is empty.")

    if isinstance(param_values[0], str) or param_values[0] is None:
        return trial.suggest_categorical(keyword, param_values)
    if isinstance(param_values[0], int):
        if len(param_values) >= 2:
            if isinstance(param_values[1], int):
                return trial.suggest_int(keyword, min(param_values), max(param_values))
        else:
            return trial.suggest_float(keyword, min(param_values), max(param_values))
    if isinstance(param_values[0], float):
        return trial.suggest_float(keyword, min(param_values), max(param_values))


def _trail_params_retrive(trial, params_dict):
    """
    An internal function that returns trial suggestions using the dictionary of estimator parameters.

    Example:
        _trail_param_retrive(trial, {
            "eval_metric": ["auc"],
            "max_depth": [2, 3],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
            "booster": ["gbtree", "gblinear", "dart"],
        }) --> returns params where

        params = {
            "eval_metric": trial.suggest_categorical("eval_metric", ["auc"]),
            "max_depth": trial.suggest_int("max_depth", 2, 3),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 0.9),
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        }

    Parameters:
        trial (Optuna trial): A trial is a process of evaluating an objective function.
                              For more information, visit:
                              https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
        params_dict (dict): A dictionary of estimator parameters.
                            Example: {
                                "eval_metric": ["auc"],
                                "max_depth": [2, 3],
                                "min_child_weight": [0.1, 0.9],
                                "gamma": [1, 9],
                                "booster": ["gbtree", "gblinear", "dart"],
                            }
    """
    if not isinstance(params_dict, dict):
        raise TypeError("The params_dict argument must be a dictionary.")

    params = {}
    for keyword in params_dict.keys():
        if not isinstance(keyword, str):
            raise TypeError("The keyword in params_dict must be a string.")

        if keyword not in params_dict:
            raise ValueError(f"The keyword '{keyword}' is not found in params_dict.")

        param_values = params_dict[keyword]
        if not isinstance(param_values, list):
            raise TypeError(f"The value for the keyword '{keyword}' must be a list.")

        if not param_values:
            raise ValueError(f"The list for the keyword '{keyword}' is empty.")

        if isinstance(param_values[0], str) or param_values[0] is None:
            params[keyword] = trial.suggest_categorical(keyword, param_values)
        elif isinstance(param_values[0], int):
            if len(param_values) >= 2:
                if isinstance(param_values[1], int):
                    params[keyword] = trial.suggest_int(
                        keyword, min(param_values), max(param_values)
                    )
            elif len(param_values) == 1:
                if isinstance(param_values[0], int):
                    params[keyword] = trial.suggest_int(
                        keyword, min(param_values), max(param_values)
                    )
            else:
                params[keyword] = trial.suggest_float(
                    keyword, min(param_values), max(param_values)
                )
        elif isinstance(param_values[0], float):
            params[keyword] = trial.suggest_float(
                keyword, min(param_values), max(param_values)
            )
        else:
            raise TypeError(
                f"The values in the list for the keyword '{keyword}' must be either strings, integers, or floats."
            )

    return params
