# Import necessary libraries
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold, train_test_split

from lohrasb.best_estimator import BaseModel

# Define hyperparameters for the AdaBoostClassifier and Ridge regressor
adb_params = {
    "n_estimators": [50, 200],
    "learning_rate": [0.01, 1.0],
    "algorithm": ["SAMME", "SAMME.R"],
}
ridge_params_reg = {
    "fit_intercept": [True, False],
    "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
}


# Function for training and evaluating a classification model
def using_tune_classification(estimator, params):
    # Create a synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=3,
        n_redundant=10,
        n_classes=3,
        random_state=42,
    )
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the estimator
    est = estimator()

    # Use Optuna for hyperparameter optimization
    obj = BaseModel().optimize_by_optuna(
        kwargs={
            "fit_optuna_kwargs": {},
            "main_optuna_kwargs": {
                "estimator": est,
                "estimator_params": params,
                "refit": True,
                "measure_of_accuracy": 'f1_score(y_true, y_pred,average="weighted")',
            },
            "train_test_split_kwargs": {
                "test_size": 0.3,
            },
            "study_search_kwargs": {
                "storage": None,
                "sampler": TPESampler(),
                "pruner": HyperbandPruner(),
                "study_name": "example of optuna optimizer",
                "direction": "maximize",
                "load_if_exists": False,
            },
            "optimize_kwargs": {
                "n_trials": 20,
                "timeout": 600,
                "catch": (),
                "callbacks": None,
                "gc_after_trial": False,
                "show_progress_bar": False,
            },
        }
    )

    # Fit the model and make predictions
    obj.fit(X_train, y_train)
    y_pred = obj.predict(X_test)

    # Evaluate and print the model performance
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"f1_score is {f1}")


# Function for training and evaluating a regression model
def using_tune_regression(estimator, params):
    # Create a synthetic regression dataset
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1
    )

    # Initialize the estimator
    est = estimator()

    # Use Optuna for hyperparameter optimization
    obj = BaseModel().optimize_by_optuna(
        kwargs={
            "fit_optuna_kwargs": {},
            "main_optuna_kwargs": {
                "estimator": est,
                "estimator_params": params,
                "refit": True,
                "measure_of_accuracy": "mean_absolute_error(y_true, y_pred, multioutput='uniform_average')",
            },
            "train_test_split_kwargs": {
                "test_size": 0.3,
            },
            "study_search_kwargs": {
                "storage": None,
                "sampler": TPESampler(),
                "pruner": HyperbandPruner(),
                "study_name": "example of optuna optimizer",
                "direction": "maximize",
                "load_if_exists": False,
            },
            "optimize_kwargs": {
                "n_trials": 20,
                "timeout": 600,
                "catch": (),
                "callbacks": None,
                "gc_after_trial": False,
                "show_progress_bar": False,
            },
        }
    )

    # Fit the model and make predictions
    obj.fit(X, y)
    predictions = obj.predict(X)

    # Evaluate and print the model performance
    r2 = r2_score(y, predictions)
    print(f"r2_score is {r2}")


# Run regression examples
using_tune_regression(Ridge, ridge_params_reg)

# Run classification examples
using_tune_classification(AdaBoostClassifier, adb_params)
