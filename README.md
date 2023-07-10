![GitHub Repo stars](https://img.shields.io/github/stars/drhosseinjavedani/lohrasb) ![GitHub forks](https://img.shields.io/github/forks/drhosseinjavedani/lohrasb) ![GitHub language count](https://img.shields.io/github/languages/count/drhosseinjavedani/lohrasb) ![GitHub repo size](https://img.shields.io/github/repo-size/drhosseinjavedani/lohrasb) ![GitHub](https://img.shields.io/github/license/drhosseinjavedani/lohrasb)![PyPI - Downloads](https://img.shields.io/pypi/dd/lohrasb) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lohrasb) 

# Lohrasb

Lohrasb is a  tool built to ease machine learning development by tuning hyper-parameters of estimators in a scalable way. It uses [Optuna](https://optuna.readthedocs.io/en/stable/index.html), [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), and [Ray tune Scikit-Learn API](https://docs.ray.io/en/latest/tune/api_docs/sklearn.html) to tune most of the estimators of sickit-learn. It is compatible with [scikit-learn](https://scikit-learn.org) pipeline, [XGBoost Survival Embeddings](https://github.com/loft-br/xgboost-survival-embeddings) and, [InterpretML](https://github.com/interpretml/interpret/).


### Introduction

BaseModel of the Lohrasb package can receive various parameters. From an estimator class to its tunning parameters and GridsearchCV, RandomizedSearchCV, or Optuna to their parameters. Samples will be split to train and validation set, and then optimization will estimate optimal related parameters using these optimizing engines.

### Installation

Lohrasb package is available on PyPI and can be installed with pip:

```sh
pip install lohrasb
```


### Supported estimators for this package
Lohrasb supports almost all machine learning estimators for classification and regression.

### Usage

- Tunning best parameters of a machine learning model using [Optuna](https://optuna.readthedocs.io/en/stable/index.html) , [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), [TuneGridSearchCV, and TuneSearchCV](https://docs.ray.io/en/latest/tune/api_docs/sklearn.html) from [Ray](https://docs.ray.io/en/latest/index.html) tune Scikit-Learn API (tune.sklearn).

### Some examples

In this section we will provide some examples for users
#### optimize_by_tune
This method use Tune from Ray for hyperparameter optimization. It has all capability of tune with simpler interface for machine learning problems.
```
from ray.tune.search.hyperopt import HyperOptSearch
import optuna

# Datasets
from sklearn.datasets import make_classification, make_regression

# Custom estimator
from lohrasb.best_estimator import BaseModel

# Gradient boosting frameworks
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# metrics
from sklearn.metrics import r2_score, f1_score

# Imbalanced-learn ensemble
from imblearn.ensemble import BalancedRandomForestClassifier
from ray import air, tune

# others
from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    Ridge,
    SGDRegressor,
)

# Define hyperparameters for each model
xgb_params = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.randint(6, 15),
    "learning_rate": tune.uniform(0.001, 0.1),
}
cb_params = {
    "iterations": tune.randint(50, 200),
    "depth": tune.randint(4, 8),
    "learning_rate": tune.uniform(0.001, 0.1),
}
brf_params = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.choice([None, 10, 20]),
}

# Put models and hyperparameters into a list of tuples
estimators_params_clfs = [
    (XGBClassifier, xgb_params),
    (CatBoostClassifier, cb_params),
    (BalancedRandomForestClassifier, brf_params),
]

# Define hyperparameters for each model
ridge_params_reg = {"alpha": tune.uniform(0.5, 1.0)}
sgr_params_reg = {
    "loss": tune.choice(["squared_loss", "huber", "epsilon_insensitive"]),
    "penalty": tune.choice(["l2", "l1", "elasticnet"]),
}

# Put models and hyperparameters into a list of tuples
estimators_params_regs = [(Ridge, ridge_params_reg), (SGDRegressor, sgr_params_reg)]


def using_tune_classification(estimator, params):
    # Create synthetic dataset
    search_alg = HyperOptSearch()
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=3,
        n_redundant=10,
        n_classes=3,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the estimator
    est = estimator()

    # Create keyword arguments for tune
    kwargs = {
        # define kwargs for base model
        "kwargs": {  # params for fit method
            "fit_tune_kwargs": {
                "sample_weight": None,
            },
            # params for TuneCV
            "main_tune_kwargs": {
                "cv": 3,
                "scoring": "f1_macro",
                "estimator": est,
            },
            # kwargs of Tuner
            "tuner_kwargs": {
                "tune_config": tune.TuneConfig(
                    search_alg=search_alg,
                    mode="max",
                    metric="score",
                ),
                "param_space": params,
                "run_config": air.RunConfig(stop={"training_iteration": 20}),
            },
        }
    }

    # Run optimize_by_tune
    obj = BaseModel().optimize_by_tune(**kwargs)
    obj.fit(X_train, y_train)

    # Predict on test data
    y_pred = obj.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"f1_score is {f1}")


def using_tune_regressiom(estimator, params):
    search_alg = HyperOptSearch()
    # Create synthetic regression dataset
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1
    )

    # Initialize the estimator
    est = estimator()

    # Create keyword arguments for tune
    kwargs = {
        # define kwargs for base model
        "kwargs": {  # params for fit method
            "fit_tune_kwargs": {
                "sample_weight": None,
            },
            # params for TuneCV
            "main_tune_kwargs": {
                "cv": 3,
                "scoring": "r2",
                "estimator": est,
            },
            # kwargs of Tuner
            "tuner_kwargs": {
                "tune_config": tune.TuneConfig(
                    search_alg=search_alg,
                    mode="max",
                    metric="score",
                ),
                "param_space": params,
                "run_config": air.RunConfig(stop={"training_iteration": 20}),
            },
        }
    }

    # Create obj of the class
    obj = BaseModel().optimize_by_tune(**kwargs)

    # Check if instance created successfully
    assert obj is not None

    # Fit data and predict
    obj.fit(X, y)
    predictions = obj.predict(X)
    r2 = r2_score(y, predictions)
    print(f"r2_score is {r2}")

    (Ridge, ridge_params_reg),
    (SGDRegressor, sgr_params_reg)


# some regression examples
using_tune_regressiom(Ridge, ridge_params_reg)
using_tune_regressiom(SGDRegressor, sgr_params_reg)
# some classification examples
using_tune_classification(CatBoostClassifier, cb_params)
using_tune_classification(XGBClassifier, xgb_params)
using_tune_classification(BalancedRandomForestClassifier, brf_params)

```
#### optimize_by_gridsearchcv 

```
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, r2_score
from lohrasb.best_estimator import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression

# Define hyperparameters for the classifiers and regressors
rf_params = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
lr_params_reg = {"fit_intercept": [True, False]}
lasso_params_reg = {"alpha": [0.1, 0.5, 1.0]}

def using_tune_classification(estimator, params):
    # Create synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=3, n_redundant=10, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the estimator
    est = estimator()

    # Create the model with the chosen hyperparameters
    obj = BaseModel().optimize_by_gridsearchcv(
        kwargs={
            "fit_grid_kwargs": {
                "sample_weight": None,
            },
            "grid_search_kwargs": {
                "estimator": est,
                "param_grid": params,
                "scoring": "f1_micro",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
        }
    )

    # Fit the model and make predictions
    obj.fit(X_train, y_train)
    y_pred = obj.predict(X_test)

    # Evaluate the model performance
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"f1_score is {f1}")

def using_tune_regression(estimator, params):
    # Create synthetic regression dataset
    X, y = make_regression(n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1)

    # Initialize the estimator
    est = estimator()

    # Create the model with the chosen hyperparameters
    obj = BaseModel().optimize_by_gridsearchcv(
        kwargs={
            "fit_grid_kwargs": {
                "sample_weight": None,
            },
            "grid_search_kwargs": {
                "estimator": est,
                "param_grid": params,
                "scoring": "r2",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
        }
    )

    # Fit the model and make predictions
    obj.fit(X, y)
    predictions = obj.predict(X)

    # Evaluate the model performance
    r2 = r2_score(y, predictions)
    print(f"r2_score is {r2}")

# Regression examples
using_tune_regression(Lasso, lasso_params_reg)
using_tune_regression(LinearRegression, lr_params_reg)

# Classification examples
using_tune_classification(RandomForestClassifier, rf_params)

```
#### optimize_by_randomsearchcv 

```
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, r2_score
from lohrasb.best_estimator import BaseModel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Ridge

# Define hyperparameters for the classifiers and regressors
adb_params = {"n_estimators": [50, 100, 200], "learning_rate": [0.001, 0.01, 0.1]}
ridge_params_reg = {"fit_intercept": [True, False]}


def using_tune_classification(estimator, params):
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=3,
        n_redundant=10,
        n_classes=3,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the estimator
    est = estimator()

    # Create the model with the chosen hyperparameters
    obj = BaseModel().optimize_by_randomsearchcv(
        kwargs={
            "fit_random_kwargs": {
                "sample_weight": None,
            },
            "random_search_kwargs": {
                "estimator": est,
                "param_distributions": params,
                "scoring": "f1_micro",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
                "n_iter": 10,
            },
            "main_random_kwargs": {},
        }
    )

    # Fit the model and make predictions
    obj.fit(X_train, y_train)
    y_pred = obj.predict(X_test)

    # Evaluate the model performance
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"f1_score is {f1}")


def using_tune_regression(estimator, params):
    # Create synthetic regression dataset
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1
    )

    # Initialize the estimator
    est = estimator()

    # Create the model with the chosen hyperparameters
    obj = BaseModel().optimize_by_randomsearchcv(
        kwargs={
            "fit_random_kwargs": {
                "sample_weight": None,
            },
            "random_search_kwargs": {
                "estimator": est,
                "param_distributions": params,
                "scoring": "r2",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
                "n_iter": 10,
            },
            "main_random_kwargs": {},
        }
    )

    # Fit the model and make predictions
    obj.fit(X, y)
    predictions = obj.predict(X)

    # Evaluate the model performance
    r2 = r2_score(y, predictions)
    print(f"r2_score is {r2}")


# Regression examples
using_tune_regression(Ridge, ridge_params_reg)

# Classification examples
using_tune_classification(AdaBoostClassifier, adb_params)
```
#### optimize_by_optunasearchcv

```
from sklearn.datasets import make_classification, make_regression
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, r2_score
from lohrasb.best_estimator import BaseModel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Ridge

# Define hyperparameters for the classifiers and regressors
adb_params = {
    "n_estimators": optuna.distributions.IntDistribution(50, 200),
    "learning_rate": optuna.distributions.FloatDistribution(0.001, 0.1),
}
ridge_params_reg = {
    "fit_intercept": optuna.distributions.CategoricalDistribution(choices=[True, False])
}


def using_tune_classification(estimator, params):
    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=3,
        n_redundant=10,
        n_classes=3,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the estimator and create a model with the specified hyperparameters
    est = estimator()
    obj = BaseModel().optimize_by_optunasearchcv(
        kwargs={
            "fit_newoptuna_kwargs": {"sample_weight": None},
            "newoptuna_search_kwargs": {
                "estimator": est,
                "param_distributions": params,
                "scoring": "f1_micro",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
            "main_newoptuna_kwargs": {},
        }
    )

    # Fit the model and make predictions
    obj.fit(X_train, y_train)
    y_pred = obj.predict(X_test)

    # Evaluate and print the model performance
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"f1_score is {f1}")


def using_tune_regression(estimator, params):
    # Create synthetic regression dataset
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1
    )

    # Initialize the estimator and create a model with the specified hyperparameters
    est = estimator()
    obj = BaseModel().optimize_by_optunasearchcv(
        kwargs={
            "fit_newoptuna_kwargs": {"sample_weight": None},
            "newoptuna_search_kwargs": {
                "estimator": est,
                "param_distributions": params,
                "scoring": "r2",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
            "main_newoptuna_kwargs": {},
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
```
#### 

```
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, r2_score
from lohrasb.best_estimator import BaseModel
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier

# Define hyperparameters for the classifiers and regressors
cat_params_reg = {"n_estimators": [50, 100, 200], "learning_rate": [0.001, 0.01, 0.1]}
lgbm_params = {"max_depth": [5, 6, 7, 10], "gamma": [0.01, 0.1, 1, 1.2]}

def using_tune_classification(estimator, params):
    # Create synthetic classification dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=3, n_redundant=10, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the estimator and create a model with the specified hyperparameters
    est = estimator()
    obj = BaseModel().optimize_by_tunegridsearchcv(
        kwargs={
            "fit_tunegrid_kwargs": {"sample_weight": None},
            "tunegrid_search_kwargs": {
                "estimator": est,
                "param_grid": params,
                "scoring": "f1_micro",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
            "main_tunegrid_kwargs": {},
        }
    )

    # Fit the model and make predictions
    obj.fit(X_train, y_train)
    y_pred = obj.predict(X_test)

    # Evaluate and print the model performance
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"f1_score is {f1}")

def using_tune_regression(estimator, params):
    # Create synthetic regression dataset
    X, y = make_regression(n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1)

    # Initialize the estimator and create a model with the specified hyperparameters
    est = estimator()
    obj = BaseModel().optimize_by_tunegridsearchcv(
        kwargs={
            "fit_tunegrid_kwargs": {"sample_weight": None},
            "tunegrid_search_kwargs": {
                "estimator": est,
                "param_grid": params,
                "scoring": "r2",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
            "main_tunegrid_kwargs": {},
        }
    )

    # Fit the model and make predictions
    obj.fit(X, y)
    predictions = obj.predict(X)

    # Evaluate and print the model performance
    r2 = r2_score(y, predictions)
    print(f"r2_score is {r2}")

# Run regression examples
using_tune_regression(CatBoostRegressor, cat_params_reg)

# Run classification examples
using_tune_classification(LGBMClassifier, lgbm_params)
```

####
```
# Import necessary libraries
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, r2_score
from lohrasb.best_estimator import BaseModel
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMClassifier

# Define hyperparameters for the MLPRegressor and LGBMClassifier
# These will be the values that the hyperparameter search function will iterate through.
mlp_params_reg = {
    "hidden_layer_sizes": [(5, 5, 5), (5, 10, 5), (10,)],
    "activation": ["tanh", "relu"],
    "solver": ["sgd", "adam"],
    "alpha": [0.0001, 0.05],
    "learning_rate": ["constant", "adaptive"],
}
lgbm_params = {"max_depth": [5, 6, 7, 10]}

# Function for training and evaluating a classification model
def using_tune_classification(estimator, params):
    # Create a synthetic classification dataset with 1000 samples, 20 features, and 3 classes
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

    # Use the hyperparameter search function provided by the BaseModel class to find the best parameters
    obj = BaseModel().optimize_by_tunesearchcv(
        kwargs={
            "fit_tune_kwargs": {"sample_weight": None},
            "tune_search_kwargs": {
                "estimator": est,
                "param_distributions": params,
                "scoring": "f1_micro",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
            "main_tune_kwargs": {},
        }
    )

    # Fit the model to the training data
    obj.fit(X_train, y_train)
    # Predict the labels for the test data
    y_pred = obj.predict(X_test)

    # Compute the F1 score of the model
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"f1_score is {f1}")


# Function for training and evaluating a regression model
def using_tune_regression(estimator, params):
    # Create a synthetic regression dataset with 1000 samples and 10 features
    X, y = make_regression(
        n_samples=1000, n_features=10, n_informative=5, n_targets=1, random_state=1
    )

    # Initialize the estimator
    est = estimator()

    # Use the hyperparameter search function provided by the BaseModel class to find the best parameters
    obj = BaseModel().optimize_by_tunesearchcv(
        kwargs={
            "fit_tune_kwargs": {},
            "tune_search_kwargs": {
                "estimator": est,
                "param_distributions": params,
                "scoring": "r2",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
            "main_tune_kwargs": {},
        }
    )

    # Fit the model to the data
    obj.fit(X, y)
    # Predict the targets for the data
    predictions = obj.predict(X)

    # Compute the R2 score of the model
    r2 = r2_score(y, predictions)
    print(f"r2_score is {r2}")


# Run the regression function using the MLPRegressor and the specified parameters
using_tune_regression(MLPRegressor, mlp_params_reg)

# Run the classification function using the LGBMClassifier and the specified parameters
using_tune_classification(LGBMClassifier, lgbm_params)
```
####

```
# Import necessary libraries
from sklearn.datasets import make_classification, make_regression
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, r2_score
from lohrasb.best_estimator import BaseModel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Ridge
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# Define hyperparameters for the AdaBoostClassifier and Ridge regressor
adb_params = {
    'n_estimators': [50,  200],
    'learning_rate': [0.01,  1.0],
    'algorithm': ['SAMME', 'SAMME.R'],
}
ridge_params_reg = {
    'fit_intercept': [True, False],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
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
            "fit_optuna_kwargs": {
            },
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
            "fit_optuna_kwargs": {
            },
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
```
There are some more examples  available in the [examples](https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb/examples) webpage. 

#### License
Licensed under the [BSD 2-Clause](https://opensource.org/licenses/BSD-2-Clause) License.