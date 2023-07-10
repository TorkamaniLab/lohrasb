![GitHub Repo stars](https://img.shields.io/github/stars/drhosseinjavedani/lohrasb) 
![GitHub forks](https://img.shields.io/github/forks/drhosseinjavedani/lohrasb) 
![GitHub language count](https://img.shields.io/github/languages/count/drhosseinjavedani/lohrasb) 
![GitHub repo size](https://img.shields.io/github/repo-size/drhosseinjavedani/lohrasb) 
![GitHub](https://img.shields.io/github/license/drhosseinjavedani/lohrasb)
![PyPI - Downloads](https://img.shields.io/pypi/dd/lohrasb) 
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lohrasb)
![GitHub issues](https://img.shields.io/github/issues/drhosseinjavedani/lohrasb)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Codacy Badge](https://img.shields.io/codacy/grade/e27821fb6289410b8f58338c7e0bc686)
![GitHub contributors](https://img.shields.io/github/contributors/drhosseinjavedani/lohrasb)
![GitHub last commit](https://img.shields.io/github/last-commit/drhosseinjavedani/lohrasb)

# Lohrasb
Introducing **Lohrasb**, a powerful tool designed to streamline machine learning development by providing scalable hyperparameter tuning solutions. Lohrasb incorporates several robust optimization frameworks including [Optuna](https://optuna.readthedocs.io/en/stable/index.html), [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), and [Ray Tune Scikit-Learn API](https://docs.ray.io/en/latest/tune/api_docs/sklearn.html). Its compatibility extends to the majority of estimators from Scikit-learn as well as popular machine learning libraries such as [CatBoost](https://catboost.ai/) and [LightGBM](https://lightgbm.readthedocs.io/en/latest/), offering a seamless hyperparameter tuning experience.

Lohrasb is also flexible enough to cater to models conforming to standard Scikit-learn API conventions, such as those implementing `fit` and `predict` methods. This means if you're working with a custom model that adheres to these conventions, or any machine learning model from other libraries that use these methods, Lohrasb can assist you in optimizing the model's hyperparameters.

In addition to model flexibility, Lohrasb provides flexibility in optimization metrics as well. It naturally supports standard Scikit-learn metrics like `f1_score` or `r2_score`. Beyond these, it allows the use of custom evaluation metrics for optimization purposes. This could include specially designed metrics like `f1_plus_tn` or any other specific, customized metric that aligns with your project's requirements.

Overall, whether you're tuning a Scikit-learn estimator, a CatBoost model, a LightGBM classifier, or even a custom model, Lohrasb is designed to streamline your workflow and make the process of hyperparameter optimization more efficient and effective. Its broad compatibility ensures that you can achieve the best performance possible from your models, guided by optimization metrics that are most aligned with your project's goals.

### Introduction
The BaseModel of the Lohrasb package is designed with versatility and flexibility in mind. It accepts a variety of parameters ranging from an estimator class and its tuning parameters to different optimization engines like GridSearchCV, RandomizedSearchCV, or Optuna, and their associated parameters. In this process, the data samples are divided into training and validation sets, providing a robust setup for model validation.

Using these optimizing engines, Lohrasb effectively estimates the optimal parameters for your selected estimator. This results in an enhanced model performance, optimized specifically for your data and problem space. 
### Installation

Lohrasb package is available on PyPI and can be installed with pip:

```sh
pip install lohrasb
```


### Supported estimators for this package
Lohrasb supports almost all machine learning estimators for classification and regression.

### Usage

Lohrasb presents an effective solution for tuning the optimal parameters of a machine learning model. It leverages robust optimization engines, namely [Optuna](https://optuna.readthedocs.io/en/stable/index.html), [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), along with [TuneGridSearchCV, and TuneSearchCV](https://docs.ray.io/en/latest/tune/api_docs/sklearn.html) from [Ray](https://docs.ray.io/en/latest/index.html) tune Scikit-Learn API (tune.sklearn). 

These capabilities empower Lohrasb users to perform comprehensive hyperparameter tuning on a broad range of machine learning models. Whether you are using a model from Scikit-learn, CatBoost, LightGBM, or even a custom model, Lohrasb's functionality enables you to extract the best performance from your model by optimizing its parameters using the most suitable engine.
### Some examples
In the following section, we will showcase several examples that illustrate how users can leverage various optimization engines incorporated within Lohrasb for effective hyperparameter tuning. This guidance aims to equip users with practical knowledge for harnessing the full potential of Lohrasb's diverse optimization capabilities.

#### Utilizing Ray Tune for Hyperparameter Optimization
Lohrasb's optimize_by_tune feature seamlessly integrates the powerful Tune tool from Ray, thereby streamlining hyperparameter optimization for Scikit-learn-based machine learning models. This feature harmoniously combines Tune's robust capabilities with a user-friendly interface, reducing the complexity of hyperparameter tuning and increasing its accessibility. Consequently, optimize_by_tune allows developers to concentrate on core model development while effectively managing hyperparameter optimization. This process leverages the full range of Tune's advanced functionalities. See the example below on how to utilize it:

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
#### Embracing GridSearchCV for Hyperparameter Optimization
The `optimize_by_gridsearchcv` function in Lohrasb incorporates GridSearchCV's robust capabilities, making the process of hyperparameter optimization streamlined and efficient, specifically for Scikit-learn-based machine learning models. This function merges GridSearchCV's comprehensive search abilities with a user-friendly interface, thereby simplifying hyperparameter tuning and making it more accessible. 
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
#### Exploring the Use of RandomizedSearchCV Interface
The `optimize_by_randomsearchcv` function in Lohrasb harnesses the robust capabilities of RandomizedSearchCV, thereby simplifying and enhancing the efficiency of hyperparameter optimization, particularly for Scikit-learn-based machine learning models. By merging RandomizedSearchCV's stochastic search capabilities with an intuitive interface, `optimize_by_randomsearchcv` makes the process of hyperparameter tuning more accessible and less complex. 

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
#### Streamlining Optimization with `optimize_by_optunasearchcv`
Lohrasb's `optimize_by_optunasearchcv` utilizes the power and flexibility of OptunaSearchCV, streamlining hyperparameter optimization for Scikit-learn models. This function melds Optuna's robust search abilities with an intuitive interface, simplifying tuning tasks. It allows developers to focus on key model development aspects while managing hyperparameter optimization using OptunaSearchCV's advanced features. 
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
#### Enhancing Optimization with `optimize_by_tunegridsearchcv`
TuneGridSearchCV is a highly versatile extension of Tune's capabilities, designed to replace Scikit-learn's GridSearchCV. It leverages Tune's scalability and flexibility to perform efficient hyperparameter searching over a predefined grid, offering precise and comprehensive tuning for diverse machine learning frameworks including Scikit-learn, CatBoost, LightGBM, and Imbalanced-learn.

The `optimize_by_tunegridsearchcv` feature in Lohrasb harnesses this power and versatility. This function simplifies and enhances hyperparameter optimization not only for Scikit-learn models, but also for models developed using CatBoost, LightGBM, and Imbalanced-learn. By leveraging TuneGridSearchCV's systematic and efficient grid-based search capabilities, `optimize_by_tunegridsearchcv` offers a user-friendly interface that makes hyperparameter tuning less complex and more accessible. This enables developers to focus on the core aspects of model development, while the `optimize_by_tunegridsearchcv` function efficiently manages the detailed tuning process. Hence, `optimize_by_tunegridsearchcv` enriches the overall machine learning workflow, utilizing TuneGridSearchCV's advanced features for a robust and efficient grid-based search across multiple frameworks.

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

#### Illustrating the Use of `optimize_by_tunesearchcv`
TuneSearchCV is a flexible and powerful tool that combines the strengths of Tune, a project by Ray, with the convenience of Scikit-learn's GridSearchCV and RandomizedSearchCV for hyperparameter tuning. TuneSearchCV provides an optimized and scalable solution for hyperparameter search, capable of handling a large number of hyperparameters and high-dimensional spaces with precision and speed.

The `optimize_by_tunesearchcv` feature within Lohrasb employs this powerhouse to make hyperparameter tuning easier and more efficient. 
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
#### Navigating Hyperparameter Tuning with `optimize_by_optuna`
The `optimize_by_optuna` feature in Lohrasb is a versatile function that leverages the extensive capabilities of the Optuna framework, aiming to simplify hyperparameter tuning for a wide range of machine learning models, including CatBoost, XGBoost, LightGBM, and Scikit-learn models. Optuna, known for its flexibility and efficiency in hyperparameter optimization, significantly enhances the model training process.

This function provides a flexible and customizable interface, accommodating a variety of machine learning tasks. Users can manipulate arguments for different Optuna submodules, such as 'study' and 'optimize', to tailor the function to their specific needs. This flexibility empowers developers to create and manage comprehensive optimization tasks with ease, all within their specific context.

In essence, `optimize_by_optuna` simplifies the tuning process by making the robust capabilities of Optuna readily accessible. Developers can focus on the core aspects of model development, with `optimize_by_optuna` managing the complexity of hyperparameter optimization. Thus, `optimize_by_optuna` augments the machine learning workflow, tapping into Optuna's advanced capabilities to deliver efficient, tailor-made hyperparameter optimization solutions. 

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
#### More Real-World Scenarios 

Lohrasb is not just limited to the above functionalities; it offers a multitude of solutions to tackle a variety of problems in machine learning. To get a better understanding of how Lohrasb can be utilized in real-world scenarios, you can visit the [examples](https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb/examples) webpage. Here you will find a plethora of practical applications demonstrating how Lohrasb's various modules can be adapted to solve specific challenges in hyperparameter tuning across different machine learning frameworks.

### Summary
Lohrasb offers a range of modules specifically designed to simplify and streamline the process of hyperparameter optimization across multiple machine learning frameworks. It integrates the power of various hyperparameter optimization tools such as Tune, GridSearchCV, RandomizedSearchCV, OptunaSearchCV, TuneGridSearchCV, and TuneSearchCV, and brings them into a single, easy-to-use interface.

The `optimize_by_tune` feature melds the robust abilities of Tune with a user-friendly interface, while `optimize_by_gridsearchcv` and `optimize_by_randomsearchcv` employ the exhaustive and stochastic search capabilities of GridSearchCV and RandomizedSearchCV, respectively. The `optimize_by_optunasearchcv` function leverages the flexibility of OptunaSearchCV, and `optimize_by_tunegridsearchcv` and `optimize_by_tunesearchcv` utilize Tune's scalability for grid and randomized searches. In addition, the `optimize_by_optuna` function harnesses the extensive capabilities of the Optuna framework, providing a customizable interface for various machine learning tasks. Across multiple machine learning frameworks, including Scikit-learn, CatBoost, LightGBM, and Imbalanced-learn, Lohrasb provides accessible and efficient tools for hyperparameter tuning, enabling developers to focus on core model development.

### References

We gratefully acknowledge the following open-source libraries which have been essential for developing Lohrasb:

1. **Scikit-learn** - Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830. [Website](https://scikit-learn.org/stable/)

2. **GridSearchCV & RandomizedSearchCV** - Part of Scikit-learn library. Refer to the above citation.

3. **Tune (Ray)** - Liaw, R., Liang, E., Nishihara, R., Moritz, P., Gonzalez, J.E., and Stoica, I. (2020). Tune: A Research Platform for Distributed Model Selection and Training. arXiv preprint arXiv:2001.04935. [Website](https://docs.ray.io/en/master/tune/)

4. **Optuna** - Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '19). Association for Computing Machinery, New York, NY, USA, 2623–2631. [Website](https://optuna.org/)

5. **Feature-engine** - Sole, S. (2020). Feature-engine. [Website](https://feature-engine.readthedocs.io/)

6. **XGBoost** - Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16). Association for Computing Machinery, New York, NY, USA, 785–794. [Website](https://xgboost.readthedocs.io/en/latest/)

7. **CatBoost** - Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. In Advances in Neural Information Processing Systems. [Website](https://catboost.ai/)

8. **LightGBM** - Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In Advances in Neural Information Processing Systems. [Website](https://lightgbm.readthedocs.io/en/latest/)


### License
Licensed under the [BSD 2-Clause](https://opensource.org/licenses/BSD-2-Clause) License.