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
