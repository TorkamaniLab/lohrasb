# General purpose
from ray import air,tune
from ray.air import session
from ray.tune.search.hyperopt import HyperOptSearch
import joblib
import numpy as np
import os
import pytest
import time


# Datasets
from sklearn.datasets import make_classification, make_regression

# Model selection tools
from sklearn.model_selection import KFold, train_test_split

# Metrics
from sklearn.metrics import f1_score, make_scorer, r2_score

# Custom estimator
from lohrasb.best_estimator import BaseModel

from lohrasb.tests_conf import *

@pytest.mark.parametrize("estimator, params", estimators_params_tunesearch_clfs)
def test_optimize_by_tune_classification(estimator, params):
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
                "scoring": "f1",
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

    # Check if f1 score is above acceptable threshold (0.5 here)
    assert (
        f1_score(y_test, y_pred, average="macro") > 0.5
    )  # change f1_score to support multiclass


@pytest.mark.parametrize("estimator, params", estimators_params_tunesearch_clfs)
def test_optimize_by_tune_overfitting_classification(estimator, params):
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
                "scoring": "f1",
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
    score_train = f1_score(y_train, obj.predict(X_train), average="macro")
    score_test = f1_score(y_test, obj.predict(X_test), average="macro")
    assert score_train - score_test < 0.25, "The model is overfitting."


@pytest.mark.parametrize("estimator, params", estimators_params_tunesearch_clfs)
def test_optimize_by_tune_model_persistence_classification(estimator, params):
    search_alg = HyperOptSearch()
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=3,
        n_redundant=10,
        n_classes=3,
        random_state=42,
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
                "scoring": "f1",
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
    obj.fit(X, y)
    joblib.dump(obj, "test_model.pkl")
    loaded_model = joblib.load("test_model.pkl")
    assert np.allclose(
        obj.predict(X), loaded_model.predict(X)
    ), "The saved model does not match the loaded model."
    os.remove("test_model.pkl")


@pytest.mark.parametrize("estimator, params", estimators_params_tunesearch_clfs)
def test_optimize_by_tune_efficiency_classification(estimator, params):
    search_alg = HyperOptSearch()
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=3,
        n_redundant=10,
        n_classes=3,
        random_state=42,
    )
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
                "scoring": "f1",
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

    start_time = time.time()
    obj = BaseModel().optimize_by_tune(**kwargs)
    obj.fit(X, y)
    end_time = time.time()
    assert end_time - start_time < 100, "The model took too long to train."



@pytest.mark.parametrize("estimator, params", estimators_params_tunesearch_regs)
def test_optimize_by_tune_regression(estimator, params):
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
    score = r2_score(y, predictions)
    assert (
        score >= 0.7
    ), f"Expected r2_score to be greater than or equal to 0.7, but got {score}"


@pytest.mark.parametrize("estimator, params", estimators_params_tunesearch_regs)
def test_optimize_by_tune_overfitting_regression(estimator, params):
    search_alg = HyperOptSearch()
    # Create synthetic dataset
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1
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

    # Run optimize_by_tune
    obj = BaseModel().optimize_by_tune(**kwargs)
    obj.fit(X_train, y_train)
    score_train = r2_score(y_train, obj.predict(X_train))
    score_test = r2_score(y_test, obj.predict(X_test))
    assert score_train - score_test < 0.25, "The model is overfitting."


@pytest.mark.parametrize("estimator, params", estimators_params_tunesearch_regs)
def test_optimize_by_tune_model_persistence_regression(estimator, params):
    search_alg = HyperOptSearch()
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

    # Run optimize_by_tune
    obj = BaseModel().optimize_by_tune(**kwargs)
    obj.fit(X, y)
    joblib.dump(obj, "test_model.pkl")
    loaded_model = joblib.load("test_model.pkl")
    assert np.allclose(
        obj.predict(X), loaded_model.predict(X)
    ), "The saved model does not match the loaded model."
    os.remove("test_model.pkl")


@pytest.mark.parametrize("estimator, params", estimators_params_tunesearch_regs)
def test_optimize_by_tune_efficiency_regression(estimator, params):
    search_alg = HyperOptSearch()
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1
    )
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
                "estimator": est
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

    start_time = time.time()
    obj = BaseModel().optimize_by_tune(**kwargs)
    obj.fit(X, y)
    end_time = time.time()
    assert end_time - start_time < 100, "The model took too long to train."
