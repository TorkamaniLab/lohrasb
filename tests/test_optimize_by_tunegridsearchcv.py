# General purpose
import dill
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

@pytest.mark.parametrize('estimator, params', estimators_params_clfs)
def test_optimize_by_tunegridsearchcv_classification(estimator, params):
    # Create synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=3, n_redundant=10, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the estimator
    est = estimator()

    # Create keyword arguments for tunegridSearchCV
    kwargs = {
        'kwargs': {
            'tunegrid_search_kwargs': {
                'estimator': est,
                'param_grid': params,
                'scoring': 'f1_macro', # change scoring to support multiclass
                'verbose':3,
                'n_jobs':None,
                'cv':KFold(3),
                'early_stopping':None, 
                'refit':True, 
                'error_score':'raise', 
                'return_train_score':False, 
                'local_dir':'~/ray_results', 
                'name':None, 
                'max_iters':1, 
                'use_gpu':False, 
                'loggers':None, 
                'pipeline_auto_early_stop':True, 
                'stopper':None, 
                'time_budget_s':None, 
                'mode':None,
             },
            'main_tunegrid_kwargs': {},
            'fit_tune_kwargs': {}
        }
    }

    # Run tests/test_optimize_by_tunegridsearchcv
    obj = BaseModel().optimize_by_tunegridsearchcv(**kwargs)
    obj.fit(X_train,y_train)

    # Predict on test data
    y_pred = obj.predict(X_test)

    # Check if f1 score is above acceptable threshold (0.5 here)
    assert f1_score(y_test, y_pred, average='macro') > 0.5  # change f1_score to support multiclass


@pytest.mark.parametrize('estimator, params', estimators_params_clfs)
def test_optimize_by_tunegridsearchcv_overfitting_classification(estimator, params):
    # Create synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=3, n_redundant=10, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the estimator
    est = estimator()

    # Create keyword arguments for tunegridSearchCV
    kwargs = {
        'kwargs': {
            'tunegrid_search_kwargs': {
                'estimator': est,
                'param_grid': params,
                'scoring': 'f1_macro', # change scoring to support multiclass
                'verbose':3,
                'n_jobs':None,
                'cv':KFold(3),
                'early_stopping':None, 
                'refit':True, 
                'error_score':'raise', 
                'return_train_score':False, 
                'local_dir':'~/ray_results', 
                'name':None, 
                'max_iters':1, 
                'use_gpu':False, 
                'loggers':None, 
                'pipeline_auto_early_stop':True, 
                'stopper':None, 
                'time_budget_s':None, 
                'mode':None,
             },
            'main_tunegrid_kwargs': {},
            'fit_tune_kwargs': {}
        }
    }

    # Run optimize_by_tunegridsearchcv
    obj = BaseModel().optimize_by_tunegridsearchcv(**kwargs)
    obj.fit(X_train,y_train)
    score_train = f1_score(y_train, obj.predict(X_train), average='macro')
    score_test = f1_score(y_test, obj.predict(X_test), average='macro')
    assert score_train - score_test < 0.25, "The model is overfitting."


@pytest.mark.parametrize('estimator, params', estimators_params_clfs)
def test_optimize_by_tunegridsearchcv_model_persistence_classification(estimator, params):
    X, y = make_classification(n_samples=100, n_features=20, n_informative=3, n_redundant=10, n_classes=3, random_state=42)

    # Initialize the estimator
    est = estimator()
    # Create keyword arguments for tunegridSearchCV
    kwargs = {
        'kwargs': {
            'tunegrid_search_kwargs': {
                'estimator': est,
                'param_grid': params,
                'scoring': 'f1_macro', # change scoring to support multiclass
                'verbose':3,
                'n_jobs':None,
                'cv':KFold(3),
                'early_stopping':None, 
                'refit':True, 
                'error_score':'raise', 
                'return_train_score':False, 
                'local_dir':'~/ray_results', 
                'name':None, 
                'max_iters':1, 
                'use_gpu':False, 
                'loggers':None, 
                'pipeline_auto_early_stop':True, 
                'stopper':None, 
                'time_budget_s':None, 
                'mode':None,
             },
            'main_tunegrid_kwargs': {},
            'fit_tune_kwargs': {}
        }
    }

    # Run optimize_by_tunegridsearchcv
    obj = BaseModel().optimize_by_tunegridsearchcv(**kwargs)
    obj.fit(X,y)
    with open(obj.__class__.__name__+'test_model.pkl','wb') as f:
        dill.dump(obj, f)
    with open(obj.__class__.__name__+'test_model.pkl','rb') as f:
        loaded_model = dill.load(f)
    assert np.allclose(obj.predict(X), loaded_model.predict(X)), "The saved model does not match the loaded model."
    os.remove(obj.__class__.__name__+'test_model.pkl')

@pytest.mark.parametrize('estimator, params', estimators_params_clfs)
def test_optimize_by_tunegridsearchcv_efficiency_classification(estimator, params):
    X, y = make_classification(n_samples=100, n_features=20, n_informative=3, n_redundant=10, n_classes=3, random_state=42)
    est = estimator()
    # Create keyword arguments for tunegridSearchCV
    kwargs = {
        'kwargs': {
            'tunegrid_search_kwargs': {
                'estimator': est,
                'param_grid': params,
                'scoring': 'f1_macro', # change scoring to support multiclass
                'verbose':3,
                'n_jobs':None,
                'cv':KFold(3),
                'early_stopping':None, 
                'refit':True, 
                'error_score':'raise', 
                'return_train_score':False, 
                'local_dir':'~/ray_results', 
                'name':None, 
                'max_iters':1, 
                'use_gpu':False, 
                'loggers':None, 
                'pipeline_auto_early_stop':True, 
                'stopper':None, 
                'time_budget_s':None, 
                'mode':None,
            },
            'main_tunegrid_kwargs': {},
            'fit_tune_kwargs': {}
        }
    }

    start_time = time.time()
    obj = BaseModel().optimize_by_tunegridsearchcv(**kwargs)
    obj.fit(X, y)
    end_time = time.time()
    assert end_time - start_time < 1000, "The model took too long to train."


@pytest.mark.parametrize('estimator, params', estimators_params_regs)
def test_optimize_by_tunegridsearchcv_regression(estimator, params):
    # Create synthetic regression dataset
    X, y = make_regression(n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1)

    # Initialize the estimator
    est = estimator()

    # Create keyword arguments for tunegridSearchCV
    kwargs = {
        'kwargs': {
            'tunegrid_search_kwargs': {
                'estimator': est,
                'param_grid': params,
                'scoring':make_scorer(r2_score, greater_is_better=True),
                'verbose':3,
                'n_jobs':None,
                'cv':KFold(3),
                'early_stopping':None, 
                'refit':True, 
                'error_score':'raise', 
                'return_train_score':False, 
                'local_dir':'~/ray_results', 
                'name':None, 
                'max_iters':1, 
                'use_gpu':False, 
                'loggers':None, 
                'pipeline_auto_early_stop':True, 
                'stopper':None, 
                'time_budget_s':None, 
                'mode':None,
             },
            'main_tunegrid_kwargs': {},
            'fit_tune_kwargs': {}
        }
    }

    # Create obj of the class
    obj = BaseModel().optimize_by_tunegridsearchcv(
        **kwargs
    )

    # Check if instance created successfully
    assert obj is not None

    # Fit data and predict
    obj.fit(X, y)
    predictions = obj.predict(X)
    score = r2_score(y, predictions)
    assert score >= 0.7, f"Expected r2_score to be greater than or equal to 0.7, but got {score}"

@pytest.mark.parametrize('estimator, params', estimators_params_regs)
def test_optimize_by_tunegridsearchcv_overfitting_regression(estimator, params):
    # Create synthetic dataset
    X, y = make_regression(n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the estimator
    est = estimator()

    # Create keyword arguments for tunegridSearchCV
    kwargs = {
        'kwargs': {
            'tunegrid_search_kwargs': {
                'estimator': est,
                'param_grid': params,
                'scoring':make_scorer(r2_score, greater_is_better=True),
                'verbose':3,
                'n_jobs':None,
                'cv':KFold(3),
                'early_stopping':None, 
                'refit':True, 
                'error_score':'raise', 
                'return_train_score':False, 
                'local_dir':'~/ray_results', 
                'name':None, 
                'max_iters':1, 
                'use_gpu':False, 
                'loggers':None, 
                'pipeline_auto_early_stop':True, 
                'stopper':None, 
                'time_budget_s':None, 
                'mode':None,
             },
            'main_tunegrid_kwargs': {},
            'fit_tune_kwargs': {}
        }
    }

    # Run optimize_by_tunegridsearchcv
    obj = BaseModel().optimize_by_tunegridsearchcv(**kwargs)
    obj.fit(X_train,y_train)
    score_train = r2_score(y_train, obj.predict(X_train))
    score_test = r2_score(y_test, obj.predict(X_test))
    assert score_train - score_test < 0.25, "The model is overfitting."


@pytest.mark.parametrize('estimator, params', estimators_params_regs)
def test_optimize_by_tunegridsearchcv_model_persistence_regression(estimator, params):
    X, y = make_regression(n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1)
    # Initialize the estimator
    est = estimator()
    # Create keyword arguments for tunegridSearchCV
    kwargs = {
        'kwargs': {
            'tunegrid_search_kwargs': {
                'estimator': est,
                'param_grid': params,
                'scoring':make_scorer(r2_score, greater_is_better=True),
                'verbose':3,
                'n_jobs':None,
                'cv':KFold(3),
                'early_stopping':None, 
                'refit':True, 
                'error_score':'raise', 
                'return_train_score':False, 
                'local_dir':'~/ray_results', 
                'name':None, 
                'max_iters':1, 
                'use_gpu':False, 
                'loggers':None, 
                'pipeline_auto_early_stop':True, 
                'stopper':None, 
                'time_budget_s':None, 
                'mode':None,
             },
            'main_tunegrid_kwargs': {},
            'fit_tune_kwargs': {}
        }
    }

    # Run optimize_by_tunegridsearchcv
    obj = BaseModel().optimize_by_tunegridsearchcv(**kwargs)
    obj.fit(X,y)
    with open(obj.__class__.__name__+'test_model.pkl','wb') as f:
        dill.dump(obj, f)
    with open(obj.__class__.__name__+'test_model.pkl','rb') as f:
        loaded_model = dill.load(f)
    assert np.allclose(obj.predict(X), loaded_model.predict(X)), "The saved model does not match the loaded model."
    os.remove(obj.__class__.__name__+'test_model.pkl')

@pytest.mark.parametrize('estimator, params', estimators_params_regs)
def test_optimize_by_tunegridsearchcv_efficiency_regression(estimator, params):
    X, y = make_regression(n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1)
    est = estimator()
    # Create keyword arguments for tunegridSearchCV
    kwargs = {
        'kwargs': {
            'tunegrid_search_kwargs': {
                'estimator': est,
                'param_grid': params,
                'scoring':make_scorer(r2_score, greater_is_better=True),
                'verbose':3,
                'n_jobs':None,
                'cv':KFold(3),
                'early_stopping':None, 
                'refit':True, 
                'error_score':'raise', 
                'return_train_score':False, 
                'local_dir':'~/ray_results', 
                'name':None, 
                'max_iters':1, 
                'use_gpu':False, 
                'loggers':None, 
                'pipeline_auto_early_stop':True, 
                'stopper':None, 
                'time_budget_s':None, 
                'mode':None,
             },
            'main_tunegrid_kwargs': {},
            'fit_tune_kwargs': {}
        }
    }

    start_time = time.time()
    obj = BaseModel().optimize_by_tunegridsearchcv(**kwargs)
    obj.fit(X, y)
    end_time = time.time()
    assert end_time - start_time < 1000, "The model took too long to train."
