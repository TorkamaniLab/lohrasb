import os
import time
import pytest
import joblib
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, r2_score
from lohrasb.best_estimator import BaseModel
from lohrasb.tests_conf import estimators_params_regs, estimators_params_clfs

TUNE_SEARCH_KWARGS = {
    'verbose': 3,
    'n_jobs': None,
    'cv': KFold(2),
    'early_stopping': None,
    'refit': True,
    'error_score': 'raise',
    'return_train_score': False,
    'local_dir': '~/ray_results',
    'name': None,
    'max_iters': 1,
    'use_gpu': False,
    'loggers': None,
    'pipeline_auto_early_stop': True,
    'stopper': None,
    'time_budget_s': None,
    'mode': None,
}

def setup_test(estimator, params, score_func, task_type):
    data_params = {
        'classification': {
            'func': make_classification,
            'args': {'n_samples': 1000, 'n_features': 20, 'n_informative': 3, 'n_redundant': 10, 'n_classes': 3, 'random_state': 42}
        },
        'regression': {
            'func': make_regression,
            'args': {'n_samples': 100, 'n_features': 10, 'n_informative': 5, 'n_targets': 1, 'random_state': 1}
        }
    }

    X, y = data_params[task_type]['func'](**data_params[task_type]['args'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    est = estimator()
    kwargs = {
        'kwargs': {
            'tune_search_kwargs': {**TUNE_SEARCH_KWARGS, 'estimator': est, 'param_distributions': params, 'scoring': score_func},
            'main_tune_kwargs': {},
            'fit_tune_kwargs': {}
        }
    }

    obj = BaseModel().optimize_by_tunesearchcv(**kwargs)
    obj.fit(X_train, y_train)
    score = obj.predict(X_train)

    return obj, score, X_test, y_test

for test_type, func in [('clf', f1_score), ('reg', r2_score)]:
    @pytest.mark.parametrize('estimator, params', globals()[f'estimators_params_{test_type}s'])
    def test_performance(estimator, params):
        obj, score, X_test, y_test = setup_test(estimator, params, 'f1_macro' if test_type == 'clf' else 'r2', 'classification' if test_type == 'clf' else 'regression')
        assert func(y_test, obj.predict(X_test)) > 0.5

    @pytest.mark.parametrize('estimator, params', globals()[f'estimators_params_{test_type}s'])
    def test_overfitting(estimator, params):
        obj, score, X_test, y_test = setup_test(estimator, params, 'f1_macro' if test_type == 'clf' else 'r2', 'classification' if test_type == 'clf' else 'regression')
        score_test = func(y_test, obj.predict(X_test), average='macro')
        assert score - score_test < 0.25, "The model is overfitting."

    @pytest.mark.parametrize('estimator, params', globals()[f'estimators_params_{test_type}s'])
    def test_persistence(estimator, params):
        obj, X_test, y_test = setup_test(estimator, params, 'f1_macro' if test_type == 'clf' else 'r2', 'classification' if test_type == 'clf' else 'regression')
        model_path = 'test_model.pkl'
        joblib.dump(obj, model_path)
        loaded_model = joblib.load(model_path)
        assert np.allclose(obj.predict(X_test), loaded_model.predict(X_test)), "The saved model does not match the loaded model."
        os.remove(model_path)

    @pytest.mark.parametrize('estimator, params', globals()[f'estimators_params_{test_type}s'])
    def test_efficiency(estimator, params):
        start_time = time.time()
        obj, X_test, y_test = setup_test(estimator, params, 'f1_macro' if test_type == 'clf' else 'r2', 'classification' if test_type == 'clf' else 'regression')
        end_time = time.time()
        assert end_time - start_time < 100, "The model took too long to train."
