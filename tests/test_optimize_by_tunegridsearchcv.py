# General purpose
import dill
import numpy as np
import os
import pytest
import time

# Datasets
from sklearn.datasets import make_classification, make_regression

# Estimators
# Linear models
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, SGDRegressor

# Gradient boosting frameworks
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

# Ensemble methods
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor, 
                              GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)

# Other classifiers and regressors
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, LinearSVR, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Imbalanced-learn ensemble
from imblearn.ensemble import BalancedRandomForestClassifier

# Model selection tools
from sklearn.model_selection import KFold, train_test_split

# Metrics
from sklearn.metrics import f1_score, make_scorer, r2_score

# Custom estimator
from lohrasb.best_estimator import BaseModel

# Define hyperparameters for each model
lr_params = {'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
xgb_params = {'n_estimators': [50, 100, 200], 'max_depth': [6, 10, 15], 'learning_rate': [0.001, 0.01, 0.1]}
cb_params = {'iterations': [50, 100, 200], 'depth': [4, 6, 8], 'learning_rate': [0.001, 0.01, 0.1]}
lgbm_params = {'n_estimators': [50, 100, 200], 'max_depth': [-1, 10, 20], 'learning_rate': [0.001, 0.01, 0.1]}
brf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
mlp_params = {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh', 'logistic']}
gbc_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [3, 5, 10]}
abc_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.001, 0.01, 0.1]}
svc_params = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
dtc_params = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10]}
etc_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
gnb_params = {}

# Put models and hyperparameters into a list of tuples
estimators_params = [(LogisticRegression, lr_params),
                     (RandomForestClassifier, rf_params),
                     (XGBClassifier, xgb_params),
                     (CatBoostClassifier, cb_params),
                     (LGBMClassifier, lgbm_params),
                     (BalancedRandomForestClassifier, brf_params),
                     (MLPClassifier, mlp_params),
                     (GradientBoostingClassifier, gbc_params),
                     (AdaBoostClassifier, abc_params),
                     (SVC, svc_params),
                     (KNeighborsClassifier, knn_params),
                     (DecisionTreeClassifier, dtc_params),
                     (ExtraTreesClassifier, etc_params),
                     (GaussianNB, gnb_params)]

@pytest.mark.parametrize('estimator, params', estimators_params)
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


@pytest.mark.parametrize('estimator, params', estimators_params)
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


@pytest.mark.parametrize('estimator, params', estimators_params)
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

@pytest.mark.parametrize('estimator, params', estimators_params)
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





# Define hyperparameters for each model
lr_params = {'fit_intercept': [True, False]}
ridge_params = {'alpha': [0.5, 1.0]}
lasso_params = {'alpha': [0.5, 1.0]}
elastic_params = {'alpha': [0.5, 1.0], 'l1_ratio': [0.5, 0.7]}
xgb_params = {'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
lgbm_params = {'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
cb_params = {'depth': [3, 5], 'learning_rate': [0.01, 0.1]}
svr_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [0.5, 1.0]}
lsvr_params = {'C': [0.5, 1.0]}
# knr_params = {'n_neighbors': [5, 7, 10, 15, 20],  # Increasing the number of neighbors can help in making the model more generalized.
#     'weights': ['uniform', 'distance'],  # The 'distance' option can give more importance to closer instances, which may help reduce overfitting.
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # The algorithm used to compute the nearest neighbors can sometimes have an effect on overfitting, but it generally depends more on the dataset.
#     'p': [1, 2]  # This corresponds to the power parameter for the Minkowski metric. 1 is for manhattan_distance and 2 for euclidean_distance.
#     }
# dtr_params = {'max_depth': [5, 10], 'min_samples_split': [3, 5],'min_samples_leaf': [ 2, 3]}
rfr_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
gbr_params = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10], 'learning_rate': [0.01, 0.1]}
etr_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
abr_params = {'n_estimators': [50, 100, 200], 'learning_rate': [ 0.01, 0.1]}
# mlpr_params = {'hidden_layer_sizes': [(100,)], 'activation': ['relu', 'tanh']}
sgr_params = {'loss': [ 'huber', 'epsilon_insensitive'], 'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [ 0.001, 0.01]}

# Put models and hyperparameters into a list of tuples
estimators_params = [(LinearRegression, lr_params),
                     (Ridge, ridge_params),
                     (Lasso, lasso_params),
                     (ElasticNet, elastic_params),
                     (XGBRegressor, xgb_params),
                     (LGBMRegressor, lgbm_params),
                     (CatBoostRegressor, cb_params),
                     (SVR, svr_params),
                     (LinearSVR, lsvr_params),
                     # TODO overfit 
                     # (KNeighborsRegressor, knr_params),
                     # TODO overfit
                     # (DecisionTreeRegressor, dtr_params),
                     (RandomForestRegressor, rfr_params),
                     (GradientBoostingRegressor, gbr_params),
                     (ExtraTreesRegressor, etr_params),
                     (AdaBoostRegressor, abr_params),
                     # TODO underfit
                     # (MLPRegressor, mlpr_params),
                     (SGDRegressor, sgr_params)]


@pytest.mark.parametrize('estimator, params', estimators_params)
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

@pytest.mark.parametrize('estimator, params', estimators_params)
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


@pytest.mark.parametrize('estimator, params', estimators_params)
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

@pytest.mark.parametrize('estimator, params', estimators_params)
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
