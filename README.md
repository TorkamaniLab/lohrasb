# lohrasb

lohrasb is a package built to ease machine learning development. It uses [Optuna](https://optuna.readthedocs.io/en/stable/index.html), [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), and [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) to tune most of the tree-based estimators of sickit-learn. It is compatible with [scikit-learn](https://scikit-learn.org) pipeline.


## Introduction

BaseModel of the Lohrasb package can receive various parameters. From an estimator class to its tunning parameters and GridsearchCV, RandomizedSearchCV, or Optuna to their parameters. Samples will be split to train and validation set, and then optimization will estimate optimal related parameters using these optimizing engines.

## Installation

lohrasb package is available on PyPI and can be installed with pip:

```sh
pip install lohrasb
```


## Supported estimators for this package
Almost all machine learning estimators for classification and regression supported by Lohrasb.

## Usage

- Tunning best parameters of a machine learning model using [Optuna](https://optuna.readthedocs.io/en/stable/index.html) , [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) or [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).

## Factories
For ease of use of BestModel, some factories are available to build associated instances corresponding to each optimization engine. For example, the following factory can be used for  GridSearchCV:

```
obj = BaseModel().optimize_by_gridsearchcv(
            estimator=XGBClassifier(),
            estimator_params={
                            "booster": ["gbtree","dart"],
                            "eval_metric": ["auc"],
                            "max_depth": [4, 5],
                            "gamma": [0.1, 1.2],
                            "subsample": [0.8],
                        },
            measure_of_accuracy="f1_score",
            verbose=3,
            n_jobs=-1,
            random_state=42,
            cv=KFold(2),
        )
```

## One example : Computer Hardware (Part 1: Use BestModel in sklearn pipeline)

#### Import some required libraries
```
from lohrasb.best_estimator import BaseModel
from optuna.pruners import HyperbandPruner
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.model_selection import KFold,train_test_split
import pandas as pd
import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from feature_engine.imputation import (
    CategoricalImputer,
    MeanMedianImputer
    )
from category_encoders import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score)
from sklearn.metrics import f1_score, mean_absolute_error,r2_score
from sklearn.linear_model import *
from sklearn.svm import *
from xgboost import *
from sklearn.linear_model import *
from catboost import *
from lightgbm import *
from sklearn.neural_network import *
from imblearn.ensemble import *
from sklearn.ensemble import *

```
#### Computer Hardware Data Set (a regression problem)
  
https://archive.ics.uci.edu/ml/datasets/Computer+Hardware

```
urldata= "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data"
# column names
col_names=[
    "vendor name",
    "Model Name",
    "MYCT",
    "MMIN",
    "MMAX",
    "CACH",
    "CHMIN",
    "CHMAX",
    "PRP"
]
# read data
data = pd.read_csv(urldata,header=None,names=col_names,sep=',')
data
```
#### Train test split

```
X = data.loc[:, data.columns != "PRP"]
y = data.loc[:, data.columns == "PRP"]
y = y.values.ravel()


X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33, random_state=42)

```
#### Find feature types for later use

```
int_cols =  X_train.select_dtypes(include=['int']).columns.tolist()
float_cols =  X_train.select_dtypes(include=['float']).columns.tolist()
cat_cols =  X_train.select_dtypes(include=['object']).columns.tolist()
```

####  Define estimator and set its arguments  
```
estimator = LinearRegression()
estimator_params= {
        "fit_intercept": [True, False],
    }
```
#### Use factory

```
obj = BaseModel().optimize_by_optuna(
            estimator=estimator,
            estimator_params=estimator_params,
            measure_of_accuracy="r2_score",
            with_stratified=False,
            test_size=.3,
            add_extra_args_for_measure_of_accuracy = False,
            verbose=3,
            n_jobs=-1,
            random_state=42,
            # optuna params
            # optuna study init params
            study=optuna.create_study(
                storage=None,
                sampler=TPESampler(),
                pruner=HyperbandPruner(),
                study_name=None,
                direction="maximize",
                load_if_exists=False,
                directions=None,
            ),
            # optuna optimization params
            study_optimize_objective=None,
            study_optimize_objective_n_trials=10,
            study_optimize_objective_timeout=600,
            study_optimize_n_jobs=-1,
            study_optimize_catch=(),
            study_optimize_callbacks=None,
            study_optimize_gc_after_trial=False,
            study_optimize_show_progress_bar=False,
        )
```
####  Build sklearn pipeline
```
pipeline =Pipeline([
            # int missing values imputers
            ('intimputer', MeanMedianImputer(
                imputation_method='median', variables=int_cols)),
            # category missing values imputers
            ('catimputer', CategoricalImputer(variables=cat_cols)),
            #
            ('catencoder', OrdinalEncoder()),
            # regression model 
            ('obj', obj),


 ])


```
#### Run Pipeline

```
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
```
#### Check performance of the pipeline

```
print('r2 score : ')
print(r2_score(y_test,y_pred))
```

## Part 2:  Use BestModel as a standalone estimator 
```
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33, random_state=42)
```

#### Transform features to make them ready for model input
```
transform_pipeline =Pipeline([
            # int missing values imputers
            ('intimputer', MeanMedianImputer(
                imputation_method='median', variables=int_cols)),
            # category missing values imputers
            ('catimputer', CategoricalImputer(variables=cat_cols)),
            #
            ('catencoder', OrdinalEncoder()),
            # classification model

 ])
```
#### Transform X_train and X_test

```
X_train=transform_pipeline.fit_transform(X_train,y_train)
X_test=transform_pipeline.transform(X_test)
```

#### Train model and predict
```
obj.fit(X_train,y_train)
y_pred = obj.predict(X_test)
```

#### Check performance of the model

```
print('r2 score : ')
print(r2_score(y_test,y_pred))

print(obj.get_best_estimator())

print(obj.best_estimator)

OptunaObj = obj.get_optimized_object()
print(OptunaObj.trials)
```



There are some examples  available in the [examples](https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb/examples). 

## License
Licensed under the [BSD 2-Clause](https://opensource.org/licenses/BSD-2-Clause) License.