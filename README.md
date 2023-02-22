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

### Factories
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
            fit_params = None,
            measure_of_accuracy=make_scorer(f1_score, greater_is_better=True),
            # instead of measure_of_accuracy it is possible to use :  scoring="f1",
            verbose=3,
            n_jobs=-1,
            random_state=42,
            cv=KFold(2),
        )
```

#### Example 1: Computer Hardware (Part 1: Use BestModel in sklearn pipeline)

##### Import some required libraries
```
from lohrasb.best_estimator import BaseModel
import xgboost
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
from sklearn.metrics import (
    make_scorer)
from sklearn.metrics import r2_score
from sklearn.linear_model import *


```
##### Computer Hardware [Data Set](https://archive.ics.uci.edu/ml/datasets/Computer+Hardware)
 (a regression problem)
  

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
##### Train test split

```
X = data.loc[:, data.columns != "PRP"]
y = data.loc[:, data.columns == "PRP"]
y = y.values.ravel()


X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33, random_state=42)

```
##### Find feature types for later use

```
int_cols =  X_train.select_dtypes(include=['int']).columns.tolist()
float_cols =  X_train.select_dtypes(include=['float']).columns.tolist()
cat_cols =  X_train.select_dtypes(include=['object']).columns.tolist()
```

#####  Define estimator and set its arguments  
```
estimator = LinearRegression()
estimator_params= {
        "fit_intercept": [True, False],
    }
```
##### Use factory

```
obj = BaseModel().optimize_by_optuna(
            estimator=estimator,
            estimator_params=estimator_params,
            measure_of_accuracy="mean_absolute_error(y_true, y_pred, multioutput='uniform_average')",
            with_stratified=False,
            test_size=.3,
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
                direction="minimize",
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
#####  Build sklearn pipeline
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
##### Run Pipeline

```
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
```
##### Check performance of the pipeline

```
print('r2 score : ')
print(r2_score(y_test,y_pred))
```

##### Part 2: Use BestModel as a standalone estimator 
```
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33, random_state=42)
```

##### Transform features to make them ready for model input
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
##### Transform X_train and X_test

```
X_train=transform_pipeline.fit_transform(X_train,y_train)
X_test=transform_pipeline.transform(X_test)
```

##### Train model and predict
```
obj.fit(X_train,y_train)
y_pred = obj.predict(X_test)
```

##### Check performance of the model

```
print('r2 score : ')
print(r2_score(y_test,y_pred))
print('mean_absolute_error : ')
print(mean_absolute_error(y_test,y_pred))


print(obj.get_best_estimator())

print(obj.best_estimator)

OptunaObj = obj.get_optimized_object()
print(OptunaObj.trials)
```

#### Example 2: XGBoost Survival Embeddings (XGBSEKaplanNeighbors)
For more information refer to visit [XGBoost Survival Embeddings](https://loft-br.github.io/xgboost-survival-embeddings/examples/confidence_interval.html) page.


##### Import some required libraries
```
! pip3 install torch==1.12.1
import sys
sys.path.append('/usr/local/lib/python3.10/site-packages')
import torch
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from lohrasb.best_estimator import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from xgbse.converters import convert_to_structured
from xgbse.metrics import (
    concordance_index,
    approx_brier_score
)
from xgbse import (
    XGBSEKaplanNeighbors,
    XGBSEKaplanTree,
    XGBSEBootstrapEstimator
)
from pycox.datasets import metabric

```


##### Read data metabric

```
df = metabric.read_df()
df.head()
```

##### Define labels and train-test split 

```
# splitting to X, T, E format
X = df.drop(['duration', 'event'], axis=1)
y = convert_to_structured(df['duration'], df['event'])


# splitting between train, and validation 
(X_train, X_test,
 y_train, y_test) = \
train_test_split(X, y, test_size=0.2, random_state=42)
```


##### Define estimator and set its arguments
```
estimator_params = {
    'n_estimators' :[100,200]

}

PARAMS_TREE = {
    'objective': 'survival:cox',
    'eval_metric': 'cox-nloglik',
    'tree_method': 'hist', 
    'max_depth': 100, 
    'booster':'dart', 
    'subsample': 1.0,
    'min_child_weight': 50, 
    'colsample_bynode': 1.0
}
base_model = XGBSEKaplanTree(PARAMS_TREE)

TIME_BINS = np.arange(15, 315, 15)
```

##### Define estimator and fit params

```
estimator=XGBSEBootstrapEstimator(base_model)
fit_params = {"time_bins":TIME_BINS}
```
##### Define BaseModel estimator using random search CV

```
obj = BaseModel().optimize_by_randomsearchcv(
            estimator=estimator,
            fit_params = fit_params,
            estimator_params=estimator_params,
            measure_of_accuracy=make_scorer(approx_brier_score, greater_is_better=False),
            verbose=3,
            n_jobs=-1,
            n_iter=2,
            random_state=42,
            cv=KFold(2),
        )
```

##### Build sklearn pipeline

```
pipeline =Pipeline([
            ('obj', obj)

 ])
```
##### Run Pipeline
```
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
```

##### Check performance of the pipeline

```
print(f'C-index: {concordance_index(y_test, y_pred)}')
print(f'Avg. Brier Score: {approx_brier_score(y_test, y_pred)}')
```

#### Example 3: Using InterpretML

[InterpretML](https://interpret.ml/docs/getting-started) is a package that supports training interpretable models (glassbox), as well as explaining existing ML pipelines (blackbox). In this example we will use [Shapley Additive Explanations](https://interpret.ml/docs/shap.html) of this package on [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult) (a classification problem).

##### Import some required libraries

```
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from lohrasb.best_estimator import BaseModel
from sklearn.pipeline import Pipeline
from feature_engine.imputation import (
    CategoricalImputer,
    MeanMedianImputer
    )
from category_encoders import OrdinalEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score)
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from interpret import show
from interpret.blackbox import ShapKernel
import shap
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())```
```

##### Read data
```
urldata= "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
# column names
col_names=["age", "workclass", "fnlwgt" , "education" ,"education-num",
"marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week",
"native-country","label"
]
# read data
data = pd.read_csv(urldata,header=None,names=col_names,sep=',')
data.head()
```

##### Define labels
```
data.loc[data['label']=='<=50K','label']=0
data.loc[data['label']==' <=50K','label']=0

data.loc[data['label']=='>50K','label']=1
data.loc[data['label']==' >50K','label']=1

data['label']=data['label'].astype(int)
```

##### Train test split
```

X = data.loc[:, data.columns != "label"]
y = data.loc[:, data.columns == "label"]

X_train, X_test, y_train, y_test =train_test_split(X, y, \
     test_size=0.33, stratify=y['label'], random_state=42)

y_train=y_train.values.ravel()
y_test=y_test.values.ravel()
```


##### Find feature types for later use
```

int_cols =  X_train.select_dtypes(include=['int']).columns.tolist()
float_cols =  X_train.select_dtypes(include=['float']).columns.tolist()
cat_cols =  X_train.select_dtypes(include=['object']).columns.tolist()
```


##### Define estimator and set its arguments 
```

estimator = RandomForestClassifier()
estimator_params = {
        "max_depth": [4, 5],
        "n_estimators":[100,200],
        "max_features" :["sqrt", "log2"],


    }
    

blackbox_model = BaseModel().optimize_by_gridsearchcv(
            estimator=estimator,
            estimator_params=estimator_params,
            fit_params = None,
            measure_of_accuracy=make_scorer(f1_score, greater_is_better=True),
            verbose=3,
            n_jobs=-1,
            random_state=42,
            cv=KFold(2),
        )
```

##### Build sklearn pipeline


```

pipeline =Pipeline([
            # int missing values imputers
            ('intimputer', MeanMedianImputer(
                imputation_method='median', variables=int_cols)),
            # category missing values imputers
            ('catimputer', CategoricalImputer(variables=cat_cols)),
            #
            ('catencoder', OrdinalEncoder()),
            # classification model
            #('obj', RandomForestClassifier())

 ])
```


##### Run Pipeline
```

X_train = pipeline.fit_transform(X_train,y_train)
X_test = pipeline.transform(X_test)

blackbox_model.fit(X_train, y_train)
y_pred = blackbox_model.predict(X_test)
```


##### Check performance of the pipeline
```

print('F1 score : ')
print(f1_score(y_test,y_pred))
print('Classification report : ')
print(classification_report(y_test,y_pred))
print('Confusion matrix : ')
print(confusion_matrix(y_test,y_pred))
```


#####  Interpret of the pipeline and its decisions with SHAP. The visualizations provided will be for local explanations.

(train set is big only we use 500 as a sample)
```

shap_kernel = ShapKernel(predict_fn=blackbox_model.predict_proba, data=shap.sample(X_train,500))
shap_local = shap_kernel.explain_local(X_test[:20], y_test[:20])
show(shap_local)

```

#### Example 4: Explaining [Ray](https://docs.ray.io/en/latest/index.html) [Scikit-Learn API (tune.sklearn)](https://docs.ray.io/en/latest/tune/api_docs/sklearn.html)

##### imports
```
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from lohrasb.best_estimator import BaseModel
from sklearn.pipeline import Pipeline
from feature_engine.imputation import (
    CategoricalImputer,
    MeanMedianImputer
    )
from category_encoders import OrdinalEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score)
from sklearn.metrics import f1_score, make_scorer
from xgboost import *
from lohrasb import logger

logger.info("Using Scikit-Learn API (tune-sklearn) for an example of classification")
```
##### Example: Use [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/Adult) (a classification problem)

```
urldata= "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
# column names
col_names=["age", "workclass", "fnlwgt" , "education" ,"education-num",
"marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week",
"native-country","label"
]
# read data
data = pd.read_csv(urldata,header=None,names=col_names,sep=',')
data.head()
```
##### Define labels
```
data.loc[data['label']=='<=50K','label']=0
data.loc[data['label']==' <=50K','label']=0

data.loc[data['label']=='>50K','label']=1
data.loc[data['label']==' >50K','label']=1

data['label']=data['label'].astype(int)
```
##### Train test split
```
X = data.loc[:, data.columns != "label"]
y = data.loc[:, data.columns == "label"]


X_train, X_test, y_train, y_test =train_test_split(X, y, \
test_size=0.33, stratify=y['label'], random_state=42)
```
##### Find feature types for later use
```
int_cols =  X_train.select_dtypes(include=['int']).columns.tolist()
float_cols =  X_train.select_dtypes(include=['float']).columns.tolist()
cat_cols =  X_train.select_dtypes(include=['object']).columns.tolist()
```
##### Define estimator and set its arguments 
```
estimator = XGBClassifier()
estimator_params = {
        "booster": ["gbtree","dart"],
        "eval_metric": ["auc"],
        "max_depth": [4, 5],
        "gamma": [0.1, 1.2],
        "subsample": [0.8],

    }


obj = BaseModel().optimize_by_tunesearchcv(
            estimator=estimator,
            estimator_params=estimator_params,
            fit_params = None,
            measure_of_accuracy=make_scorer(f1_score, greater_is_better=True),
            verbose=3,
            random_state=44,
            n_jobs=None,
            cv=KFold(3),
            early_stopping=None, 
            n_trials=10,
            scoring=None, 
            refit=True, 
            error_score='raise', 
            return_train_score=False, 
            local_dir='~/ray_results', 
            name=None, 
            max_iters=1, 
            search_optimization='hyperopt',
            use_gpu=False, 
            loggers=None, 
            pipeline_auto_early_stop=True, 
            stopper=None, 
            time_budget_s=None, 
            mode=None,
            search_kwargs=None, 

        )
```
##### Build sklearn pipeline


```
pipeline =Pipeline([
            # int missing values imputers
            ('intimputer', MeanMedianImputer(
                imputation_method='median', variables=int_cols)),
            # category missing values imputers
            ('catimputer', CategoricalImputer(variables=cat_cols)),
            #
            ('catencoder', OrdinalEncoder()),
            # classification model
            ('obj', obj)

 ])
```
##### Run Pipeline
```
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
```
##### Check performance of the pipeline
```
print('F1 score : ')
print(f1_score(y_test,y_pred))
print('Classification report : ')
print(classification_report(y_test,y_pred))
print('Confusion matrix : ')
print(confusion_matrix(y_test,y_pred))
```


There are some more examples  available in the [examples](https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb/examples) webpage. 

#### License
Licensed under the [BSD 2-Clause](https://opensource.org/licenses/BSD-2-Clause) License.