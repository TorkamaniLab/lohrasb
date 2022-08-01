# lohrasb

lohrasb is a package built to ease machine learning development. It uses [Optuna](https://optuna.readthedocs.io/en/stable/index.html) to tune most of the tree-based estimators of sickit-learn. It is compatible with [scikit-learn](https://scikit-learn.org) pipeline.


## Introduction

BaseModel of lohrasb package can receive various parameters. From a tree-based estimator class to its tunning parameters and from Grid search, Random Search, or [Optuna](https://optuna.readthedocs.io/en/stable/index.html)  to their parameters. Samples will be split to train and validation set, and then optimization will estimate optimal related parameters.

## Installation

lohrasb package is available on PyPI and can be installed with pip:

```sh
pip install lohrasb
```


## Supported estimators for this package

- XGBRegressor  [XGBoost](https://github.com/dmlc/xgboost)
- XGBClassifier [XGBoost](https://github.com/dmlc/xgboost)
- RandomForestClassifier 
- RandomForestRegressor 
- CatBoostClassifier 
- CatBoostRegressor 
- BalancedRandomForestClassifier 
- LGBMClassifier [LightGBM](https://github.com/microsoft/LightGBM)
- LGBMRegressor [LightGBM](https://github.com/microsoft/LightGBM)

## Usage

- Tunning best parameters of a tree-based model using [Optuna](https://optuna.readthedocs.io/en/stable/index.html) , [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) or [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).


## Examples 

There are some examples  available in the [examples](https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb/examples). 

### Import required libraries
```
from lohrasb.best_estimator import BaseModel
import xgboost
from optuna.pruners import HyperbandPruner
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.model_selection import KFold,train_test_split
import pandas as pd
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
```

### Use Adult Data Set (a classification problem)
```
urldata= "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
# column names
col_names=[
"age", "workclass", "fnlwgt" , "education" ,"education-num",
"marital-status","occupation","relationship","race","sex","capital-gain",
"capital-loss","hours-per-week","native-country","label"
]
data.head()
# read data
data = pd.read_csv(urldata,header=None,names=col_names,sep=',')
```
### Define labels
```
data.loc[data['label']=='<=50K','label']=0
data.loc[data['label']==' <=50K','label']=0

data.loc[data['label']=='>50K','label']=1
data.loc[data['label']==' >50K','label']=1

data['label']=data['label'].astype(int)

```

### Train test split
```
X = data.loc[:, data.columns != "label"]
y = data.loc[:, data.columns == "label"]
X_train, X_test, y_train, y_test =train_test_split(X, y, 
    test_size=0.33, stratify=y['label'], random_state=42)

```

### Find feature types for later use

```
int_cols =  X_train.select_dtypes(include=['int']).columns.tolist()
float_cols =  X_train.select_dtypes(include=['float']).columns.tolist()
cat_cols =  X_train.select_dtypes(include=['object']).columns.tolist()

```

### Define estimator and set its arguments 
```


SFC_XGBCLS_GRID = BaseModel(
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 5],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
            "booster": ["gbtree"],
        },
        hyper_parameter_optimization_method="grid",
        measure_of_accuracy="f1",
        test_size=0.33,
        cv=KFold(n_splits=3,random_state=42,shuffle=True),
        with_stratified=True,
        verbose=3,
        random_state=42,
        n_jobs=-1,
        n_iter=100,
        eval_metric="auc",
        number_of_trials=10,
        sampler=TPESampler(),
        pruner=HyperbandPruner(),

    )


```

### Build sklearn Pipeline  
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
            ('xgboost', SFC_XGBCLS_GRID)


 ])

```
### Run Pipeline  

```
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
```

## License
Licensed under the [BSD 2-Clause](https://opensource.org/licenses/BSD-2-Clause) License.