# %%
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


# %% [markdown]
# #### Example: Use Adult Data Set (a classification problem)
#   
# https://archive.ics.uci.edu/ml/datasets/Adult

# %% [markdown]
# #### Part 1: Use BestModel in sklearn pipeline
# 

# %%
urldata= "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
# column names
col_names=["age", "workclass", "fnlwgt" , "education" ,"education-num",
"marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week",
"native-country","label"
]
# read data
data = pd.read_csv(urldata,header=None,names=col_names,sep=',')
data.head()

# %% [markdown]
# #### Define labels
# 

# %%
data.loc[data['label']=='<=50K','label']=0
data.loc[data['label']==' <=50K','label']=0

data.loc[data['label']=='>50K','label']=1
data.loc[data['label']==' >50K','label']=1

data['label']=data['label'].astype(int)

# %% [markdown]
# #### Train test split

# %%
X = data.loc[:, data.columns != "label"]
y = data.loc[:, data.columns == "label"]


X_train, X_test, y_train, y_test =train_test_split(X, y, \
     test_size=0.33, stratify=y['label'], random_state=42)


# %% [markdown]
# #### Find feature types for later use

# %%
int_cols =  X_train.select_dtypes(include=['int']).columns.tolist()
float_cols =  X_train.select_dtypes(include=['float']).columns.tolist()
cat_cols =  X_train.select_dtypes(include=['object']).columns.tolist()


# %% [markdown]
# #### Define estimator and set its arguments 
# 

# %%
estimator = XGBClassifier()
estimator_params = {
        "booster": ["gbtree","dart"],
        "eval_metric": ["auc"],
        "max_depth": [4, 5],
        "gamma": [0.1, 1.2],
        "subsample": [0.8],

    }

kwargs = {  # params for fit method or fit_params 
            'fit_tunegrid_kwargs' :{
            'sample_weight':None,
            },
            # params for RandomSearchCV 
            'tunegrid_search_kwargs' : {
            'estimator':estimator,
            'param_grid':estimator_params,
            'n_jobs':None,
            'cv':KFold(3),
            'early_stopping':None, 
            'scoring':None, 
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
                }
            }
    

# %%

obj = BaseModel().optimize_by_tunegridsearchcv(kwargs=kwargs)

# %% [markdown]
# #### Build sklearn pipeline

# %%


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


# %% [markdown]
# #### Run Pipeline

# %%
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)


# %% [markdown]
# #### Check performance of the pipeline

# %%
print('F1 score : ')
print(f1_score(y_test,y_pred))
print('Classification report : ')
print(classification_report(y_test,y_pred))
print('Confusion matrix : ')
print(confusion_matrix(y_test,y_pred))


# %% [markdown]
# #### Part 2:  Use BestModel as a standalone estimator 

# %%
X_train, X_test, y_train, y_test =train_test_split(X, y, \
     test_size=0.33, stratify=y['label'], random_state=42)

# %% [markdown]
# #### Transform features to make them ready for model input

# %%
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

# %% [markdown]
# #### Transform X_train and X_test

# %%
X_train=transform_pipeline.fit_transform(X_train,y_train)
X_test=transform_pipeline.transform(X_test)


# %% [markdown]
# #### Train model and predict

# %%
obj.fit(X_train,y_train)
y_pred = obj.predict(X_test)

# %% [markdown]
# #### Check performance of the pipeline

# %%
print('F1 score : ')
print(f1_score(y_test,y_pred))
print('Classification report : ')
print(classification_report(y_test,y_pred))
print('Confusion matrix : ')
print(confusion_matrix(y_test,y_pred))

# %%
obj.get_best_estimator()

# %%
obj.best_estimator

# %% [markdown]
# #### Get fitted grid search object and its attributes

# %%
GridSearchObj = obj.get_optimized_object()
GridSearchObj.cv_results_


