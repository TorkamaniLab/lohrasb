# https://stackoverflow.com/questions/10724495/getting-all-arguments-and-values-passed-to-a-function
# from abc import ABC
# import abc
# import xgboost
# import logging
# class BaseModelParamsValidotr(metaclass=abc.ABCMeta):
#     @abc.abstractmethod
#     def validate_model_params(self):pass

# class ModelParamsValidotr(BaseModelParamsValidotr):
#     def __init__(self,model,params):
#         self.model=model
#         self.params = params
#     def validate_model_params(self):
#         if self.model(**self.params):
#             return self.params
#         else:
#             raise ValueError('The {print(model)} can not be set with {self.params} ')

# class BaseModelOptimizor(metaclass=abc.ABCMeta):
#     @abc.abstractmethod
#     def model_optimizor(self):pass
#     def best_estimator(self):pass
#     def best_params(self):pass


# class GridSearchCVOptimizor(BaseModelOptimizor):
#     def __init__(self,optimizor_engine,optimizor_params,model,model_params,X,y):
#         self.optimizor_engine=optimizor_engine
#         self.optimizor_params=optimizor_params
#         self.model=model
#         self.model_params=model_params
#         self.X=X
#         self.y=y
#         optimizor=Optimizor()
#         self.best_model = None
#     def model_optimizor(self):
#         optimizor.optimizor_engine=self.optimizor_engine
#         optimizor.optimizor_params=self.optimizor_params
#         optimizor.model=self.model
#         optimizor.model_params=self.model_params
#         best_model = optimizor(X,y)
#         return best_model
