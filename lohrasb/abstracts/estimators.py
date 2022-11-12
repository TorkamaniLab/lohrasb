from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator


class AbstractEstimator(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def fit(*args, **kwargs):
        pass

    @abstractmethod
    def predict(*args, **kwargs):
        pass

    def predict_proba(*args, **kwargs):
        pass
