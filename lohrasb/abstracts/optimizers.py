from abc import ABCMeta, abstractmethod


class OptimizerABC(metaclass=ABCMeta):
    """Internal function for returning best estimator using
    assigned parameters by Search CV, e.g., GridSearchCV, RandomizedSearchCV, etc."""

    def __init__(self, *args, **kwargs):

        """
        Class initalizer
        """
        pass

    def prepare_data(self):
        pass

    def optimize(self, *args, **kwargs):
        """
        Optimize estimator using params
        """
        pass

    def get_best_estimator(self, *args, **kwargs):
        """
        Return a best_estimator, aproduct of Search CV.

        """
        pass

    def get_optimized_object(self, *args, **kwargs):
        """
        Return whole object, a product of Search CV .

        """
        pass
