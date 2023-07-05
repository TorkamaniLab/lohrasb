from abc import ABCMeta, abstractmethod


class OptimizerABC(metaclass=ABCMeta):
    """
    Abstract base class for optimizers used in Search CV.

    This class provides a framework for optimizing estimators using parameters assigned by Search CV, such as GridSearchCV, RandomizedSearchCV, etc.

    Parameters:
    *args: Variable length argument list.
    **kwargs: Arbitrary keyword arguments.

    Methods:
    --------
    prepare_data():
        Prepare the data for optimization.

    optimize(*args, **kwargs):
        Optimize the estimator using the assigned parameters.

    get_best_estimator(*args, **kwargs):
        Return the best_estimator, a product of Search CV.

    get_optimized_object(*args, **kwargs):
        Return the entire optimized object, a product of Search CV.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the OptimizerABC class.

        Parameters:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """
        pass

    @abstractmethod
    def prepare_data(self):
        """
        Prepare the data for optimization.
        """
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        """
        Optimize the estimator using the assigned parameters.

        Parameters:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """
        pass

    @abstractmethod
    def get_best_estimator(self, *args, **kwargs):
        """
        Return the best_estimator, a product of Search CV.

        Parameters:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """
        pass

    @abstractmethod
    def get_optimized_object(self, *args, **kwargs):
        """
        Return the entire optimized object, a product of Search CV.

        Parameters:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """
        pass
