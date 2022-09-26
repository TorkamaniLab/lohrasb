from abc import ABCMeta, abstractmethod


class OptimizerFactory(metaclass=ABCMeta):
    @abstractmethod
    def optimizer_builder(*args, **kargs):
        """
        Return a optimizer CV instanse, ready to use
        ...

        Attributes
        ----------
        *args: list
            A list of possible argumnets
        **kwargs: dict
            A dict of possible argumnets
        """
        pass
