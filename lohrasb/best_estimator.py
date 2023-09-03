from abc import ABCMeta
from pickletools import optimize

from sklearn.base import BaseEstimator

from lohrasb import logger
from lohrasb.abstracts.estimators import AbstractEstimator
from lohrasb.base_classes.optimizer_bases import (
    GridSearch,
    NewOptunaSearch,
    OptunaSearch,
    RandomSearch,
    TuneCV,
    TuneGridSearch,
    TuneSearch,
)


class OptunaBestEstimator(AbstractEstimator):
    """
    BestModel estimation using Optuna optimization.

    This class provides functionality for estimating the best model using Optuna optimization.
    It includes methods for fitting the estimator, making predictions, and accessing the best estimator and optimized object.
    """

    def __init__(self, *args, **kwargs):
        self.best_estimator = None
        self.optimized_object = None
        self.args = args
        self.kwargs = kwargs

    def optimize(self, X, y, *args, **kwargs):
        """
        Fit the estimator using the best parameters extracted from optimization methods.
        """
        self.optimized_object = (
            OptunaSearch(X, y, *self.args, **self.kwargs).prepare_data().optimize()
        )

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the estimator using the best parameters extracted from optimization methods.
        """
        self.optimize(X, y, *args, **kwargs)
        self.optimized_object.fit(X, y, *args, **kwargs)
        self.best_estimator = self.optimized_object.get_best_estimator()

    def get_optimized_object(self):
        """
        Get the Optuna trial object after optimization.
        """
        return self.optimized_object.get_optimized_object()

    def predict(self, X):
        """
        Predict using the best estimator model.
        """
        return self.best_estimator.predict(X)

    def get_best_estimator(self):
        """
        Return the best estimator if the model is already fitted.
        """
        return self.best_estimator

    def predict_proba(self, X):
        """
        Predict class probabilities using the best estimator model.
        """
        try:
            return self.best_estimator.predict_proba(X)
        except AttributeError as e:
            raise ValueError(
                "The selected estimator does not have the predict_proba method."
            ) from e


class GridBestEstimator(AbstractEstimator):
    """
    BestModel estimation using GridSearchCV optimization.

    This class provides functionality for estimating the best model using GridSearchCV optimization.
    It includes methods for fitting the estimator, making predictions, and accessing the best estimator and optimized object.
    """

    def __init__(self, *args, **kwargs):
        self.best_estimator = None
        self.optimized_object = None
        self.args = args
        self.kwargs = kwargs

    def optimize(self, X, y, *args, **kwargs):
        """
        Fit the estimator using the best parameters extracted from optimization methods.
        """
        self.optimized_object = GridSearch(X, y, *self.args, **self.kwargs).optimize()

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the estimator using the best parameters extracted from optimization methods.
        """
        self.optimize(X, y, *args, **kwargs)
        self.optimized_object.fit(X, y, *args, **kwargs)
        self.best_estimator = self.optimized_object.get_best_estimator()

    def get_optimized_object(self):
        """
        Get the GridSearchCV object after optimization.
        """
        return self.optimized_object.get_optimized_object()

    def predict(self, X):
        """
        Predict using the best estimator model.
        """
        return self.best_estimator.predict(X)

    def get_best_estimator(self):
        """
        Return the best estimator if the model is already fitted.
        """
        return self.best_estimator

    def predict_proba(self, X):
        """
        Predict class probabilities using the best estimator model.
        """
        try:
            return self.best_estimator.predict_proba(X)
        except AttributeError as e:
            raise ValueError(
                "The selected estimator does not have the predict_proba method."
            ) from e


class NewOptunaBestEstimator(AbstractEstimator):
    """
    BestModel estimation using OptunaSearchCV optimization.

    This class provides functionality for estimating the best model using OptunaSearchCV optimization.
    It includes methods for fitting the estimator, making predictions, and accessing the best estimator and optimized object.
    """

    def __init__(self, *args, **kwargs):
        self.best_estimator = None
        self.optimized_object = None
        self.args = args
        self.kwargs = kwargs

    def optimize(self, X, y, *args, **kwargs):
        """
        Fit the feature selection estimator using the best parameters extracted from optimization methods.
        """
        self.optimized_object = NewOptunaSearch(
            X, y, *self.args, **self.kwargs
        ).optimize()

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the feature selection estimator using the best parameters extracted from optimization methods.
        """
        self.optimize(X, y, *args, **kwargs)
        self.optimized_object.fit(X, y, *args, **kwargs)
        self.best_estimator = self.optimized_object.get_best_estimator()

    def get_optimized_object(self):
        """
        Get the OptunaSearchCV object after optimization.
        """
        return self.optimized_object.get_optimized_object()

    def predict(self, X):
        """
        Predict using the best estimator model.
        """
        return self.best_estimator.predict(X)

    def get_best_estimator(self):
        """
        Return the best estimator if the model is already fitted.
        """
        return self.best_estimator

    def predict_proba(self, X):
        """
        Predict class probabilities using the best estimator model.
        """
        try:
            return self.best_estimator.predict_proba(X)
        except AttributeError as e:
            raise ValueError(
                "The selected estimator does not have the predict_proba method."
            ) from e


class RandomBestEstimator(AbstractEstimator):
    """
    BestModel estimation using RandomizedSearchCV optimization.

    This class provides functionality for estimating the best model using RandomizedSearchCV optimization.
    It includes methods for fitting the estimator, making predictions, and accessing the best estimator and optimized object.
    """

    def __init__(self, *args, **kwargs):
        self.best_estimator = None
        self.optimized_object = None
        self.args = args
        self.kwargs = kwargs

    def optimize(self, X, y, *args, **kwargs):
        """
        Fit the feature selection estimator using the best parameters extracted from optimization methods.
        """
        self.optimized_object = RandomSearch(X, y, *self.args, **self.kwargs).optimize()

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the feature selection estimator using the best parameters extracted from optimization methods.
        """
        self.optimize(X, y, *args, **kwargs)
        self.optimized_object.fit(X, y, *args, **kwargs)
        self.best_estimator = self.optimized_object.get_best_estimator()

    def get_optimized_object(self):
        """
        Get the RandomizedSearchCV object after optimization.
        """
        return self.optimized_object.get_optimized_object()

    def predict(self, X):
        """
        Predict using the best estimator model.
        """
        return self.best_estimator.predict(X)

    def get_best_estimator(self):
        """
        Return the best estimator if the model is already fitted.
        """
        return self.best_estimator

    def predict_proba(self, X):
        """
        Predict class probabilities using the best estimator model.
        """
        try:
            return self.best_estimator.predict_proba(X)
        except AttributeError as e:
            raise ValueError(
                "The selected estimator does not have the predict_proba method."
            ) from e


class TuneBestEstimator(AbstractEstimator):

    """
    BestModel estimation using Ray Tune optimization.

    This class provides functionality for estimating the best model using Ray Tune optimization.
    It includes methods for fitting the estimator, making predictions, and accessing the best estimator and optimized object.

    Parameters
    ----------
    estimator : object
        An unfitted estimator that has fit and predict methods.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.

    Methods
    -------
    fit(X, y, *args, **kwargs)
        Fit the feature selection estimator using the best parameters extracted from optimization methods.
    predict(X)
        Predict using the best estimator model.
    predict_proba(X)
        Predict class probabilities using the best estimator model.
    get_best_estimator()
        Return the best estimator if already fitted.
    get_optimized_object()
        Get the TuneCV object after optimization.

    Notes
    -----
    It is recommended to use available factories to create a new instance of this class.
    """

    def __init__(self, *args, **kwargs):
        self.best_estimator = None
        self.optimized_object = None
        self.args = args
        self.kwargs = kwargs

    def optimize(self, X, y, *args, **kwargs):
        """
        Fit the feature selection estimator using the best parameters extracted from optimization methods.

        Parameters
        ----------
        X : array-like or pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection step of the pipeline.
        y : array-like or pandas DataFrame or pandas Series
            Training targets. Must fulfill label requirements of the feature selection step of the pipeline.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        self.optimized_object = TuneCV(X, y, *self.args, **self.kwargs).optimize()

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the feature selection estimator using the best parameters extracted from optimization methods.

        Parameters
        ----------
        X : array-like or pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection step of the pipeline.
        y : array-like or pandas DataFrame or pandas Series
            Training targets. Must fulfill label requirements of the feature selection step of the pipeline.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        self.optimize(X, y, *args, **kwargs)
        self.optimized_object.fit(X, y, *args, **kwargs)
        self.best_estimator = self.optimized_object.get_best_estimator()

    def get_optimized_object(self):
        """
        Get the TuneCV object after optimization.

        Returns
        -------
        TuneCV or None
            The TuneCV object if available, otherwise None.
        """
        return self.optimized_object.get_optimized_object()

    def predict(self, X):
        """
        Predict using the best estimator model.

        Parameters
        ----------
        X : array-like or pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection step of the pipeline.

        Returns
        -------
        array-like
            Predicted values.
        """
        return self.best_estimator.predict(X)

    def get_best_estimator(self):
        """
        Return the best estimator if the model is already fitted.

        Returns
        -------
        BaseEstimator or None
            The best estimator if available, otherwise None.
        """
        return self.best_estimator

    def predict_proba(self, X):
        """
        Predict class probabilities using the best estimator model.

        Parameters
        ----------
        X : array-like or pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection step of the pipeline.

        Returns
        -------
        array-like
            Class probabilities.

        Raises
        ------
        ValueError
            If the selected estimator does not have the predict_proba method.
        """
        try:
            return self.best_estimator.predict_proba(X)
        except AttributeError as e:
            raise ValueError(
                f"The selected estimator does not have the predict_proba method: {e}"
            )


class TuneGridBestEstimator(AbstractEstimator):
    """
    BestModel estimation using TuneGridSearchCV optimization.

    This class provides functionality for estimating the best model using TuneGridSearchCV optimization.
    It includes methods for fitting the estimator, making predictions, and accessing the best estimator and optimized object.
    """

    def __init__(self, *args, **kwargs):
        self.best_estimator = None
        self.optimized_object = None
        self.args = args
        self.kwargs = kwargs

    def optimize(self, X, y, *args, **kwargs):
        """
        Fit the estimator using the best parameters extracted from optimization methods.
        """
        self.optimized_object = TuneGridSearch(
            X, y, *self.args, **self.kwargs
        ).optimize()

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the estimator using the best parameters extracted from optimization methods.
        """
        self.optimize(X, y, *args, **kwargs)
        self.optimized_object.fit(X, y, *args, **kwargs)
        self.best_estimator = self.optimized_object.get_best_estimator()

    def get_optimized_object(self):
        """
        Get the TuneGridSearchCV object after optimization.
        """
        return self.optimized_object.get_optimized_object()

    def predict(self, X):
        """
        Predict using the best estimator model.
        """
        return self.best_estimator.predict(X)

    def get_best_estimator(self):
        """
        Return the best estimator if the model is already fitted.
        """
        return self.best_estimator

    def predict_proba(self, X):
        """
        Predict class probabilities using the best estimator model.
        """
        try:
            return self.best_estimator.predict_proba(X)
        except AttributeError as e:
            raise ValueError(
                "The selected estimator does not have the predict_proba method."
            ) from e


class TuneSearchBestEstimator(AbstractEstimator):
    """
    BestModel estimation using Tune optimization.

    This class provides functionality for estimating the best model using Tune optimization.
    It includes methods for fitting the estimator, making predictions, and accessing the best estimator and optimized object.
    """

    def __init__(self, *args, **kwargs):
        self.best_estimator = None
        self.optimized_object = None
        self.args = args
        self.kwargs = kwargs

    def optimize(self, X, y, *args, **kwargs):
        """
        Fit the estimator using the best parameters extracted from optimization methods.
        """
        self.optimized_object = TuneSearch(X, y, *self.args, **self.kwargs).optimize()

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the estimator using the best parameters extracted from optimization methods.
        """
        self.optimize(X, y, *args, **kwargs)
        self.optimized_object.fit(X, y, *args, **kwargs)
        self.best_estimator = self.optimized_object.get_best_estimator()

    def get_optimized_object(self):
        """
        Get the TuneSearchCV object after optimization.
        """
        return self.optimized_object.get_optimized_object()

    def predict(self, X):
        """
        Predict using the best estimator model.
        """
        return self.best_estimator.predict(X)

    def get_best_estimator(self):
        """
        Return the best estimator if the model is already fitted.
        """
        return self.best_estimator

    def predict_proba(self, X):
        """
        Predict class probabilities using the best estimator model.
        """
        try:
            return self.best_estimator.predict_proba(X)
        except AttributeError as e:
            raise ValueError(
                "The selected estimator does not have the predict_proba method."
            ) from e


class BaseModel(BaseEstimator, metaclass=ABCMeta):

    """
    A base class with a variety of tools for hyperparameter optimization capabilities.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments for the method.

    Methods
    -------
    fit(X, y)
        Fit the feature selection estimator using the best parameters extracted from optimization methods.
    predict(X)
        Predict using the best estimator model.
    get_best_estimator()
        Return the best estimator if already fitted.

    optimize_by_gridsearchcv(cls, *args, **kwargs):
    optimize_by_optunasearchcv(cls, *args, **kwargs):
    optimize_by_randomsearchcv(cls, *args, **kwargs):
    optimize_by_tunesearchcv(cls, *args, **kwargs):
    optimize_by_optuna(cls, *args, **kwargs):
    optimize_by_tune(cls, *args, **kwargs):
    Notes
    -----
    It is recommended to use available factories to create a new instance of this class.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def optimize_by_gridsearchcv(cls, *args, **kwargs):
        """
        Optimize hyperparameters using GridSearchCV.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.

        **kwargs : dict

            grid_search_kwargs
                All arguments for creating a study using GridSearch. Read at [GridSeachCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
                For example:
                - estimator : object
                    An unfitted estimator that has `fit` and `predict` methods.

                - param_grid : dict
                    Parameters that were passed to find the best estimator using the optimization method.

                - measure_of_accuracy : str
                    Measurement of performance for classification and regression estimator during hyperparameter optimization while estimating the best estimator.

            main_grid_kwargs : dict
                Some required arguments for creating `BaseModel`:

            fit_grid_kwargs : dict
                All additional parameters to the `fit()` method of the estimator during training, except `X` and `y`.

        Returns
        -------
        GridBestEstimator
            GridBestEstimator instance.
        """
        # Ensure grid_search_kwargs, main_grid_kwargs are provided and are dictionaries
        required_kwargs = ["grid_search_kwargs"]
        for kwarg in required_kwargs:
            if kwarg not in kwargs["kwargs"]:
                raise ValueError(f"Missing required keyword argument: {kwarg}")
            if not isinstance(kwargs["kwargs"][kwarg], dict):
                raise TypeError(f"Expected a dictionary for keyword argument: {kwarg}")

        # fit_grid_kwargs should be a dictionary, if provided
        if "fit_grid_kwargs" in kwargs["kwargs"] and not isinstance(
            kwargs["kwargs"]["fit_grid_kwargs"], dict
        ):
            raise TypeError(
                "Expected a dictionary for keyword argument: fit_grid_kwargs"
            )

        # Ensure the 'estimator' key exists within the grid_search_kwargs and is an estimator
        if "estimator" not in kwargs["kwargs"]["grid_search_kwargs"]:
            raise ValueError(
                "The 'estimator' key must exist within the grid_search_kwargs."
            )
        if not hasattr(
            kwargs["kwargs"]["grid_search_kwargs"]["estimator"], "fit"
        ) or not hasattr(
            kwargs["kwargs"]["grid_search_kwargs"]["estimator"], "predict"
        ):
            raise TypeError(
                "The 'estimator' must have both 'fit' and 'predict' methods."
            )

        # Ensure the 'param_grid' key exists within the grid_search_kwargs and is a dictionary
        if "param_grid" not in kwargs["kwargs"]["grid_search_kwargs"]:
            raise ValueError(
                "The 'param_grid' key must exist within the grid_search_kwargs."
            )
        if not isinstance(kwargs["kwargs"]["grid_search_kwargs"]["param_grid"], dict):
            raise TypeError("'param_grid' must be a dictionary of parameters to tune.")

        return GridBestEstimator(**kwargs)

    @classmethod
    def optimize_by_optunasearchcv(cls, *args, **kwargs):
        """
        Optimize hyperparameters using OptunaSearchCV.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Arbitrarily keyworded arguments.
                'kwargs' : dict
                    Additional keyword arguments including:
                    - newoptuna_search_kwargs : dict
                        Arguments for configuring OptunaSearchCV, e.g., estimator, CV, etc.
                        See  https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.OptunaSearchCV.html
                    - fit_newoptuna_kwargs : dict
                        Parameters for the `fit` method.
                    - main_newoptuna_kwargs : dict
                        Other additional parameters.

        Returns
        -------
        NewOptunaBestEstimator
            Instance of NewOptunaBestEstimator configured with provided parameters.
        """

        # Ensure kwargs is provided and is a dictionary
        if "kwargs" not in kwargs or not isinstance(kwargs["kwargs"], dict):
            raise TypeError("The 'kwargs' key must exist and must be a dictionary.")

        # Ensure newoptuna_search_kwargs, main_newoptuna_kwargs are provided and are dictionaries
        required_kwargs = ["newoptuna_search_kwargs", "main_newoptuna_kwargs"]
        for kwarg in required_kwargs:
            if kwarg not in kwargs["kwargs"]:
                raise ValueError(f"Missing required keyword argument: {kwarg}")
            if not isinstance(kwargs["kwargs"][kwarg], dict):
                raise TypeError(f"Expected a dictionary for keyword argument: {kwarg}")

        # fit_newoptuna_kwargs should be a dictionary, if provided
        if "fit_newoptuna_kwargs" in kwargs["kwargs"] and not isinstance(
            kwargs["kwargs"]["fit_newoptuna_kwargs"], dict
        ):
            raise TypeError(
                "Expected a dictionary for keyword argument: fit_newoptuna_kwargs"
            )

        # Ensure the 'estimator' key exists within the newoptuna_search_kwargs and is an estimator
        if "estimator" not in kwargs["kwargs"]["newoptuna_search_kwargs"]:
            raise ValueError(
                "The 'estimator' key must exist within the newoptuna_search_kwargs."
            )
        if not hasattr(
            kwargs["kwargs"]["newoptuna_search_kwargs"]["estimator"], "fit"
        ) or not hasattr(
            kwargs["kwargs"]["newoptuna_search_kwargs"]["estimator"], "predict"
        ):
            raise TypeError(
                "The 'estimator' within newoptuna_search_kwargs must have both 'fit' and 'predict' methods."
            )

        # Additional checks can be added as necessary...

        return NewOptunaBestEstimator(**kwargs)

    @classmethod
    def optimize_by_randomsearchcv(cls, *args, **kwargs):
        """
        Optimize hyperparameters using RandomSearchCV.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Arbitrarily keyworded arguments.
                'kwargs' : dict
                    Additional keyword arguments including:
                    - random_search_kwargs : dict
                        Arguments for configuring RandomSearchCV, e.g., estimator, param_distributions, scoring etc.
                        See  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
                    - fit_random_kwargs : dict
                        Parameters for the `fit` method.
                    - main_random_kwargs : dict
                        Other additional parameters.

        Returns
        -------
        RandomBestEstimator
            Instance of RandomBestEstimator configured with provided parameters.
        """

        # Ensure kwargs is provided and is a dictionary
        if "kwargs" not in kwargs or not isinstance(kwargs["kwargs"], dict):
            raise TypeError("The 'kwargs' key must exist and must be a dictionary.")

        # Ensure random_search_kwargs, main_random_kwargs are provided and are dictionaries
        required_kwargs = ["random_search_kwargs", "main_random_kwargs"]
        for kwarg in required_kwargs:
            if kwarg not in kwargs["kwargs"]:
                raise ValueError(f"Missing required keyword argument: {kwarg}")
            if not isinstance(kwargs["kwargs"][kwarg], dict):
                raise TypeError(f"Expected a dictionary for keyword argument: {kwarg}")

        # fit_random_kwargs should be a dictionary, if provided
        if "fit_random_kwargs" in kwargs["kwargs"] and not isinstance(
            kwargs["kwargs"]["fit_random_kwargs"], dict
        ):
            raise TypeError(
                "Expected a dictionary for keyword argument: fit_random_kwargs"
            )

        # Ensure the 'estimator' key exists within the random_search_kwargs and is an estimator
        if "estimator" not in kwargs["kwargs"]["random_search_kwargs"]:
            raise ValueError(
                "The 'estimator' key must exist within the random_search_kwargs."
            )
        if not hasattr(
            kwargs["kwargs"]["random_search_kwargs"]["estimator"], "fit"
        ) or not hasattr(
            kwargs["kwargs"]["random_search_kwargs"]["estimator"], "predict"
        ):
            raise TypeError(
                "The 'estimator' within random_search_kwargs must have both 'fit' and 'predict' methods."
            )

        # Additional checks can be added as necessary...

        return RandomBestEstimator(**kwargs)

    @classmethod
    def optimize_by_tunegridsearchcv(cls, *args, **kwargs):
        """
        Optimize hyperparameters using TuneGridSearchCV.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Arbitrarily keyworded arguments.
                'kwargs' : dict
                    Additional keyword arguments including:
                    - tunegrid_search_kwargs : dict
                        Arguments for configuring TuneGridSearchCV, e.g., estimator, param_grid, scoring etc.
                        See https://docs.ray.io/en/latest/tune/api/sklearn.html
                    - fit_tunegrid_kwargs : dict
                        Parameters for the `fit` method.
                    - main_tunegrid_kwargs : dict
                        Other additional parameters.

        Returns
        -------
        TuneGridBestEstimator
            Instance of TuneGridBestEstimator configured with provided parameters.
        """

        # Ensure kwargs is provided and is a dictionary
        if "kwargs" not in kwargs or not isinstance(kwargs["kwargs"], dict):
            raise TypeError("The 'kwargs' key must exist and must be a dictionary.")

        # Ensure tunegrid_search_kwargs, main_tunegrid_kwargs are provided and are dictionaries
        required_kwargs = ["tunegrid_search_kwargs", "main_tunegrid_kwargs"]
        for kwarg in required_kwargs:
            if kwarg not in kwargs["kwargs"]:
                raise ValueError(f"Missing required keyword argument: {kwarg}")
            if not isinstance(kwargs["kwargs"][kwarg], dict):
                raise TypeError(f"Expected a dictionary for keyword argument: {kwarg}")

        # fit_tune_kwargs should be a dictionary, if provided
        if "fit_tunegrid_kwargs" in kwargs["kwargs"] and not isinstance(
            kwargs["kwargs"]["fit_tunegrid_kwargs"], dict
        ):
            raise TypeError(
                "Expected a dictionary for keyword argument: fit_tunegrid_kwargs"
            )

        # Ensure the 'estimator' key exists within the tunegrid_search_kwargs and is an estimator
        if "estimator" not in kwargs["kwargs"]["tunegrid_search_kwargs"]:
            raise ValueError(
                "The 'estimator' key must exist within the tunegrid_search_kwargs."
            )
        if not hasattr(
            kwargs["kwargs"]["tunegrid_search_kwargs"]["estimator"], "fit"
        ) or not hasattr(
            kwargs["kwargs"]["tunegrid_search_kwargs"]["estimator"], "predict"
        ):
            raise TypeError(
                "The 'estimator' within tunegrid_search_kwargs must have both 'fit' and 'predict' methods."
            )

        # Additional checks can be added as necessary...

        return TuneGridBestEstimator(**kwargs)

    @classmethod
    def optimize_by_tunesearchcv(cls, *args, **kwargs):
        """
        Optimize hyperparameters using TuneSearchCV.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Arbitrarily keyworded arguments.
                'kwargs' : dict
                    Additional keyword arguments including:
                    - tune_search_kwargs : dict
                        Arguments for configuring TuneSearchCV, e.g., estimator, param_distributions, scoring etc.
                        See https://docs.ray.io/en/latest/tune/api/sklearn.html
                    - fit_tune_kwargs : dict
                        Parameters for the `fit` method.
                    - main_tune_kwargs : dict
                        Other additional parameters.

        Returns
        -------
        TuneSearchBestEstimator
            Instance of TuneSearchBestEstimator configured with provided parameters.
        """

        # Ensure kwargs is provided and is a dictionary
        if "kwargs" not in kwargs or not isinstance(kwargs["kwargs"], dict):
            raise TypeError("The 'kwargs' key must exist and must be a dictionary.")

        # Ensure tune_search_kwargs, main_tune_kwargs are provided and are dictionaries
        required_kwargs = ["tune_search_kwargs", "main_tune_kwargs"]
        for kwarg in required_kwargs:
            if kwarg not in kwargs["kwargs"]:
                raise ValueError(f"Missing required keyword argument: {kwarg}")
            if not isinstance(kwargs["kwargs"][kwarg], dict):
                raise TypeError(f"Expected a dictionary for keyword argument: {kwarg}")

        # fit_tune_kwargs should be a dictionary, if provided
        if "fit_tune_kwargs" in kwargs["kwargs"] and not isinstance(
            kwargs["kwargs"]["fit_tune_kwargs"], dict
        ):
            raise TypeError(
                "Expected a dictionary for keyword argument: fit_tune_kwargs"
            )

        # Ensure the 'estimator' key exists within the tune_search_kwargs and is an estimator
        if "estimator" not in kwargs["kwargs"]["tune_search_kwargs"]:
            raise ValueError(
                "The 'estimator' key must exist within the tune_search_kwargs."
            )
        if not hasattr(
            kwargs["kwargs"]["tune_search_kwargs"]["estimator"], "fit"
        ) or not hasattr(
            kwargs["kwargs"]["tune_search_kwargs"]["estimator"], "predict"
        ):
            raise TypeError(
                "The 'estimator' within tune_search_kwargs must have both 'fit' and 'predict' methods."
            )

        # Additional checks can be added as necessary...

        return TuneSearchBestEstimator(**kwargs)

    @classmethod
    def optimize_by_optuna(cls, *args, **kwargs):
        """
        Optimize hyperparameters using Optuna.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Arbitrarily keyworded arguments.
                'kwargs' : dict
                    Additional keyword arguments including:
                    - study_search_kwargs : dict
                        Arguments for creating a study using Optuna.
                        See https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html
                    - main_optuna_kwargs : dict
                        Required arguments for `BaseModel`, e.g., estimator, estimator_params, measure_of_accuracy.
                    - optimize_kwargs : dict
                        Arguments for `optimize` method of Optuna, except `objective`.
                        See https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
                    - train_test_split_kwargs : dict
                        Arguments for `train_test_split` function, such as `cv`, `random_state`, etc. except the *array.
                        See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
                    - fit_optuna_kwargs : dict
                        Additional parameters for the `fit` method of the estimator, except `X` and `y`.

        Returns
        -------
        OptunaBestEstimator
            Instance of OptunaBestEstimator configured with provided parameters.
        """
        if "kwargs" not in kwargs or not isinstance(kwargs["kwargs"], dict):
            raise TypeError("The 'kwargs' key must exist and must be a dictionary.")

        required_kwargs = [
            "study_search_kwargs",
            "main_optuna_kwargs",
            "optimize_kwargs",
            "train_test_split_kwargs",
        ]
        for kwarg in required_kwargs:
            if kwarg not in kwargs["kwargs"]:
                raise ValueError(f"Missing required keyword argument: {kwarg}")
            if not isinstance(kwargs["kwargs"][kwarg], dict):
                raise TypeError(f"Expected a dictionary for keyword argument: {kwarg}")

        if "fit_optuna_kwargs" in kwargs["kwargs"] and not isinstance(
            kwargs["kwargs"]["fit_optuna_kwargs"], dict
        ):
            raise TypeError(
                "Expected a dictionary for keyword argument: fit_optuna_kwargs"
            )

        if "estimator" not in kwargs["kwargs"]["main_optuna_kwargs"]:
            raise ValueError(
                "The 'estimator' key must exist within the main_optuna_kwargs."
            )
        if not hasattr(
            kwargs["kwargs"]["main_optuna_kwargs"]["estimator"], "fit"
        ) or not hasattr(
            kwargs["kwargs"]["main_optuna_kwargs"]["estimator"], "predict"
        ):
            raise TypeError(
                "The 'estimator' within main_optuna_kwargs must have both 'fit' and 'predict' methods."
            )

        return OptunaBestEstimator(**kwargs)

    @classmethod
    def optimize_by_tune(cls, *args, **kwargs):
        """
        Optimize hyperparameters using Tune.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Arbitrarily keyworded arguments.
                'kwargs' : dict
                    Additional keyword arguments including:
                    - tuner_kwargs : dict
                        Arguments for the tuner from Ray.
                        See https://docs.ray.io/en/latest/tune/index.html
                    - main_tune_kwargs : dict
                        Arguments for the main tuning process including `cv`, `scoring`, `estimator`, etc.
                    - fit_tune_kwargs : dict
                        Additional parameters for the `fit` method of the estimator, except `X` and `y`.

        Returns
        -------
        TuneBestEstimator
            Instance of TuneBestEstimator configured with provided parameters.
        """
        if "kwargs" not in kwargs or not isinstance(kwargs["kwargs"], dict):
            raise TypeError("The 'kwargs' key must exist and must be a dictionary.")

        required_kwargs = ["tuner_kwargs", "main_tune_kwargs"]
        for kwarg in required_kwargs:
            if kwarg not in kwargs["kwargs"]:
                raise ValueError(f"Missing required keyword argument: {kwarg}")
            if not isinstance(kwargs["kwargs"][kwarg], dict):
                raise TypeError(f"Expected a dictionary for keyword argument: {kwarg}")

        if "fit_tune_kwargs" in kwargs["kwargs"] and not isinstance(
            kwargs["kwargs"]["fit_tune_kwargs"], dict
        ):
            raise TypeError(
                "Expected a dictionary for keyword argument: fit_tune_kwargs"
            )

        if "estimator" not in kwargs["kwargs"]["main_tune_kwargs"]:
            raise ValueError(
                "The 'estimator' key must exist within the main_tune_kwargs."
            )
        if not hasattr(
            kwargs["kwargs"]["main_tune_kwargs"]["estimator"], "fit"
        ) or not hasattr(kwargs["kwargs"]["main_tune_kwargs"]["estimator"], "predict"):
            raise TypeError(
                "The 'estimator' within main_tune_kwargs must have both 'fit' and 'predict' methods."
            )

        return TuneBestEstimator(**kwargs)

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the feature selection estimator using the best parameters extracted from optimization methods.

        Parameters
        ----------
        X : Pandas DataFrame or array-like
            Training data. Must fulfill the input requirements of the feature selection step of the pipeline.
        y : Pandas DataFrame, Pandas series, or array-like
            Training targets. Must fulfill the label requirements of the feature selection step of the pipeline.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        pass

    def predict(self, X):
        """
        Predict using the best estimator model.

        Parameters
        ----------
        X : Pandas DataFrame or array-like
            Input data. Must fulfill the input requirements of the feature selection step of the pipeline.
        """
        pass

    def get_best_estimator(self):
        """
        Return the best estimator if the model is already fitted.
        """
        pass
