from abc import ABCMeta
from pickletools import optimize
from sklearn.base import BaseEstimator
from lohrasb import logger
from lohrasb.abstracts.estimators import AbstractEstimator
from lohrasb.base_classes.optimizer_bases import (
    GridSearch,
    OptunaSearch,
    RandomSearch,
    TuneGridSearch,
    TuneSearch,
    NewOptunaSearch,
    TuneCV,
)


class OptunaBestEstimator(AbstractEstimator):

    """
    BestModel estimation using Optuna optimization.

    This class provides functionality for estimating the best model using Optuna optimization.
    It includes methods for fitting the estimator, making predictions, and accessing the best estimator and optimized object.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.

            study_search_kwargs : dict
                All arguments for creating a study using Optuna. Read more from `study.create` for Optuna at [Optuna Documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html).

            main_optuna_kwargs : dict
                Some required arguments for creating `BaseModel`:

                - estimator : object
                    An unfitted estimator that has `fit` and `predict` methods.

                - estimator_params : dict
                    Parameters that were passed to find the best estimator using the optimization method.

                - measure_of_accuracy : str
                    Measurement of performance for classification and regression estimator during hyperparameter optimization while estimating the best estimator.

            optimize_kwargs : dict
                All arguments for the `optimize` object of Optuna, except `objective`. Check at [Optuna Study.optimize](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize).

            train_test_split_kwargs : dict
                All arguments for `train_test_split` function, such as `cv`, `random_state`, except the *array. Check at [scikit-learn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

            fit_optuna_kwargs : dict
                All additional parameters to the `fit()` method of the estimator during training, except `X` and `y`.


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
        Get the Optuna trial object after optimization.

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

        self.optimized_object = OptunaSearch(X, y, *self.args, **self.kwargs).prepare_data().optimize()
    
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
        Get the Optuna trial object after optimization.

        Returns
        -------
        OptunaTrial or None
            The Optuna trial object if available, otherwise None.
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
        except Exception as e:
            raise ValueError(f"The selected estimator does not have the predict_proba method: {e}")


class GridBestEstimator(AbstractEstimator):

    """
    BestModel estimation using GridSearchCV optimization.

    This class provides functionality for estimating the best model using GridSearchCV optimization.
    It includes methods for fitting the estimator, making predictions, and accessing the best estimator and optimized object.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments.

    kwargs : dict
        Kwargs of the class.

        grid_search_kwargs : dict

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
        Get the GridSearchCV object after optimization.

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
        self.optimized_object = GridSearch(X, y, *self.args, **self.kwargs).optimize()
    
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
        Get the GridSearchCV object after optimization.

        Returns
        -------
        GridSearchCV or None
            The GridSearchCV object if available, otherwise None.
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
        except Exception as e:
            raise ValueError(f"The selected estimator does not have the predict_proba method: {e}")


class NewOptunaBestEstimator(AbstractEstimator):

    """BestModel estimation using OptunaSearchCV optimization.
    ...

    Parameters
    ----------
    X : pd.DataFrame
        Input for training.
    y : pd.DataFrame
        Target or label.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
            newoptuna_search_kwargs: dict
                arguments for OptunaSearchCV, e.g., estimator, CV, etc. 
               see  https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.OptunaSearchCV.html
            fit_newoptuna_kwargs : dict
                fit params 
            main_newoptuna_kwargs : dict
                other parameters  
    Methods
    -------
    fit(X, y)
        Fit the feature selection estimator by the best parameters extracted
        from optimization methods.
    predict(X)
        Predict using the best estimator model.
    predict_proba(X)
        Predict class probabilities using the best estimator model.
    get_best_estimator()
        Return best estimator, if aleardy fitted.
    get_optimized_object():
        Get GridSearchCV object after optimization.
    Notes
    -----
    It is recommended to use available factories
    to create a new instance of this class.

    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.best_estimator = None
        self.optimized_object = None
        self.args = args
        self.kwargs = kwargs

    @property
    def best_estimator(self):
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, value):
        self._best_estimator = value

    @property
    def optimized_object(self):
        return self._optimized_object

    @optimized_object.setter
    def optimized_object(self, value):
        self._optimized_object = value

    @property
    def kwargs(self):
        return self._kwargs

    @kwargs.setter
    def kwargs(self, value):
        self._kwargs = value

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        self._args = value

    def optimize(self, X, y, *args, **kwargs):
        """Fit the feature selection estimator by best params extracted
        from optimization methods.
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection
            step of the pipeline.
        y : Pandas DataFrame or Pandas series
            Training targets. Must fulfill label requirements of feature selection
            step of the pipeline.
        """
        self.optimized_object = NewOptunaSearch(X, y, *self.args, **self.kwargs).\
            optimize()
    
    def fit(self, X, y, *args, **kwargs):
        self.optimize(X, y, *args, **kwargs)
        """Fit the feature selection estimator by best params extracted
        from optimization methods.
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection
            step of the pipeline.
        y : Pandas DataFrame or Pandas series
            Training targets. Must fulfill label requirements of feature selection
            step of the pipeline.
        """
        self.optimized_object.fit(X, y, *args, **kwargs)
        self.best_estimator = self.optimized_object.get_best_estimator()

    def get_optimized_object(self):
        """
        Get OptunaSearchCV  object after optimization.
        """
        return self.optimized_object.get_optimized_object()

    def predict(self, X):
        """Predict using the best estimator model.
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection
            step of the pipeline.
        """
        return self.best_estimator.predict(X)

    def get_best_estimator(self):
        """Return best estimator if model already fitted."""
        return self.best_estimator

    def predict_proba(self, X):
        """Predict using the best estimator model.
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection
            step of the pipeline.
        """
        try:
            return self.best_estimator.predict_proba(X)
        except Exception as e:
            raise ValueError(
                f"probobly the selected estimator \
                does not have predict_proba method! {e}"
            )
class NewOptunaBestEstimator(AbstractEstimator):

    """
    BestModel estimation using OptunaSearchCV optimization.

    This class provides functionality for estimating the best model using OptunaSearchCV optimization.
    It includes methods for fitting the estimator, making predictions, and accessing the best estimator and optimized object.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
            Additional keyword arguments.

            tuner_kwargs : dict, optional
                Keyword arguments for the tuner from Ray. These arguments will be passed to the tuner during initialization.
                For more information on tuner arguments, refer to the Tune documentation: https://docs.ray.io/en/latest/tune/index.html

            main_tune_kwargs : dict, optional
                Keyword arguments for the main tuning process. These arguments will be passed to the main tuning function.
                The `cv` parameter represents the cross-validation generator or an iterable used for evaluation.
                The `scoring` parameter represents the strategy to evaluate the performance of the cross-validated model on the test set.
                It can be a string, callable, list, tuple, or dictionary. Default is None.
                The `estimator` parameter represents the estimator object used for optimization. 
                For scoring check https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                and https://scikit-learn.org/stable/modules/model_evaluation.html#scoring.
                For CV check https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
                as an example.

            fit_tune_kwargs : dict, optional
                Additional keyword arguments to be passed to the `fit()` method of the estimator during training.


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
        Get the OptunaSearchCV object after optimization.

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
        self.optimized_object = NewOptunaSearch(X, y, *self.args, **self.kwargs).optimize()
    
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
        Get the OptunaSearchCV object after optimization.

        Returns
        -------
        OptunaSearchCV or None
            The OptunaSearchCV object if available, otherwise None.
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
        except Exception as e:
            raise ValueError(f"The selected estimator does not have the predict_proba method: {e}")

class RandomBestEstimator(AbstractEstimator):

    """
    BestModel estimation using RandomizedSearchCV optimization.

    This class provides functionality for estimating the best model using RandomizedSearchCV optimization.
    It includes methods for fitting the estimator, making predictions, and accessing the best estimator and optimized object.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
            random_search_kwargs : dict
                All arguments for creating a study using RandomSearch. Read at [RandomizedSeachCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
                For example:

                - estimator : object
                    An unfitted estimator that has `fit` and `predict` methods.

                - param_distributions : dict
                    Parameters that were passed to find the best estimator using the optimization method.

                - scoring : str, callable, list, tuple or dict, default=None
                    Measurement of performance for classification and regression estimator during hyperparameter optimization while estimating the best estimator.

            main_random_kwargs : dict
                Some required arguments for creating `BaseModel`:

            fit_random_kwargs : dict
                All additional parameters to the `fit()` method of the estimator during training, except `X` and `y`.


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
        Get the RandomizedSearchCV object after optimization.

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
        self.optimized_object = RandomSearch(X, y,  *self.args, **self.kwargs).optimize()

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
        Get the RandomizedSearchCV object after optimization.

        Returns
        -------
        RandomizedSearchCV or None
            The RandomizedSearchCV object if available, otherwise None.
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
        except Exception as e:
            raise ValueError(f"The selected estimator does not have the predict_proba method: {e}")

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

    def __init__(self,*args, **kwargs):
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
            raise ValueError(f"The selected estimator does not have the predict_proba method: {e}")


class TuneGridBestEstimator(AbstractEstimator):

    """
    BestModel estimation using TuneGridSearchCV optimization.

    This class provides functionality for estimating the best model using TuneGridSearchCV optimization.
    It includes methods for fitting the estimator, making predictions, and accessing the best estimator and optimized object.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        
        tunegrid_search_kwargs: dict
            All arguments for creating a study using TuneGridSeachCV. Read at [TuneGridSeachCV Documentation](https://docs.ray.io/en/latest/tune/api/sklearn.html)
            For example:
            - estimator : object
                An unfitted estimator that has `fit` and `predict` methods.

            - param_grid : dict
                Parameters that were passed to find the best estimator using the optimization method.

            - scoring : str
                Measurement of performance for classification and regression estimator during hyperparameter optimization while estimating the best estimator.

        main_tunegrid_kwargs : dict
            Some required arguments for creating `BaseModel`:

        fit_tunegrid_kwargs : dict
            All additional parameters to the `fit()` method of the estimator during training, except `X` and `y`.

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
        Get the TuneGridSearchCV object after optimization.

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
        self.optimized_object = TuneGridSearch(X, y, *self.args, **self.kwargs).optimize()

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
        Get the TuneGridSearchCV object after optimization.

        Returns
        -------
        TuneGridSearchCV or None
            The TuneGridSearchCV object if available, otherwise None.
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
            raise ValueError(f"The selected estimator does not have the predict_proba method: {e}")


class TuneSearchBestEstimator(AbstractEstimator):

    """
    BestModel estimation using Tune optimization.

    This class provides functionality for estimating the best model using Tune optimization.
    It includes methods for fitting the estimator, making predictions, and accessing the best estimator and optimized object.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
            
        tune_search_kwargs: dict
            All arguments for creating a study using TuneSearchCV. Read at [TuneSearchCV Documentation](https://docs.ray.io/en/latest/tune/api/sklearn.html)
            
            For example:
            - estimator : object
                An unfitted estimator that has `fit` and `predict` methods.

            - param_distributions : dict
                Parameters that were passed to find the best estimator using the optimization method.

            - scoring : str
                Measurement of performance for classification and regression estimator during hyperparameter optimization while estimating the best estimator.

        main_tune_kwargs : dict
            Some required arguments for creating `BaseModel`:

        fit_tune_kwargs : dict
            All additional parameters to the `fit()` method of the estimator during training, except `X` and `y`.


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
        Get the TuneSearchCV object after optimization.

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
        self.optimized_object = TuneSearch(X, y, *self.args, **self.kwargs).optimize()

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
        Get the TuneSearchCV object after optimization.

        Returns
        -------
        TuneSearchCV or None
            The TuneSearchCV object if available, otherwise None.
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
            raise ValueError(f"The selected estimator does not have the predict_proba method: {e}")


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
        gse = GridBestEstimator(**kwargs)
        return gse

    @classmethod
    def optimize_by_optunasearchcv(cls, *args, **kwargs):
        """
        Optimize hyperparameters using OptunaSearchCV.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.
        **kwargs : dict

            Additional keyword arguments.
                newoptuna_search_kwargs: dict
                    arguments for OptunaSearchCV, e.g., estimator, CV, etc. 
                see  https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.OptunaSearchCV.html
                fit_newoptuna_kwargs : dict
                    fit params 
                main_newoptuna_kwargs : dict
                    other parameters  

        Returns
        -------
        NewOptunaBestEstimator
            NewOptunaBestEstimator instance.
        """
        noe = NewOptunaBestEstimator(**kwargs)
        return noe

    @classmethod
    def optimize_by_randomsearchcv(cls, *args, **kwargs):
        """
        Optimize hyperparameters using RandomSearchCV.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.

        random_search_kwargs : dict
            Additional keyword arguments.
                random_search_kwargs : dict
                    All arguments for creating a study using RandomSearch. Read at [RandomizedSeachCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
                    For example:

                    - estimator : object
                        An unfitted estimator that has `fit` and `predict` methods.

                    - param_distributions : dict
                        Parameters that were passed to find the best estimator using the optimization method.

                    - scoring : str, callable, list, tuple or dict, default=None
                        Measurement of performance for classification and regression estimator during hyperparameter optimization while estimating the best estimator.

                main_random_kwargs : dict
                    Some required arguments for creating `BaseModel`:

                fit_random_kwargs : dict
                    All additional parameters to the `fit()` method of the estimator during training, except `X` and `y`.

        Returns
        -------
        RandomBestEstimator
            RandomBestEstimator instance.
        """
        rse = RandomBestEstimator(**kwargs)
        return rse

    @classmethod
    def optimize_by_tunegridsearchcv(cls, *args, **kwargs):
        """
        Optimize hyperparameters using TuneGridSearchCV.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.

        **tunegrid_search_kwargs: dict
            All arguments for creating a study using TuneGridSeachCV. Read at [TuneGridSeachCV Documentation](https://docs.ray.io/en/latest/tune/api/sklearn.html)
            For example:
            - estimator : object
                An unfitted estimator that has `fit` and `predict` methods.

            - param_grid : dict
                Parameters that were passed to find the best estimator using the optimization method.

            - scoring : str
                Measurement of performance for classification and regression estimator during hyperparameter optimization while estimating the best estimator.

        main_tunegrid_kwargs : dict
            Some required arguments for creating `BaseModel`:

        fit_tune_kwargs : dict
            All additional parameters to the `fit()` method of the estimator during training, except `X` and `y`.

        Returns
        -------
        TuneGridBestEstimator
            TuneGridBestEstimator instance.
        """
        tge = TuneGridBestEstimator(**kwargs)
        return tge

    @classmethod
    def optimize_by_tunesearchcv(cls, *args, **kwargs):
        """
        Optimize hyperparameters using TuneSearchCV.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Keyword arguments for TuneSearchCV.
            
            tune_search_kwargs: dict
                All arguments for creating a study using TuneSearchCV. Read at [TuneSearchCV Documentation](https://docs.ray.io/en/latest/tune/api/sklearn.html)
                
                For example:
                - estimator : object
                    An unfitted estimator that has `fit` and `predict` methods.

                - param_distributions : dict
                    Parameters that were passed to find the best estimator using the optimization method.

                - scoring : str
                    Measurement of performance for classification and regression estimator during hyperparameter optimization while estimating the best estimator.

            main_tune_kwargs : dict
                Some required arguments for creating `BaseModel`:

            fit_tune_kwargs : dict
                All additional parameters to the `fit()` method of the estimator during training, except `X` and `y`.

        Returns
        -------
        TuneSearchBestEstimator
            TuneSearchBestEstimator instance.
        """
        tse = TuneSearchBestEstimator(**kwargs)
        return tse

    @classmethod
    def optimize_by_optuna(cls, *args, **kwargs):
        """
        Optimize hyperparameters using Optuna.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.
        **optuna_search_kwargs : dict
            Keyword arguments for Optuna.
            
            study_search_kwargs : dict
                All arguments for creating a study using Optuna. Read more from `study.create` for Optuna at [Optuna Documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html).

            main_optuna_kwargs : dict
                Some required arguments for creating `BaseModel`:

                - estimator : object
                    An unfitted estimator that has `fit` and `predict` methods.

                - estimator_params : dict
                    Parameters that were passed to find the best estimator using the optimization method.

                - measure_of_accuracy : str
                    Measurement of performance for classification and regression estimator during hyperparameter optimization while estimating the best estimator.

            optimize_kwargs : dict
                All arguments for the `optimize` object of Optuna, except `objective`. Check at [Optuna Study.optimize](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize).

            train_test_split_kwargs : dict
                All arguments for `train_test_split` function, such as `cv`, `random_state`, except the *array. Check at [scikit-learn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

            fit_optuna_kwargs : dict
                All additional parameters to the `fit()` method of the estimator during training, except `X` and `y`.

        Returns
        -------
        OptunaBestEstimator
            OptunaBestEstimator instance.
        """
        obe = OptunaBestEstimator(**kwargs)
        return obe

    @classmethod
    def optimize_by_tune(cls, *args, **kwargs):
        """
        Optimize hyperparameters using Tune.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.
        **tune_search_kwargs : dict

                    Additional keyword arguments.

            tuner_kwargs : dict, optional
                Keyword arguments for the tuner from Ray. These arguments will be passed to the tuner during initialization.
                For more information on tuner arguments, refer to the Tune documentation: https://docs.ray.io/en/latest/tune/index.html

            main_tune_kwargs : dict, optional
                Keyword arguments for the main tuning process. These arguments will be passed to the main tuning function.
                The `cv` parameter represents the cross-validation generator or an iterable used for evaluation.
                The `scoring` parameter represents the strategy to evaluate the performance of the cross-validated model on the test set.
                It can be a string, callable, list, tuple, or dictionary. Default is None.
                The `estimator` parameter represents the estimator object used for optimization. 
                For scoring check https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                and https://scikit-learn.org/stable/modules/model_evaluation.html#scoring.
                For CV check https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
                as an example.

            fit_tune_kwargs : dict, optional
                Additional keyword arguments to be passed to the `fit()` method of the estimator during training.

        Returns
        -------
        TuneBestEstimator
            TuneBestEstimator instance.
        """
        tbe = TuneBestEstimator(**kwargs)
        return tbe

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
