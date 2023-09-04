import numpy as np
import optuna
import pandas as pd
from catboost import *
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import *
from optuna.integration import OptunaSearchCV
from ray import air, tune
from ray.tune.sklearn import TuneGridSearchCV, TuneSearchCV
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import *
from sklearn.svm import *
from xgboost import *

from lohrasb import logger
from lohrasb.abstracts.optimizers import OptimizerABC
from lohrasb.decorators.decorators import trackcalls
from lohrasb.utils.helper_funcs import _trail_params_retrive  # maping_mesurements,
from lohrasb.utils.metrics import *
from interpret.blackbox import *
from interpret.glassbox import *
from xgbse import *


class OptunaSearch(OptimizerABC):

    """
    OptunaSearch class for optimizing an estimator using Optuna.

    Parameters:
    -----------
    X : pd.DataFrame or numpy 2D array
        Input for training.

    y : pd.DataFrame or numpy 1D array
        Target or label.

    args : tuple
        Args of the class.

    kwargs : dict
        Kwargs of the class.

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

    Returns:
    --------
    None

    Methods:
    --------
    prepare_data():
        Prepare data to be consumed by TuneSearchCV. Pass for TuneSearchCV case.

    optimize():
        Optimize estimator using Optuna engine.

    get_optimized_object():
        Get the grid search cv after invoking fit.

    get_best_estimator():
        Return the best estimator if already fitted.

    Notes:
    ------
    It is recommended to use available factories to create a new instance of this class.
    """

    def __init__(
        self,
        X,
        y,
        *args,
        **kwargs,
    ):
        self.study_search_kwargs = kwargs["kwargs"].get("study_search_kwargs", {})
        self.main_optuna_kwargs = kwargs["kwargs"].get("main_optuna_kwargs", {})
        self.optimize_kwargs = kwargs["kwargs"].get("optimize_kwargs", {})
        self.train_test_split_kwargs = kwargs["kwargs"].get(
            "train_test_split_kwargs", {}
        )
        self.fit_optuna_kwargs = kwargs["kwargs"].get("fit_optuna_kwargs", {})
        self.__optuna_search = None
        self.__best_estimator = None
        self.X = X
        self.y = y
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.args = args

    @property
    def study_search_kwargs(self):
        return self._study_search_kwargs

    @study_search_kwargs.setter
    def study_search_kwargs(self, value):
        self._study_search_kwargs = value

    @property
    def main_optuna_kwargs(self):
        return self._main_optuna_kwargs

    @main_optuna_kwargs.setter
    def main_optuna_kwargs(self, value):
        self._main_optuna_kwargs = value

    @property
    def optimize_kwargs(self):
        return self._optimize_kwargs

    @optimize_kwargs.setter
    def optimize_kwargs(self, value):
        self._optimize_kwargs = value

    @property
    def train_test_split_kwargs(self):
        return self._train_test_split_kwargs

    @train_test_split_kwargs.setter
    def train_test_split_kwargs(self, value):
        self._train_test_split_kwargs = value

    @property
    def fit_optuna_kwargs(self):
        return self._fit_optuna_kwargs

    @fit_optuna_kwargs.setter
    def fit_optuna_kwargs(self, value):
        self._fit_optuna_kwargs = value

    def prepare_data(self):
        """
        Prepare data to be consumed by the optimizer.
        """
        # Split the data into training and testing sets using the train_test_split function
        # The resulting splits are assigned to self.X_train, self.X_test, self.y_train, and self.y_test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, **self.train_test_split_kwargs
        )

        return self

    def optimize(self):
        """
        Optimize estimator using Optuna engine.
        """

        def objective(trial):
            # Calculate the metric for evaluation using CalcMetrics
            calc_metric = CalcMetrics(
                y_true=self.y_test,
                y_pred=None,
                metric=self.main_optuna_kwargs["measure_of_accuracy"],
            )
            # Create a metric calculator using calc_make_scorer method
            metric_calculator = calc_metric.calc_make_scorer(
                self.main_optuna_kwargs["measure_of_accuracy"],
            )
            # Retrieve estimator parameters and fit parameters
            estimator_params = self.main_optuna_kwargs["estimator_params"]
            fit_params = self.fit_optuna_kwargs
            estimator = self.main_optuna_kwargs["estimator"]

            # Retrieve parameters from the trial and create an estimator
            params = _trail_params_retrive(trial, estimator_params)
            if fit_params is not None:
                est = eval(
                    estimator.__class__.__name__
                    + "(**params)"
                    + ".fit(self.X_train, self.y_train, **fit_params)"
                )
            else:
                est = eval(
                    estimator.__class__.__name__
                    + "(**params)"
                    + ".fit(self.X_train, self.y_train)"
                )

            # Make predictions using the estimator
            y_pred = est.predict(self.X_test)

            # Calculate the evaluation metric
            if (
                metric_calculator.__class__.__name__ == "_BaseScorer"
                or metric_calculator.__class__.__name__ == "_ProbaScorer"
                or metric_calculator.__class__.__name__ == "_PredictScorer"
                or metric_calculator.__class__.__name__ == "_ThresholdScorer"
            ):
                raise TypeError(
                    "make_scorer is not supported for Optuna optimizer! Read examples and documentations."
                )
            func_str = metric_calculator
            accr = eval(func_str)
            return accr

        # Create an Optuna study
        study = optuna.create_study(**self.study_search_kwargs)

        # Optimize the study using the objective function
        study.optimize(objective, **self.optimize_kwargs)

        # Handle refit and create the final estimator
        for key, value in self.main_optuna_kwargs.items():
            if key == "refit":
                if value:
                    flag = True
                    logger.info(
                        "If refit is set to True, the optimal model will be refit on the entire dataset, i.e., X_train and y_train!"
                    )
                else:
                    flag = False

        if self.fit_optuna_kwargs != {}:
            if flag:
                est = eval(
                    self.main_optuna_kwargs["estimator"].__class__.__name__
                    + "(**study.best_trial.params)"
                    + ".fit(self.X, self.y, **self.fit_optuna_kwargs)"
                )
            else:
                est = eval(
                    self.main_optuna_kwargs["estimator"].__class__.__name__
                    + "(**study.best_trial.params)"
                    + ".fit(self.X_train, self.y_train, **self.fit_optuna_kwargs)"
                )
        else:
            if flag:
                est = eval(
                    self.main_optuna_kwargs["estimator"].__class__.__name__
                    + "(**study.best_trial.params)"
                    + ".fit(self.X, self.y)"
                )
            else:
                est = eval(
                    self.main_optuna_kwargs["estimator"].__class__.__name__
                    + "(**study.best_trial.params)"
                    + ".fit(self.X_train, self.y_train)"
                )

        # Store the best estimator and the Optuna search results
        self.__best_estimator = est
        self.__optuna_search = study.best_trial
        return self

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the model to the training data.

        If the best estimator is already available, return self.
        Otherwise, optimize the estimator using the `optimize` method.

        Parameters:
        -----------
        X : pd.DataFrame or numpy 2D array
            Input for training.
        y : pd.DataFrame or numpy 1D array
            Target or label.
        args : tuple
            Additional positional arguments.
        kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        self : object
            Returns self if the best estimator is already available, or after optimizing the estimator.
        """

        if not self.__best_estimator:
            self.optimize()

        return self

    def get_optimized_object(self):
        """
        Get the best trial from the Optuna study.

        Returns:
        --------
        best_trial : optuna.trial.Trial
            The best trial found during the optimization process.

        """
        return self.__optuna_search

    def get_best_estimator(self):
        """
        Get the best estimator after invoking the fit method.

        Returns:
        --------
        best_estimator : object
            The best estimator found during the optimization process.

        Raises:
        -------
        NotImplementedError:
            If the best estimator is not available (i.e., it is None), indicating that the fit method has not been implemented.

        """
        if not self.__best_estimator:
            raise NotImplementedError(
                "It seems the best estimator is None. Maybe the fit method is not implemented!"
            )

        return self.__best_estimator


class GridSearch(OptimizerABC):
    """
    Class Factories for initializing BestModel optimizing engines, i.e.,
    GridSearchCV.

    Parameters
    ----------

        X : pd.DataFrame
            Input for training
        y : pd.DataFrame
            Target or label

        args : tuple
            Args of the class.

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

    Return
    ----------

    The best estimator of estimator optimized by GridSearchCV.

    Methods
    -------
    prepare_data()
        Prepare data to be consumed by GridSearchCV.Pass for GridSearchCV case.
    optimize()
        Optimize estimator using GridSearchCV engine.
    get_optimized_object()
        Get the grid search cv  after invoking fit.
    get_best_estimator()
        Return the best estimator if already fitted.
    Notes
    -----
    It is recommended to use available factories
    to create a new instance of this class.

    """

    def __init__(
        self,
        X,
        y,
        *args,
        **kwargs,
    ):
        self.grid_search_kwargs = kwargs["kwargs"].get("grid_search_kwargs", {})
        self.main_grid_kwargs = kwargs["kwargs"].get("main_grid_kwargs", {})
        self.fit_grid_kwargs = kwargs["kwargs"].get("fit_grid_kwargs", {})
        self.__grid_search = None
        self.__best_estimator = None
        self.__X = X
        self.__y = y
        self.args = args

    @property
    def grid_search_kwargs(self):
        return self._grid_search_kwargs

    @grid_search_kwargs.setter
    def grid_search_kwargs(self, value):
        self._grid_search_kwargs = value

    @property
    def main_grid_kwargs(self):
        return self._main_grid_kwargs

    @main_grid_kwargs.setter
    def main_grid_kwargs(self, value):
        self._main_grid_kwargs = value

    @property
    def fit_grid_kwargs(self):
        return self._fit_grid_kwargs

    @fit_grid_kwargs.setter
    def fit_grid_kwargs(self, value):
        self._fit_grid_kwargs = value

    def prepare_data(self):
        """
        Prepare data to be consumed by GridSearchCV.
        """
        pass

    @trackcalls
    def optimize(self):
        """
        Optimize estimator using GridSearchCV engine.
        """
        self.__grid_search = GridSearchCV(**self.grid_search_kwargs)
        self.__grid_search.fit(self.__X, self.__y, *self.args, **self.fit_grid_kwargs)
        self.__best_estimator = self.__grid_search.best_estimator_

        return self

    def fit(self, X, y, *args, **kwargs):
        for key, value in self.grid_search_kwargs.items():
            if key == "refit":
                if value:
                    self.__grid_search.fit(
                        self.__X, self.__y, *self.args, **self.fit_grid_kwargs
                    )
                    logger.info(
                        "If refit is set to True, the optimal model will be refit on the entire dataset again!"
                    )
        self.__best_estimator = self.__grid_search.best_estimator_
        return self

    def get_best_estimator(self, *args, **kwargs):
        """
        Get the best estimator after invoking fit on it.
        """
        if self.__best_estimator is not None:
            return self.__best_estimator
        else:
            logger.error("The best estimator is None !")

    def get_optimized_object(self, *args, **kwargs):
        """
        Get the grid search cv  after invoking fit.
        """
        if self.__grid_search is not None:
            return self.__grid_search
        else:
            raise NotImplementedError(
                "GridSearch has not been implemented \
                or best_estomator is null"
            )


class NewOptunaSearch(OptimizerABC):

    """
    Class for initializing BestModel optimizing engines, i.e., GridSearchCV.

    This class provides factories for initializing the BestModel optimizing engine, OptunaSearchCV.
    It allows for preparing data, optimizing an estimator, and accessing the optimized object and the best estimator.

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

    Returns
    -------
    The best estimator of the estimator optimized by OptunaSearchCV.

    Methods
    -------
    prepare_data()
        Prepare data to be consumed by GridSearchCV. Pass for OptunaSearchCV case.
    optimize()
        Optimize estimator using OptunaSearchCV engine.
    get_optimized_object()
        Get the Optuna search CV object after invoking fit.
    get_best_estimator()
        Return the best estimator if already fitted.

    Notes
    -----
    It is recommended to use available factories to create a new instance of this class.
    """

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, *args, **kwargs):
        self.newoptuna_search_kwargs = kwargs["kwargs"].get(
            "newoptuna_search_kwargs", {}
        )
        self.main_newoptuna_kwargs = kwargs["kwargs"].get("main_newoptuna_kwargs", {})
        self.fit_newoptuna_kwargs = kwargs["kwargs"].get("fit_newoptuna_kwargs", {})
        self.__newoptuna_search = None
        self.__best_estimator = None
        self.__X = X
        self.__y = y
        self.args = args

    def prepare_data(self):
        """
        Prepare data to be consumed by OptunaSearchCV.

        This method prepares the data to be used by the OptunaSearchCV optimizer.
        It is not implemented in this class and should be overridden in the derived classes.
        """
        pass

    @trackcalls
    def optimize(self):
        """
        Optimize estimator using OptunaSearchCV engine.

        This method optimizes the estimator using the OptunaSearchCV engine.
        It initializes the OptunaSearchCV with the specified keyword arguments and fits the data.
        The best estimator is stored in the '__best_estimator' attribute.

        Returns
        -------
        self : NewOptunaSearch
            Returns self after optimizing the estimator.
        """
        self.__newoptuna_search = OptunaSearchCV(**self.newoptuna_search_kwargs)
        self.__newoptuna_search.fit(
            self.__X, self.__y, *self.args, **self.fit_newoptuna_kwargs
        )
        self.__best_estimator = self.__newoptuna_search.best_estimator_

        return self

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, *args, **kwargs):
        """
        Fit the estimator using OptunaSearchCV.

        This method fits the estimator using the OptunaSearchCV engine.
        If the 'refit' keyword argument is set to True, the optimal model will be refit on the entire dataset.

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

        Returns
        -------
        self : NewOptunaSearch
            Returns self after fitting the estimator.
        """
        for key, value in self.newoptuna_search_kwargs.items():
            if key == "refit" and value:
                self.__newoptuna_search.fit(
                    self.__X, self.__y, *self.args, **self.fit_newoptuna_kwargs
                )
                logger.info(
                    "If refit is set to True, the optimal model will be refit on the entire dataset again!"
                )
        self.__best_estimator = self.__newoptuna_search.best_estimator_
        return self

    def get_best_estimator(self):
        """
        Get the best estimator after invoking fit on it.

        Returns
        -------
        BaseEstimator or None
            The best estimator if available, otherwise None.
        """
        if self.__best_estimator is not None:
            return self.__best_estimator
        else:
            logger.error("The best estimator is None!")

    def get_optimized_object(self):
        """
        Get the Optuna search CV object after invoking fit.

        Returns
        -------
        OptunaSearchCV or None
            The Optuna search CV object if available, otherwise None.
        """
        if self.__newoptuna_search is not None:
            return self.__newoptuna_search
        else:
            raise NotImplementedError(
                "OptunaSearchCV has not been implemented or the best estimator is None."
            )


class TuneCV(OptimizerABC):
    """
    Class Factories for initializing BestModel optimizing engines, using Tune from Ray.
        Initialize the BestModelFactories.

    Parameters
    ----------
    X : pd.DataFrame or numpy.ndarray
        Input for training.
    y : pd.DataFrame or numpy.ndarray
        Target or label.
    args : tuple
        Additional positional arguments.
    kwargs : dict
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
    None

    Methods
    -------
    prepare_data()
        Prepare data to be consumed by TuneCV. Pass for TuneCV.
    optimize()
        Optimize estimator using OptunaSearchCV engine.
    get_optimized_object()
        Get the Optuna search cv after invoking fit.
    get_best_estimator()
        Return the best estimator if already fitted.

    Notes
    -----
    It is recommended to use available factories to create a new instance of this class.
    The `tuner_kwargs` parameter represents the keyword arguments for the tuner from Ray.
    The `fit_tune_kwargs` parameter represents the additional keyword arguments to be passed to the `fit()` method of the estimator during training.
    The `cv` parameter in `main_tune_kwargs` represents the cross-validation strategy.
    The `scoring` parameter in `main_tune_kwargs` represents the performance evaluation strategy.
    The `estimator` parameter in `main_tune_kwargs` represents the estimator object used for optimization.

    """

    def __init__(self, X, y, *args, **kwargs):
        self.tuner_kwargs = kwargs["kwargs"].get("tuner_kwargs", {})
        self.main_tune_kwargs = kwargs["kwargs"].get("main_tune_kwargs", {})
        self.fit_tune_kwargs = kwargs["kwargs"].get("fit_tune_kwargs", {})
        self.__tune = None
        self.best_estimator = None
        self.__X = X
        self.__y = y
        self.args = args

    @property
    def tuner_kwargs(self):
        return self._tuner_kwargs

    @tuner_kwargs.setter
    def tuner_kwargs(self, value):
        self._tuner_kwargs = value

    @property
    def main_tune_kwargs(self):
        return self._main_tune_kwargs

    @main_tune_kwargs.setter
    def main_tune_kwargs(self, value):
        self._main_tune_kwargs = value

    @property
    def fit_tune_kwargs(self):
        return self._fit_tune_kwargs

    @fit_tune_kwargs.setter
    def fit_tune_kwargs(self, value):
        self._fit_tune_kwargs = value

    def prepare_data(self):
        """
        Prepare data to be consumed by OptunaSearchCV.
        """
        pass

    @trackcalls
    def optimize(self):
        """
        Optimize estimator using OptunaSearchCV engine.
        """

        def trainable(config):
            """
            Trainable function for optimization.
            """

            def objective(X, y):
                """
                Objective function for optimization.
                """
                if hasattr(self.main_tune_kwargs["estimator"], "set_params"):
                    est = self.main_tune_kwargs["estimator"].set_params(**config)
                else:
                    raise AttributeError(
                        f"{self.main_tune_kwargs['estimator']} does not have the 'set_params' attribute!"
                    )

                if self.fit_tune_kwargs:
                    # Perform cross-validation with fit parameters
                    scores = cross_val_score(
                        est,
                        X=X,
                        y=y,
                        cv=self.main_tune_kwargs["cv"],
                        fit_params=self.fit_tune_kwargs,
                        scoring=self.main_tune_kwargs["scoring"],
                    )
                else:
                    # Perform cross-validation without fit parameters
                    scores = cross_val_score(
                        est,
                        X=X,
                        y=y,
                        cv=self.main_tune_kwargs["cv"],
                        scoring=self.main_tune_kwargs["scoring"],
                    )

                return scores

            # Call the objective function with the provided X and y data
            scores = objective(self.__X, self.__y)

            # Calculate the mean score
            score = np.mean(scores)

            # Report the score to Tune
            session.report({"score": score})

        # Create a tuner with the trainable function and tuner_kwargs
        tuner = tune.Tuner(trainable, **self.tuner_kwargs)

        # Fit the tuner to optimize the estimator
        results = tuner.fit()

        # Get the best result from the optimization
        best_result = results.get_best_result()

        # Get the best trial's hyperparameters
        best_config = best_result.config

        # Set the estimator with the best hyperparameters
        est = self.main_tune_kwargs["estimator"].set_params(**best_config)

        # Store the best estimator
        self.best_estimator = est

        # Store the results from Tune
        self.__tune = results

        return self

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the model to the training data.

        If the best estimator is available, fit the data using it.

        Parameters:
        -----------
        X : pd.DataFrame or numpy 2D array
            Input for training.
        y : pd.DataFrame or numpy 1D array
            Target or label.
        args : tuple
            Additional positional arguments.
        kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        self : object
            Returns self after fitting the data using the best estimator.

        """

        if self.best_estimator is not None:
            self.best_estimator.fit(
                self.__X, self.__y, *self.args, **self.fit_tune_kwargs
            )
            return self
        else:
            raise ValueError(
                "The best estimator is not available! Call the 'optimize' method first."
            )

    def get_best_estimator(self, *args, **kwargs):
        """
        Get the best estimator after invoking fit on it.

        Returns:
        --------
        best_estimator : object
            The best estimator found during the optimization process.

        Raises:
        -------
        ValueError:
            If the best estimator is not available (i.e., it is None), indicating that the fit method has not been called or has failed.

        """

        if self.best_estimator is not None:
            return self.best_estimator
        else:
            raise ValueError(
                "The best estimator is not available! Call the 'fit' method first."
            )

    def get_optimized_object(self, *args, **kwargs):
        """
        Get the Optuna search CV  after invoking fit.
        """
        if self.__tune is not None:
            return self.__tune
        else:
            raise NotImplementedError(
                "Tune has not been implemented \
                or best_results is null"
            )


class RandomSearch(OptimizerABC):

    """
    Class Factory for initializing BestModel optimizing engines, i.e., RandomizedSearchCV.
    Initialize the RandomSearch optimizer.

    Parameters
    ----------
    X : pd.DataFrame or numpy array
        Input for training.
    y : pd.DataFrame or numpy array
        Target or label.
    args : tuple
        Additional positional arguments.
    kwargs : dict
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
    None

    Methods
    -------
    prepare_data()
        Prepare data to be consumed by RandomizedSearchCV.
    optimize()
        Optimize the estimator using RandomizedSearchCV engine.
    get_optimized_object()
        Get the RandomizedSearchCV object after invoking fit.
    get_best_estimator()
        Return the best estimator if already fitted.

    Notes
    -----
    It is recommended to use available factories to create a new instance of this class.
    """

    def __init__(
        self,
        X,
        y,
        *args,
        **kwargs,
    ):
        self.random_search_kwargs = kwargs["kwargs"].get("random_search_kwargs", {})
        self.main_random_kwargs = kwargs["kwargs"].get("main_random_kwargs", {})
        self.fit_random_kwargs = kwargs["kwargs"].get("fit_random_kwargs", {})
        self.__random_search = None
        self.__best_estimator = None
        self.__X = X
        self.__y = y
        self.args = args

    @property
    def random_search_kwargs(self):
        return self._random_search_kwargs

    @random_search_kwargs.setter
    def random_search_kwargs(self, value):
        self._random_search_kwargs = value

    @property
    def main_random_kwargs(self):
        return self._main_random_kwargs

    @main_random_kwargs.setter
    def main_random_kwargs(self, value):
        self._main_random_kwargs = value

    @property
    def fit_random_kwargs(self):
        return self._fit_random_kwargs

    @fit_random_kwargs.setter
    def fit_random_kwargs(self, value):
        self._fit_random_kwargs = value

    def prepare_data(self):
        """
        Prepare data to be consumed by RandomizedSearchCV.

        This method performs any necessary preprocessing steps on the data
        before applying RandomizedSearchCV.

        Returns
        -------
        None
        """
        # Implementation specific to prepare the data
        pass

    @trackcalls
    def optimize(self):
        """
        Optimize the estimator using RandomizedSearchCV engine.

        This method performs a randomized search over a hyperparameter space
        to find the best set of hyperparameters for the estimator.

        Returns
        -------
        self : object
            Returns self after optimizing the estimator.
        """
        # Create and fit the RandomizedSearchCV object
        self.__random_search = RandomizedSearchCV(**self.random_search_kwargs)
        self.__random_search.fit(
            self.__X, self.__y, *self.args, **self.fit_random_kwargs
        )

        # Set the best estimator as the result of the random search
        self.__best_estimator = self.__random_search.best_estimator_
        return self

    def fit(self, X, y, *args, **kwargs):
        """
        Get the RandomizedSearchCV object after invoking fit.

        This method returns the RandomizedSearchCV object after it has been
        fitted with the data.

        Returns
        -------
        optimized_object : object
            The optimized RandomizedSearchCV object.
        """
        # Check if refit is set to True
        if self.random_search_kwargs.get("refit", False):
            self.__random_search.fit(
                self.__X, self.__y, *self.args, **self.fit_random_kwargs
            )
            logger.info(
                "If refit is set to True, the optimal model will be refit on the entire dataset again!"
            )

        # Set the best estimator as the result of the random search
        self.__best_estimator = self.__random_search.best_estimator_

        return self

    def get_best_estimator(self, *args, **kwargs):
        """
        Return the best estimator if already fitted.

        This method returns the best estimator found during the optimization process
        if it has already been fitted.

        Returns
        -------
        best_estimator : object
            The best estimator found during the optimization process.
        """
        # Check if the best estimator is available
        if self.__best_estimator is not None:
            return self.__best_estimator
        else:
            logger.error("The best estimator is None !")

    def get_optimized_object(self, *args, **kwargs):
        """
        Get the RandomizedSearchCV object after invoking fit.

        This method returns the RandomizedSearchCV object after it has been
        fitted with the data.

        Parameters
        ----------
        args : tuple
            Additional positional arguments.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        optimized_object : object
            The optimized RandomizedSearchCV object.

        Raises
        ------
        NotImplementedError
            If the RandomizedSearchCV object is not available (i.e., it is None), indicating
            that the random search has not been implemented or the best estimator is null.
        """
        if self.__random_search is not None:
            return self.__random_search
        else:
            raise NotImplementedError(
                "RandomizedSearchCV has not been implemented or the best estimator is null."
            )


class TuneGridSearch(OptimizerABC):

    """
    Class Factories for initializing BestModel optimizing engines, i.e., TuneGridSearchCV.
        Initialize the TuneGridSearch optimizer.

    Parameters
    ----------
    X : pd.DataFrame
        Input for training.
    y : pd.DataFrame
        Target or label.
    args : tuple
        Additional positional arguments.
    kwargs : dict

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

    Returns
    -------
    None

    Methods
    -------
    prepare_data()
        Prepare data to be consumed by TuneGridSearchCV. (Placeholder method)
    optimize()
        Optimize estimator using TuneGridSearchCV engine.
    get_optimized_object()
        Get the grid search cv after invoking fit.
    get_best_estimator()
        Return the best estimator if already fitted.

    Notes
    -----
    It is recommended to use available factories to create a new instance of this class.
    """

    def __init__(self, X, y, *args, **kwargs):
        self._tunegrid_search_kwargs = kwargs.get("kwargs", {}).get(
            "tunegrid_search_kwargs", {}
        )
        self._main_tunegrid_kwargs = kwargs.get("kwargs", {}).get(
            "main_tunegrid_kwargs", {}
        )
        self._fit_tunegrid_kwargs = kwargs.get("kwargs", {}).get(
            "fit_tunegrid_kwargs", {}
        )
        self.__tunegrid_search = None
        self.__best_estimator = None
        self.__X = X
        self.__y = y
        self.args = args

    @property
    def tunegrid_search_kwargs(self):
        """
        Get the TuneGridSearchCV kwargs.

        Returns
        -------
        dict
            The TuneGridSearchCV kwargs.
        """
        return self._tunegrid_search_kwargs

    @tunegrid_search_kwargs.setter
    def tunegrid_search_kwargs(self, value):
        """
        Set the TuneGridSearchCV kwargs.

        Parameters
        ----------
        value : dict
            The TuneGridSearchCV kwargs.

        Returns
        -------
        None
        """
        self._tunegrid_search_kwargs = value

    @property
    def main_tunegrid_kwargs(self):
        """
        Get the main TuneGridSearchCV kwargs.

        Returns
        -------
        dict
            The main TuneGridSearchCV kwargs.
        """
        return self._main_tunegrid_kwargs

    @main_tunegrid_kwargs.setter
    def main_tunegrid_kwargs(self, value):
        """
        Set the main TuneGridSearchCV kwargs.

        Parameters
        ----------
        value : dict
            The main TuneGridSearchCV kwargs.

        Returns
        -------
        None
        """
        self._main_tunegrid_kwargs = value

    @property
    def fit_tunegrid_kwargs(self):
        """
        Get the fit TuneGridSearchCV kwargs.

        Returns
        -------
        dict
            The fit TuneGridSearchCV kwargs.
        """
        return self._fit_tunegrid_kwargs

    @fit_tunegrid_kwargs.setter
    def fit_tunegrid_kwargs(self, value):
        """
        Set the fit TuneGridSearchCV kwargs.

        Parameters
        ----------
        value : dict
            The fit TuneGridSearchCV kwargs.

        Returns
        -------
        None
        """
        self._fit_tunegrid_kwargs = value

    def prepare_data(self):
        """
        Prepare data to be consumed by TuneGridSearchCV.
        """
        pass

    @trackcalls
    def optimize(self):
        """
        Optimize the estimator using TuneGridSearchCV engine.

        This method optimizes the estimator using TuneGridSearchCV.
        It performs an exhaustive search over a specified parameter grid
        to find the best set of hyperparameters for the estimator.

        Returns
        -------
        self : object
            Returns self after optimizing the estimator.
        """
        self.__tunegrid_search = TuneGridSearchCV(**self.tunegrid_search_kwargs)
        self.__tunegrid_search.fit(
            self.__X, self.__y, *self.args, **self.fit_tunegrid_kwargs
        )
        self.__best_estimator = self.__tunegrid_search.best_estimator_

        return self

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the TuneGridSearchCV object with the given data.

        This method fits the TuneGridSearchCV object with the provided data.
        It can also perform refitting if the 'refit' parameter is set to True.

        Parameters
        ----------
        X : pd.DataFrame
            Input for training.
        y : pd.DataFrame
            Target or label.
        args : tuple
            Additional positional arguments.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        self : object
            Returns self after fitting the TuneGridSearchCV object.

        Notes
        -----
        If refit is set to True, the optimal model will be refit on the entire dataset again.
        """
        for key, value in self.tunegrid_search_kwargs.items():
            if key == "refit" and value:
                self.__tunegrid_search.fit(
                    self.__X, self.__y, *self.args, **self.fit_tunegrid_kwargs
                )
                logger.info(
                    "If refit is set to True, the optimal model will be refit on the entire dataset again!"
                )
        self.__best_estimator = self.__tunegrid_search.best_estimator_
        return self

    def get_best_estimator(self, *args, **kwargs):
        """
        Return the best estimator if already fitted.

        This method returns the best estimator found during the optimization process
        if it has already been fitted.

        Parameters
        ----------
        args : tuple
            Additional positional arguments.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        best_estimator : object
            The best estimator found during the optimization process.

        Notes
        -----
        If the best estimator is None, an error message is logged.
        """
        if self.__best_estimator is not None:
            return self.__best_estimator
        else:
            logger.error("The best estimator is None!")

    def get_optimized_object(self, *args, **kwargs):
        """
        Get the TuneGridSearchCV object after invoking fit.

        This method returns the TuneGridSearchCV object after it has been
        fitted with the data.

        Parameters
        ----------
        args : tuple
            Additional positional arguments.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        optimized_object : object
            The optimized TuneGridSearchCV object.

        Raises
        ------
        NotImplementedError
            If the TuneGridSearchCV object is not available (i.e., it is None), indicating
            that the grid search has not been implemented or the best estimator is null.
        """
        if self.__tunegrid_search is not None:
            return self.__tunegrid_search
        else:
            raise NotImplementedError(
                "TuneGridSearchCV has not been implemented or the best estimator is null."
            )


class TuneSearch(OptimizerABC):

    """
    Class for initializing BestModel optimizing engines, i.e., TuneSearchCV.

    This class provides factories for initializing the BestModel optimizing engine, TuneSearchCV.
    It allows for preparing data, optimizing an estimator, and accessing the optimized object and the best estimator.

    Parameters
    ----------
    X : pd.DataFrame
        Input for training.
    y : pd.DataFrame
        Target or label.
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

    Returns
    -------
    The best estimator of the estimator optimized by TuneSearchCV.

    Methods
    -------
    prepare_data()
        Prepare data to be consumed by TuneSearchCV.
    optimize()
        Optimize estimator using TuneSearchCV engine.
    get_optimized_object()
        Get the grid search CV object after invoking fit.
    get_best_estimator()
        Return the best estimator if already fitted.

    Notes
    -----
    It is recommended to use available factories to create a new instance of this class.
    """

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, *args, **kwargs):
        self.tune_search_kwargs = kwargs["kwargs"].get("tune_search_kwargs", {})
        self.main_tune_kwargs = kwargs["kwargs"].get("main_tune_kwargs", {})
        self.fit_tune_kwargs = kwargs["kwargs"].get("fit_tune_kwargs", {})
        self.__tune_search = None
        self.__best_estimator = None
        self.__X = X
        self.__y = y
        self.args = args

    def prepare_data(self):
        """
        Prepare data to be consumed by TuneSearchCV.

        This method prepares the data to be used by the TuneSearchCV optimizer.
        It is not implemented in this class and should be overridden in the derived classes.
        """
        pass

    @trackcalls
    def optimize(self):
        """
        Optimize estimator using TuneSearchCV engine.

        This method optimizes the estimator using the TuneSearchCV engine.
        It initializes the TuneSearchCV with the specified keyword arguments and fits the data.
        The best estimator is stored in the '__best_estimator' attribute.

        Returns
        -------
        self : TuneSearch
            Returns self after optimizing the estimator.
        """
        self.__tune_search = TuneSearchCV(**self.tune_search_kwargs)
        self.__tune_search.fit(self.__X, self.__y, *self.args, **self.fit_tune_kwargs)
        self.__best_estimator = self.__tune_search.best_estimator_

        return self

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, *args, **kwargs):
        """
        Fit the estimator using TuneSearchCV.

        This method fits the estimator using the TuneSearchCV engine.
        If the 'refit' keyword argument is set to True, the optimal model will be refit on the entire dataset.

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

        Returns
        -------
        self : TuneSearch
            Returns self after fitting the estimator.
        """
        for key, value in self.tune_search_kwargs.items():
            if key == "refit" and value:
                self.__tune_search.fit(
                    self.__X, self.__y, *self.args, **self.fit_tune_kwargs
                )
                logger.info(
                    "If refit is set to True, the optimal model will be refit on the entire dataset again!"
                )
        self.__best_estimator = self.__tune_search.best_estimator_
        return self

    def get_best_estimator(self):
        """
        Get the best estimator after invoking fit on it.

        Returns
        -------
        BaseEstimator or None
            The best estimator if available, otherwise None.
        """
        if self.__best_estimator is not None:
            return self.__best_estimator
        else:
            logger.error("The best estimator is None!")

    def get_optimized_object(self):
        """
        Get the TuneSearchCV object after invoking fit.

        Returns
        -------
        TuneSearchCV or None
            The TuneSearchCV object if available, otherwise None.
        """
        if self.__tune_search is not None:
            return self.__tune_search
        else:
            raise NotImplementedError(
                "TuneSearchCV has not been implemented or the best estimator is None."
            )
