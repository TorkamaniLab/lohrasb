import subprocess
import numpy as np
import pandas as pd
import ray
from imblearn.ensemble import *
from interpret.blackbox import *
from interpret.glassbox import *
from lightgbm import *
from ray.tune.sklearn import TuneGridSearchCV, TuneSearchCV
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import *
from sklearn.svm import *
from xgboost import *
from xgbse import *
from lohrasb import logger
from lohrasb.abstracts.optimizers import OptimizerABC
from lohrasb.decorators.decorators import trackcalls
from lohrasb.utils.helper_funcs import _trail_params_retrive  # maping_mesurements,
from lohrasb.utils.metrics import *
import optuna

class OptunaSearch(OptimizerABC):
    def __init__(
        self,
        X,
        y,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------

            X : pd.DataFrame
                Input for training
            y : pd.DataFrame
                Target or label
        Return
        ----------

        The best estimator of estimator optimized by TuneSearchCV.

        Methods
        -------
        prepare_data()
            Prepare data to be consumed by TuneSearchCV.Pass for TuneSearchCV case.
        optimize()
            Optimize estimator using TuneSearchCV engine.
        get_optimized_object()
            Get the grid search cv  after invoking fit.
        get_best_estimator()
            Return the best estimator if already fitted.
        Notes
        -----
        It is recommended to use available factories
        to create a new instance of this class.

        """
        self.study_search_kwargs = kwargs['kwargs'].get('study_search_kwargs',{})
        self.main_optuna_kwargs = kwargs['kwargs'].get('main_optuna_kwargs',{})
        self.optimize_kwargs = kwargs['kwargs'].get('optimize_kwargs',{})
        self.train_test_split_kwargs = kwargs['kwargs'].get('train_test_split_kwargs',{})
        self.fit_optuna_kwargs= kwargs['kwargs'].get('fit_optuna_kwargs',{})
        self.__optuna_search = None
        self.__best_estimator = None
        self.__X = X
        self.__y = y
        self.args = args
    
    @property
    def study_search_kwargs(self):
        return self._study_search_kwargs

    @study_search_kwargs.setter
    def study_search_kwargs(self, value):
        self._study_search_kwargs= value

    @property
    def main_optuna_kwargs(self):
        return self._main_optuna_kwargs

    @main_optuna_kwargs.setter
    def main_optuna_kwargs(self, value):
        self._main_optuna_kwargs= value

    @property
    def optimize_kwargs(self):
        return self._optimize_kwargs

    @optimize_kwargs.setter
    def optimize_kwargs(self, value):
        self._optimize_kwargs= value

    @property
    def train_test_split_kwargs(self):
        return self._train_test_split_kwargs

    @train_test_split_kwargs.setter
    def train_test_split_kwargs(self, value):
        self._train_test_split_kwargs= value

    @property
    def fit_optuna_kwargs(self):
        return self._fit_optuna_kwargs

    @fit_optuna_kwargs.setter
    def fit_optuna_kwargs(self, value):
        self._fit_optuna_kwargs= value
    
    def prepare_data(self):
        """
        Prepare data to be consumed by the optimizer.
        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            **self.train_test_split_kwargs
        )
    
        return self

    def optimize(self):
        """
        Optimize estimator using Optuna engine.
        """

        calc_metric = CalcMetrics(
            y_true=self.y_test,
            y_pred=None,
            metric=self.main_optuna_kwargs['measure_of_accuracy'],
        )
        metric_calculator = calc_metric.calc_make_scorer(
            self.main_optuna_kwargs['measure_of_accuracy'],
        )

        def objective(trial):
            estimator_params = self.main_optuna_kwargs['estimator_params']
            fit_params = self.fit_optuna_kwargs
            estimator=self.main_optuna_kwargs['estimator']
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


            y_pred = est.predict(self.X_test)
            if (
                metric_calculator.__class__.__name__ == "_BaseScorer"
                or metric_calculator.__class__.__name__ == "_ProbaScorer"
                    or metric_calculator.__class__.__name__ == "_PredictScorer"
                or metric_calculator.__class__.__name__ == "_ThresholdScorer"
            ):
                raise TypeError(
                    "make_scorer is not supported for Optuna optimizer ! Read examples and documentations. "
                )
            func_str = metric_calculator
            accr = eval(func_str)
            return accr

        study = optuna.create_study(
            **self.study_search_kwargs
        )

        study.optimize(
            objective,
            self.optimize_kwargs
        )
        for key, value in self.main_optuna_kwargs.items():
            if key == 'refit':
                if value:
                    flag = True
                    logger.info('If refit is set to True, the optimal model will be refit on the entire dataset, i.e., X_train and y_train!')
                else:
                    flag = False

        if self.fit_optuna_kwargs is not {}:
            if flag :
                est = eval(
                    self.main_optuna_kwargs['estimator'].__class__.__name__
                    + "(**study.best_trial)"
                    + ".fit(self.X, self.y, **self.fit_optuna_kwargs)"
                )
            else:
                est = eval(
                    self.main_optuna_kwargs['estimator'].__class__.__name__
                    + "(**study.best_trial)"
                    + ".fit(self.X_train, self.y_train, **self.fit_optuna_kwargs)"
                )
        else:
            if flag :
                est = eval(
                    self.main_optuna_kwargs['estimator'].__class__.__name__
                    + "(**study.best_trial)"
                    + ".fit(self.X, self.y)"
                )
            else:
                est = eval(
                    self.main_optuna_kwargs['estimator'].__class__.__name__
                    + "(**study.best_trial)"
                    + ".fit(self.X_train, self.y_train)"
                )

        self.__best_estimator = est
        self.__optuna_search = study.best_trial
        return self

    def fit(self, X, y, *args, **kwargs):
        if self.__best_estimator:
            return self
        else:
            self.optimize()
        return self

    def get_optimized_object(self):
        """
        Get study best_trial
        """

        return self.__optuna_search

    def get_best_estimator(self):
        """
        Get the best estimator after invoking fit on it.
        """
        if self.__best_estimator:
            return self.__best_estimator
        else:
            raise NotImplementedError('It seems best estimator is Null. Maybe fit not implemented !')

class GridSearch(OptimizerABC):
    """
    Class Factories for initializing BestModel optimizing engines, i.e.,
    GridSearchCV.

    """

    def __init__(
        self,
        X,
        y,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------

            X : pd.DataFrame
                Input for training
            y : pd.DataFrame
                Target or label
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
        self.grid_search_kwargs = kwargs['kwargs'].get('grid_search_kwargs',{})
        self.main_grid_kwargs = kwargs['kwargs'].get('main_grid_kwargs',{})
        self.fit_grid_kwargs = kwargs['kwargs'].get('fit_grid_kwargs',{})
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
        self._main_grid_kwargs= value

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
        self.__grid_search.fit(self.__X, self.__y,
                               *self.args, **self.fit_grid_kwargs)
        self.__best_estimator = self.__grid_search.best_estimator_

        return self

    def fit(self, X, y, *args, **kwargs):
        for key, value in self.grid_search_kwargs.items():
            if key == 'refit':
                if value:
                    self.__grid_search.fit(self.__X, self.__y,
                                           *self.args, **self.fit_grid_kwargs)
                    logger.info('If refit is set to True, the optimal model will be refit on the entire dataset again!')
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

class RandomSearch(OptimizerABC):
    """
    Class Factories for initializing BestModel optimizing engines, i.e.,
    RandomizedSearchCV.

    """

    def __init__(
        self,
        X,
        y,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------

            X : pd.DataFrame
                Input for training
            y : pd.DataFrame
                Target or label
        Return
        ----------

        The best estimator of estimator optimized by randomSearchCV.

        Methods
        -------
        prepare_data()
            Prepare data to be consumed by randomSearchCV.Pass for randomSearchCV case.
        optimize()
            Optimize estimator using randomSearchCV engine.
        get_optimized_object()
            Get the random search cv  after invoking fit.
        get_best_estimator()
            Return the best estimator if already fitted.
        Notes
        -----
        It is recommended to use available factories
        to create a new instance of this class.

        """
        self.random_search_kwargs = kwargs['kwargs'].get('random_search_kwargs',{})
        self.main_random_kwargs = kwargs['kwargs'].get('main_random_kwargs',{})
        self.fit_random_kwargs = kwargs['kwargs'].get('fit_random_kwargs',{})
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
        Prepare data to be consumed by randomSearchCV.
        """
        pass

    @trackcalls
    def optimize(self):
        """
        Optimize estimator using randomSearchCV engine.
        """
        self.__random_search = RandomizedSearchCV(**self.random_search_kwargs)
        self.__random_search.fit(self.__X, self.__y,
                               *self.args, **self.fit_random_kwargs)
        self.__best_estimator = self.__random_search.best_estimator_

        return self

    def fit(self, X, y, *args, **kwargs):
        for key, value in self.random_search_kwargs.items():
            if key == 'refit':
                if value:
                    self.__random_search.fit(self.__X, self.__y,
                                           *self.args, **self.fit_random_kwargs)
                    logger.info('If refit is set to True, the optimal model will be refit on the entire dataset again!')
        self.__best_estimator = self.__random_search.best_estimator_
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
        Get the random search cv  after invoking fit.
        """
        if self.__random_search is not None:
            return self.__random_search
        else:
            raise NotImplementedError(
                "randomSearch has not been implemented \
                or best_estomator is null"
            )

class TuneGridSearch(OptimizerABC):
    """
    Class Factories for initializing BestModel optimizing engines, i.e.,
    TuneGridSearch.

    """

    def __init__(
        self,
        X,
        y,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------

            X : pd.DataFrame
                Input for training
            y : pd.DataFrame
                Target or label
        Return
        ----------

        The best estimator of estimator optimized by TuneGridSearchCV.

        Methods
        -------
        prepare_data()
            Prepare data to be consumed by TuneGridSearchCV.Pass for TuneGridSearchCV case.
        optimize()
            Optimize estimator using TuneGridSearchCV engine.
        get_optimized_object()
            Get the grid search cv  after invoking fit.
        get_best_estimator()
            Return the best estimator if already fitted.
        Notes
        -----
        It is recommended to use available factories
        to create a new instance of this class.

        """
        self.tunegrid_search_kwargs = kwargs['kwargs'].get('tunegrid_search_kwargs',{})
        self.main_tunegrid_kwargs = kwargs['kwargs'].get('main_tunegrid_kwargs',{})
        self.fit_tunegrid_kwargs = kwargs['kwargs'].get('fit_tunegrid_kwargs',{})
        self.__tunegrid_search = None
        self.__best_estimator = None
        self.__X = X
        self.__y = y
        self.args = args
    
    @property
    def tunegrid_search_kwargs(self):
        return self._tunegrid_search_kwargs

    @tunegrid_search_kwargs.setter
    def tunegrid_search_kwargs(self, value):
        self._tunegrid_search_kwargs = value

    @property
    def main_tunegrid_kwargs(self):
        return self._main_tunegrid_kwargs

    @main_tunegrid_kwargs.setter
    def main_tunegrid_kwargs(self, value):
        self._main_tunegrid_kwargs= value

    @property
    def fit_tunegrid_kwargs(self):
        return self._fit_tunegrid_kwargs

    @fit_tunegrid_kwargs.setter
    def fit_tunegrid_kwargs(self, value):
        self._fit_tunegrid_kwargs = value

    def prepare_data(self):
        """
        Prepare data to be consumed by TuneGridSearchCV.
        """
        pass

    @trackcalls
    def optimize(self):
        """
        Optimize estimator using TuneGridSearchCV engine.
        """
        self.__tunegrid_search = TuneGridSearchCV(**self.tunegrid_search_kwargs)
        self.__tunegrid_search.fit(self.__X, self.__y,
                               *self.args, **self.fit_tunegrid_kwargs)
        self.__best_estimator = self.__tunegrid_search.best_estimator_

        return self

    def fit(self, X, y, *args, **kwargs):
        for key, value in self.tunegrid_search_kwargs.items():
            if key == 'refit':
                if value:
                    self.__tunegrid_search.fit(self.__X, self.__y,
                                           *self.args, **self.fit_tunegrid_kwargs)
                    logger.info('If refit is set to True, the optimal model will be refit on the entire dataset again!')
        self.__best_estimator = self.__tunegrid_search.best_estimator_
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
        if self.__tunegrid_search is not None:
            return self.__tunegrid_search
        else:
            raise NotImplementedError(
                "GridSearch has not been implemented \
                or best_estomator is null"
            )


class TuneSearch(OptimizerABC):
    """
    Class Factories for initializing BestModel optimizing engines, i.e.,
    TuneSearchCV.

    """
    def __init__(
        self,
        X,
        y,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------

            X : pd.DataFrame
                Input for training
            y : pd.DataFrame
                Target or label
        Return
        ----------

        The best estimator of estimator optimized by TuneSearchCV.

        Methods
        -------
        prepare_data()
            Prepare data to be consumed by TuneSearchCV.Pass for TuneSearchCV case.
        optimize()
            Optimize estimator using TuneSearchCV engine.
        get_optimized_object()
            Get the grid search cv  after invoking fit.
        get_best_estimator()
            Return the best estimator if already fitted.
        Notes
        -----
        It is recommended to use available factories
        to create a new instance of this class.

        """
        self.tune_search_kwargs = kwargs['kwargs'].get('tune_search_kwargs',{})
        self.main_tune_kwargs = kwargs['kwargs'].get('main_tune_kwargs',{})
        self.fit_tune_kwargs = kwargs['kwargs'].get('fit_tune_kwargs',{})
        self.__tune_search = None
        self.__best_estimator = None
        self.__X = X
        self.__y = y
        self.args = args
    
    @property
    def tune_search_kwargs(self):
        return self._tune_search_kwargs

    @tune_search_kwargs.setter
    def tune_search_kwargs(self, value):
        self._tune_search_kwargs = value

    @property
    def main_tune_kwargs(self):
        return self._main_tune_kwargs

    @main_tune_kwargs.setter
    def main_tune_kwargs(self, value):
        self._main_tune_kwargs= value

    @property
    def fit_tune_kwargs(self):
        return self._fit_tune_kwargs

    @fit_tune_kwargs.setter
    def fit_tune_kwargs(self, value):
        self._fit_tune_kwargs = value

    def prepare_data(self):
        """
        Prepare data to be consumed by TuneSearchCV.
        """
        pass

    @trackcalls
    def optimize(self):
        """
        Optimize estimator using TuneSearchCV engine.
        """
        self.__tune_search = TuneSearchCV(**self.tune_search_kwargs)
        self.__tune_search.fit(self.__X, self.__y,
                               *self.args, **self.fit_tune_kwargs)
        self.__best_estimator = self.__tune_search.best_estimator_

        return self

    def fit(self, X, y, *args, **kwargs):
        for key, value in self.tune_search_kwargs.items():
            if key == 'refit':
                if value:
                    self.__tune_search.fit(self.__X, self.__y,
                                           *self.args, **self.fit_tune_kwargs)
                    logger.info('If refit is set to True, the optimal model will be refit on the entire dataset again!')
        self.__best_estimator = self.__tune_search.best_estimator_
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
        if self.__tune_search is not None:
            return self.__tune_search
        else:
            raise NotImplementedError(
                "GridSearch has not been implemented \
                or best_estomator is null"
            )
