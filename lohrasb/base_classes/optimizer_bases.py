import subprocess

import numpy as np
import pandas as pd
from imblearn.ensemble import *
from lightgbm import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import *
from sklearn.svm import *
from xgboost import *
from xgbse import *

from lohrasb.abstracts.optimizers import OptimizerABC
from lohrasb.decorators.decorators import trackcalls
from lohrasb.utils.helper_funcs import _trail_params_retrive  # maping_mesurements,
from lohrasb.utils.metrics import *


class OptunaSearch(OptimizerABC):
    """
    Class Factories for initializing BestModel optimizing engines, i.e.,
    Optuna
    """

    def __init__(
        self,
        X,
        y,
        verbose,
        random_state,
        estimator,
        estimator_params,
        fit_params,
        # grid search and random search
        measure_of_accuracy,
        n_jobs,
        # optuna params
        test_size,
        with_stratified,
        # number_of_trials=100,
        # optuna study init params
        study,
        # optuna optimization params
        study_optimize_objective,
        study_optimize_objective_n_trials,
        study_optimize_objective_timeout,
        study_optimize_n_jobs,
        study_optimize_catch,
        study_optimize_callbacks,
        study_optimize_gc_after_trial,
        study_optimize_show_progress_bar,
    ):

        """
        Parameters
        ----------
            estimator: object
                An unfitted estimator that has fit and predicts methods.
            estimator_params: dict
                Parameters were passed to find the best estimator using the optimization
                method.
            fit_params: dict
                Parameters passed to the fit method of the estimator.
            measure_of_accuracy : str
                Measurement of performance for classification and
                regression estimator during hyperparameter optimization while
                estimating best estimator.
                Classification-supported measurements are :
                "accuracy_score", "auc", "precision_recall_curve","balanced_accuracy_score",
                "cohen_kappa_score","dcg_score","det_curve", "f1_score", "fbeta_score",
                "hamming_loss","fbeta_score", "jaccard_score", "matthews_corrcoef","ndcg_score",
                "precision_score", "recall_score", "roc_auc_score", "roc_curve", "top_k_accuracy_score",
                "zero_one_loss"
                # custom
                "f1_plus_tp", "f1_plus_tn", "specificity", "roc_plus_f1", "auc_plus_f1", "precision_recall_curve"
                "precision_recall_fscore_support".
                Regression Classification-supported measurements are:
                "explained_variance_score", "max_error","mean_absolute_error","mean_squared_log_error",
                "mean_absolute_percentage_error","mean_squared_log_error","median_absolute_error",
                "mean_absolute_percentage_error","r2_score","mean_poisson_deviance","mean_gamma_deviance",
                "mean_tweedie_deviance","d2_tweedie_score","mean_pinball_loss","d2_pinball_score", "d2_absolute_error_score",
                "tn", "tp", "tn_score" ,"tp_score".
                Examples of use:
                "f1_plus_tn(y_true, y_pred)"
                "f1_score(y_true, y_pred, average='weighted')"
                "mean_poisson_deviance(y_true, y_pred)"
                and so on.

            test_size : float or int
                If float, it should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the train split during estimating the best estimator
                by optimization method. If it means the
                absolute number of train samples. If None, the value is automatically
                set to the complement of the test size.

            with_stratified: bool
                Set True if you want data split in a stratified fashion. (default ``True``)
            verbose: int
                Controls the verbosity across all objects: the higher, the more messages.
            random_state: int
                Random number seed.
            n_jobs: int
                The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
                ``-1`` means using all processors. (default -1)
            study: object
                Create an optuna study. For setting its parameters, visit
                https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study
            study_optimize_objective : object
                A callable that implements an objective function.
            study_optimize_objective_n_trials: int
                The number of trials. If this argument is set to obj:`None`, there is no
                limitation on the number of trials. If:obj:`timeout` is also set to:obj:`None,`
                the study continues to create trials until it receives a termination signal such
                as Ctrl+C or SIGTERM.
            study_optimize_objective_timeout : int
                Stop studying after the given number of seconds (s). If this argument is set to
                :obj:`None`, the study is executed without time limitation. If:obj:`n_trials` is
                also set to obj:`None,` the study continues to create trials until it receives a
                termination signal such as Ctrl+C or SIGTERM.
            study_optimize_n_jobs : int ,
                The number of parallel jobs. If this argument is set to obj:`-1`, the number is
                set to CPU count.
            study_optimize_catch: object
                A study continues to run even when a trial raises one of the exceptions specified
                in this argument. Default is an empty tuple, i.e., the study will stop for any
                exception except for class:`~optuna.exceptions.TrialPruned`.
            study_optimize_callbacks: [callback functions]
                List of callback functions that are invoked at the end of each trial. Each function
                must accept two parameters with the following types in this order:
            study_optimize_gc_after_trial: bool
                Flag to determine whether to run garbage collection after each trial automatically.
                Set to:obj:`True` to run the garbage collection: obj:`False` otherwise.
                When it runs, it runs a full collection by internally calling:func:`gc.collect`.
                If you see an increase in memory consumption over several trials, try setting this
                flag to obj:`True`.
            study_optimize_show_progress_bar: bool
                Flag to show progress bars or not. To disable the progress bar.
        Return
        ----------

        The best estimator of estimator optimized by Optuna.

        Methods
        -------
        prepare_data()
            prepare data before optimization.
        optimize()
            Optimize estimator using Optuna engine.
        get_optimized_object()
            Get study best_trial
        get_best_estimator()
            Return the best estimator if already fitted.
        Notes
        -----
        It is recommended to use available factories
        to create a new instance of this class.

        """

        self.X = X
        self.y = y
        self.verbose = verbose
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.fit_params = fit_params
        # grid search and random search
        self.measure_of_accuracy = measure_of_accuracy
        self.n_jobs = n_jobs
        # optuna params
        self.test_size = test_size
        self.with_stratified = with_stratified
        # optuna study init params
        self.study = study
        # optuna optimization params
        self.study_optimize_objective = study_optimize_objective
        self.study_optimize_objective_n_trials = study_optimize_objective_n_trials
        self.study_optimize_objective_timeout = study_optimize_objective_timeout
        self.study_optimize_n_jobs = study_optimize_n_jobs
        self.study_optimize_catch = study_optimize_catch
        self.study_optimize_callbacks = study_optimize_callbacks
        self.study_optimize_gc_after_trial = study_optimize_gc_after_trial
        self.study_optimize_show_progress_bar = study_optimize_show_progress_bar
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.objective = None
        self.trial = None
        self.calc_metric = None
        self.metric_calculator = None
        self.est = None

    @property
    def est(self):
        return self._est

    @est.setter
    def est(self, value):
        self._est = value

    @property
    def metric_calculator(self):
        return self._metric_calculator

    @metric_calculator.setter
    def metric_calculator(self, value):
        self._metric_calculator = value

    @property
    def calc_metric(self):
        return self._calc_metric

    @calc_metric.setter
    def calc_metric(self, value):
        self._calc_metric = value

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        self._random_state = value

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

    @property
    def estimator_params(self):
        return self._estimator_params

    @estimator_params.setter
    def estimator_params(self, value):
        self._estimator_params = value

    @property
    def fit_params(self):
        return self._fit_params

    @fit_params.setter
    def fit_params(self, value):
        self._fit_params = value

    @property
    def measure_of_accuracy(self):
        return self._measure_of_accuracy

    @measure_of_accuracy.setter
    def measure_of_accuracy(self, value):
        self._measure_of_accuracy = value

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    @property
    def test_size(self):
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        self._test_size = value

    @property
    def with_stratified(self):
        return self._with_stratified

    @with_stratified.setter
    def with_stratified(self, value):
        self._with_stratified = value

    @property
    def study(self):
        return self._study

    @study.setter
    def study(self, value):
        self._study = value

    @property
    def study_optimize_objective(self):
        return self._study_optimize_objective

    @study_optimize_objective.setter
    def study_optimize_objective(self, value):
        self._study_optimize_objective = value

    @property
    def study_optimize_objective_n_trials(self):
        return self._study_optimize_objective_n_trials

    @study_optimize_objective_n_trials.setter
    def study_optimize_objective_n_trials(self, value):
        self._study_optimize_objective_n_trials = value

    @property
    def study_optimize_objective_timeout(self):
        return self._study_optimize_objective_timeout

    @study_optimize_objective_timeout.setter
    def study_optimize_objective_timeout(self, value):
        self._study_optimize_objective_timeout = value

    @property
    def study_optimize_n_jobs(self):
        return self._study_optimize_n_jobs

    @study_optimize_n_jobs.setter
    def study_optimize_n_jobs(self, value):
        self._study_optimize_n_jobs = value

    @property
    def study_optimize_catch(self):
        return self._study_optimize_catch

    @study_optimize_catch.setter
    def study_optimize_catch(self, value):
        self._study_optimize_catch = value

    @property
    def study_optimize_callbacks(self):
        return self._study_optimize_callbacks

    @study_optimize_callbacks.setter
    def study_optimize_callbacks(self, value):
        self._study_optimize_callbacks = value

    @property
    def study_optimize_gc_after_trial(self):
        return self._study_optimize_gc_after_trial

    @study_optimize_gc_after_trial.setter
    def study_optimize_gc_after_trial(self, value):
        self._study_optimize_gc_after_trial = value

    @property
    def study_optimize_show_progress_bar(self):
        return self._study_optimize_show_progress_bar

    @study_optimize_show_progress_bar.setter
    def study_optimize_show_progress_bar(self, value):
        self._study_optimize_show_progress_bar = value

    @property
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, value):
        self._X_train = value

    @property
    def X_test(self):
        return self._X_test

    @X_test.setter
    def X_test(self, value):
        self._X_test = value

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        self._y_train = value

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        self._objective = value

    @property
    def trial(self):
        return self._trial

    @trial.setter
    def trial(self, value):
        self._trial = value

    def prepare_data(self):
        """
        Prepare data to be consumed by the optimizer.
        """

        if self.with_stratified and isinstance(self.y, pd.DataFrame):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=self.test_size,
                stratify=self.y[self.y.columns.to_list()[0]],
                random_state=self.random_state,
            )
        elif self.with_stratified and not isinstance(self.y, pd.DataFrame):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=self.test_size,
                stratify=self.y,
                random_state=self.random_state,
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )

        return self

    def optimize(self):
        """
        Optimize estimator using Optuna engine.
        """

        self.calc_metric = CalcMetrics(
            y_true=self.y,
            y_pred=None,
            metric=self.measure_of_accuracy,
        )
        self.metric_calculator = self.calc_metric.calc_make_scorer(
            self.measure_of_accuracy
        )

        def objective(trial):
            params = _trail_params_retrive(trial, self.estimator_params)
            if self.fit_params is not None:
                self.est = eval(
                    self.estimator.__class__.__name__
                    + "(**params)"
                    + ".fit(self.X_train, self.y_train, **self.fit_params)"
                )
            else:
                self.est = eval(
                    self.estimator.__class__.__name__
                    + "(**params)"
                    + ".fit(self.X_train, self.y_train)"
                )

            y_pred = self.est.predict(self.X)
            if (
                self.metric_calculator.__class__.__name__ == "_BaseScorer"
                or self.metric_calculator.__class__.__name__ == "_ProbaScorer"
                or self.metric_calculator.__class__.__name__ == "_PredictScorer"
                or self.metric_calculator.__class__.__name__ == "_ThresholdScorer"
            ):
                raise TypeError(
                    "make_scorer is not supported for Optuna optimizer ! Read examples and documentations. "
                )
            func_str = self.metric_calculator
            print(func_str)
            accr = eval(func_str)
            return accr

        self.study.optimize(
            objective,
            n_trials=self.study_optimize_objective_n_trials,
            timeout=self.study_optimize_objective_timeout,
            n_jobs=self.study_optimize_n_jobs,
            catch=self.study_optimize_catch,
            callbacks=self.study_optimize_callbacks,
            gc_after_trial=self.study_optimize_gc_after_trial,
            show_progress_bar=self.study_optimize_show_progress_bar,
        )
        self.trial = self.study.best_trial
        return self

    def get_optimized_object(self):
        """
        Get study best_trial
        """

        return self.study.best_trial

    def get_best_estimator(self):
        """
        Get the best estimator after invoking fit on it.
        """
        self.estimator = eval(
            self.estimator.__class__.__name__ + "(**self.trial.params)"
        )
        self.best_estimator = self.estimator.fit(self.X_train, self.y_train)
        return self.best_estimator


class GridSearch(OptimizerABC):
    """
    Class Factories for initializing BestModel optimizing engines, i.e.,
    GridSearchCV.

    """

    def __init__(
        self,
        X,
        y,
        estimator,
        estimator_params,
        fit_params,
        measure_of_accuracy,
        random_state,
        verbose,
        n_jobs,
        cv,
    ):
        """
        Parameters
        ----------

            estimator: object
                An unfitted estimator that has fit and predicts methods.
            estimator_params: dict
                Parameters were passed to find the best estimator using the optimization
                method.
            measure_of_accuracy : object of type make_scorer
                see documentation in
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html

            random_state: int
                Random number seed.
            verbose: int
                Controls the verbosity across all objects: the higher, the more messages.
            n_jobs: int
                The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
                ``-1`` means using all processors. (default -1)
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
        self.X = X
        self.y = y
        self.estimator = estimator
        self.fit_params = fit_params
        self.random_state = random_state
        self.estimator_params = estimator_params
        self.measure_of_accuracy = measure_of_accuracy
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.cv = cv
        self.grid_search = None
        self.best_estimator = None
        self.calc_metric = CalcMetrics(
            y_true=y,
            y_pred=None,
            metric=self.measure_of_accuracy,
        )

    @property
    def fit_params(self):
        return self._fit_params

    @fit_params.setter
    def fit_params(self, value):
        self._fit_params = value

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

    @property
    def estimator_params(self):
        return self._estimator_params

    @estimator_params.setter
    def estimator_params(self, value):
        self._estimator_params = value

    @property
    def measure_of_accuracy(self):
        return self._measure_of_accuracy

    @measure_of_accuracy.setter
    def measure_of_accuracy(self, value):
        self._measure_of_accuracy = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, value):
        self._cv = value

    @property
    def grid_search(self):
        return self._grid_search

    @grid_search.setter
    def grid_search(self, value):
        self._grid_search = value

    @property
    def best_estimator(self):
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, value):
        self._best_estimator = value

    @property
    def calc_metric(self):
        return self._calc_metric

    @calc_metric.setter
    def calc_metric(self, value):
        self._calc_metric = value

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
        self.grid_search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.estimator_params,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring=self.calc_metric.calc_make_scorer(self.measure_of_accuracy),
            verbose=self.verbose,
        )

        if self.fit_params is not None:
            self.grid_search.fit(self.X, self.y, **self.fit_params)
        else:
            self.grid_search.fit(self.X, self.y)
        self.best_estimator = self.grid_search.best_estimator_
        return self

    @trackcalls
    def get_best_estimator(self, *args, **kwargs):
        """
        Get the best estimator after invoking fit on it.
        """
        if self.optimize.has_been_called and self.best_estimator is not None:
            return self.best_estimator
        else:
            self.optimize(
                self.estimator,
                param_grid=self.estimator_params,
                cv=self.cv,
                n_jobs=self.n_jobs,
                scoring=self.calc_metric.calc_make_scorer(self.measure_of_accuracy),
                verbose=self.verbose,
            )

            if self.optimize.has_been_called and self.best_estimator is not None:
                return self.best_estimator
            else:
                raise NotImplementedError(
                    "RandomSearch has not been implemented \
                    or best_estomator is null"
                )

    def get_optimized_object(self, *args, **kwargs):
        """
        Get the grid search cv  after invoking fit.
        """
        if self.optimize.has_been_called and self.grid_search is not None:
            return self.grid_search
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
        estimator,
        estimator_params,
        fit_params,
        measure_of_accuracy,
        verbose,
        random_state,
        n_jobs,
        n_iter,
        cv,
    ):

        """
        Parameters
        ----------

        estimator: object
            An unfitted estimator that has fit and predicts methods.
        estimator_params: dict
            Parameters were passed to find the best estimator using the optimization
            method.
        measure_of_accuracy : object of type make_scorer
            see documentation in
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        random_state: int
            Random number seed.
        verbose: int
            Controls the verbosity across all objects: the higher, the more messages.
        n_jobs: int
            The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
            ``-1`` means using all processors. (default -1)
        n_iter : int
            Only it means full in Random Search. It is several parameter
            settings that are sampled. n_iter trades off runtime vs. quality of the solution.

        Return
        ----------

        The best estimator of estimator optimized by RandomizedSearchCV.

        Methods
        -------
        prepare_data()
            Prepare data to be consumed by RandomizedSearchCV.Pass for RandomizedSearchCV case.
        optimize()
            Optimize estimator using RandomizedSearchCV engine.
        get_optimized_object()
            Get the grid search cv  after invoking fit.
        get_best_estimator()
            Return the best estimator if already fitted.
        Notes
        -----
        It is recommended to use available factories
        to create a new instance of this class.

        """

        self.X = X
        self.y = y
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.fit_params = fit_params
        self.measure_of_accuracy = measure_of_accuracy
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.random_search = None
        self.best_estimator = None
        self.calc_metric = CalcMetrics(
            y_true=y,
            y_pred=None,
            metric=self.measure_of_accuracy,
        )

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

    @property
    def estimator_params(self):
        return self._estimator_params

    @estimator_params.setter
    def estimator_params(self, value):
        self._estimator_params = value

    @property
    def fit_params(self):
        return self._fit_params

    @fit_params.setter
    def fit_params(self, value):
        self._fit_params = value

    @property
    def measure_of_accuracy(self):
        return self._measure_of_accuracy

    @measure_of_accuracy.setter
    def measure_of_accuracy(self, value):
        self._measure_of_accuracy = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    @property
    def n_iter(self):
        return self._n_iter

    @n_iter.setter
    def n_iter(self, value):
        self._n_iter = value

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, value):
        self._cv = value

    @property
    def random_search(self):
        return self._random_search

    @random_search.setter
    def random_search(self, value):
        self._random_search = value

    @property
    def best_estimator(self):
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, value):
        self._best_estimator = value

    @property
    def calc_metric(self):
        return self._calc_metric

    @calc_metric.setter
    def calc_metric(self, value):
        self._calc_metric = value

    def prepare_data(self):
        pass

    @trackcalls
    def optimize(self):
        """
        Optimize estimator using GridSearchCV engine.
        """
        self.random_search = RandomizedSearchCV(
            self.estimator,
            param_distributions=self.estimator_params,
            cv=self.cv,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
            scoring=self.calc_metric.calc_make_scorer(self.measure_of_accuracy),
            verbose=self.verbose,
        )

        if self.fit_params is not None:
            self.random_search.fit(self.X, self.y, **self.fit_params)
        else:
            self.random_search.fit(self.X, self.y)
        self.best_estimator = self.random_search.best_estimator_
        return self

    def get_best_estimator(self, *args, **kwargs):
        """
        Get the best estimator after invoking fit on it.
        """
        if self.optimize.has_been_called and self.best_estimator is not None:
            return self.best_estimator
        else:
            self.optimize(
                self.estimator,
                param_distributions=self.estimator_params,
                cv=self.cv,
                n_iter=self.n_iter,
                n_jobs=self.n_jobs,
                scoring=self.calc_metric.calc_make_scorer(self.measure_of_accuracy),
                verbose=self.verbose,
            )
            if self.optimize.has_been_called and self.best_estimator is not None:
                return self.best_estimator
            else:
                raise NotImplementedError(
                    "RandomSearch has not been implemented \
                    or best_estomator is null"
                )

    def get_optimized_object(self, *args, **kwargs):
        """
        Get the best estimator after invoking fit on it.
        """
        if self.optimize.has_been_called and self.grid_search is not None:
            return self.grid_search
        else:
            raise NotImplementedError(
                "RandomSearch has not been implemented \
                or best_estomator is null"
            )
