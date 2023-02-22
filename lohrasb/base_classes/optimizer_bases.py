import subprocess

import numpy as np
import pandas as pd
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
        scoring,
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
            fit_params: dict
                Parameters passed to the fit method of the estimator.
            measure_of_accuracy : object of type make_scorer
                see documentation in
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
                Note: If scoring=None, measure_of_accuracy argument will be used to evaluate the performance,
                otherwise scoring argument will be used.
            scoring: str, callable, list, tuple or dict, default=None
                Note: If scoring=None, measure_of_accuracy argument will be used to evaluate the performance.
                Strategy to evaluate the performance of the cross-validated model on the test set.
                If scoring represents a single score, one can use:
                a single string (see The scoring parameter: defining model evaluation rules);
                a callable (see Defining your scoring strategy from metric functions) that returns a single value.
                If scoring represents multiple scores, one can use:
                a list or tuple of unique strings;
                a callable returning a dictionary where the keys are the metric names and the values are the metric scores;
                a dictionary with metric names as keys and callables a values.
                See Specifying multiple metrics for evaluation for an example.
                If None, the estimator’s score method is used.
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
        self.scoring = scoring
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
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, value):
        self._scoring = value

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
        if self.scoring is not None:
            scoring = self.scoring
        else:
            scoring = self.calc_metric.calc_make_scorer(self.measure_of_accuracy)

        logger.info(f"The optimization will be based on {scoring} metric!")

        self.grid_search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.estimator_params,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring=scoring,
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
        if self.scoring is not None:
            scoring = self.scoring
        else:
            scoring = self.calc_metric.calc_make_scorer(self.measure_of_accuracy)

        logger.info(f"The optimization will be based on {scoring} metric!")

        if self.optimize.has_been_called and self.best_estimator is not None:
            return self.best_estimator
        else:
            self.optimize(
                self.estimator,
                param_grid=self.estimator_params,
                cv=self.cv,
                n_jobs=self.n_jobs,
                scoring=scoring,
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
        scoring,
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
        fit_params: dict
            Parameters passed to the fit method of the estimator.
        measure_of_accuracy : object of type make_scorer
            see documentation in
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html.
            Note: If scoring=None, measure_of_accuracy argument will be used to evaluate the performance,
            otherwise scoring argument will be used.
        scoring: str, callable, list, tuple or dict, default=None
            Note: If scoring=None, measure_of_accuracy argument will be used to evaluate the performance.
            Strategy to evaluate the performance of the cross-validated model on the test set.
            If scoring represents a single score, one can use:
            a single string (see The scoring parameter: defining model evaluation rules);
            a callable (see Defining your scoring strategy from metric functions) that returns a single value.
            If scoring represents multiple scores, one can use:
            a list or tuple of unique strings;
            a callable returning a dictionary where the keys are the metric names and the values are the metric scores;
            a dictionary with metric names as keys and callables a values.
            See Specifying multiple metrics for evaluation for an example.
            If None, the estimator’s score method is used.
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
        self.scoring = scoring
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
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, value):
        self._scoring = value

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
        if self.scoring is not None:
            scoring = self.scoring
        else:
            scoring = self.calc_metric.calc_make_scorer(self.measure_of_accuracy)
        logger.info(f"The optimization will be based on {scoring} metric!")

        self.random_search = RandomizedSearchCV(
            self.estimator,
            param_distributions=self.estimator_params,
            cv=self.cv,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
            scoring=scoring,
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
        if self.scoring is not None:
            scoring = self.scoring
        else:
            scoring = self.calc_metric.calc_make_scorer(self.measure_of_accuracy)

        logger.info(f"The optimization will be based on {scoring} metric!")

        if self.optimize.has_been_called and self.best_estimator is not None:
            return self.best_estimator
        else:
            self.optimize(
                self.estimator,
                param_distributions=self.estimator_params,
                cv=self.cv,
                n_iter=self.n_iter,
                n_jobs=self.n_jobs,
                scoring=scoring,
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
        if self.optimize.has_been_called and self.random_search is not None:
            return self.random_search
        else:
            raise NotImplementedError(
                "RandomSearch has not been implemented \
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
        estimator,
        estimator_params,
        fit_params,
        measure_of_accuracy,
        verbose,
        early_stopping,
        scoring,
        n_jobs,
        cv,
        refit,
        error_score,
        return_train_score,
        local_dir,
        name,
        max_iters,
        use_gpu,
        loggers,
        pipeline_auto_early_stop,
        stopper,
        time_budget_s,
        mode,
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
        measure_of_accuracy : object of type make_scorer
            see documentation in
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        early_stopping: (bool, str or TrialScheduler, optional)
            Option to stop fitting to a hyperparameter configuration if it performs poorly. Possible inputs are:
            If True, defaults to ASHAScheduler. A string corresponding to the name of a Tune Trial Scheduler (i.e.,
            “ASHAScheduler”). To specify parameters of the scheduler, pass in a scheduler object instead of a string.
            Scheduler for executing fit with early stopping. Only a subset of schedulers are currently supported.
            The scheduler will only be used if the estimator supports partial fitting If None or False,
            early stopping will not be used.
        scoring : str, list/tuple, dict, or None)
            A single string or a callable to evaluate the predictions on the test set.
            See https://scikit-learn.org/stable/modules/model_evaluation.html #scoring-parameter
            for all options. For evaluating multiple metrics, either give a list/tuple of (unique)
            strings or a dict with names as keys and callables as values. If None, the estimator’s
            score method is used. Defaults to None.
        n_jobs : int
            Number of jobs to run in parallel. None or -1 means using all processors. Defaults to None.
            If set to 1, jobs will be run using Ray’s ‘local mode’. This can lead to significant speedups
            if the model takes < 10 seconds to fit due to removing inter-process communication overheads.
        cv : int, cross-validation generator or iterable :
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
            None, to use the default 5-fold cross validation, integer, to specify the number
            of folds in a (Stratified)KFold, An iterable yielding (train, test) splits as arrays
            of indices. For integer/None inputs, if the estimator is a classifier and y is either
            binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
            Defaults to None.
        refit : bool or str
            Refit an estimator using the best found parameters on the whole dataset.
            For multiple metric evaluation, this needs to be a string denoting the scorer
            that would be used to find the best parameters for refitting the estimator at the end.
            The refitted estimator is made available at the best_estimator_ attribute and permits using predict
            directly on this GridSearchCV instance. Also for multiple metric evaluation,
            the attributes best_index_, best_score_ and best_params_ will only be available if
            refit is set and all of them will be determined w.r.t this specific scorer.
            If refit not needed, set to False. See scoring parameter to know more about multiple
            metric evaluation. Defaults to True.
        verbose : int
            Controls the verbosity: 0 = silent, 1 = only status updates, 2 = status and trial results.
            Defaults to 0.
        error_score : 'raise' or int or float
            Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’,
            the error is raised. If a numeric value is given, FitFailedWarning is raised. This parameter
            does not affect the refit step, which will always raise the error. Defaults to np.nan.
        return_train_score :bool
            If False, the cv_results_ attribute will not include training scores. Defaults to False.
            Computing training scores is used to get insights on how different parameter settings
            impact the overfitting/underfitting trade-off. However computing the scores on the training
            set can be computationally expensive and is not strictly required to select the parameters
            that yield the best generalization performance.
        local_dir : str
            A string that defines where checkpoints will be stored. Defaults to “~/ray_results”.
        name : str
            Name of experiment (for Ray Tune)
        max_iters : int
            Indicates the maximum number of epochs to run for each hyperparameter configuration sampled.
            This parameter is used for early stopping. Defaults to 1. Depending on the classifier
            type provided, a resource parameter (resource_param = max_iter or n_estimators)
            will be detected. The value of resource_param will be treated as a “max resource value”,
            and all classifiers will be initialized with max resource value // max_iters, where max_iters
            is this defined parameter. On each epoch, resource_param (max_iter or n_estimators) is
            incremented by max resource value // max_iters.
        use_gpu : bool
            Indicates whether to use gpu for fitting. Defaults to False. If True, training will start
            processes with the proper CUDA VISIBLE DEVICE settings set. If a Ray cluster has been initialized,
            all available GPUs will be used.
        loggers : list
            A list of the names of the Tune loggers as strings to be used to log results. Possible
            values are “tensorboard”, “csv”, “mlflow”, and “json”
        pipeline_auto_early_stop : bool
            Only relevant if estimator is Pipeline object and early_stopping is enabled/True. If
            True, early stopping will be performed on the last stage of the pipeline (which must
            support early stopping). If False, early stopping will be determined by
            ‘Pipeline.warm_start’ or ‘Pipeline.partial_fit’ capabilities, which are by default
            not supported by standard SKlearn. Defaults to True.
        stopper : ray.tune.stopper.Stopper
            Stopper objects passed to tune.run().
        time_budget_s : |float|datetime.timedelta
            Global time budget in seconds after which all trials are stopped. Can also be a
            datetime.timedelta object.
        mode : str
            One of {min, max}. Determines whether objective is minimizing or maximizing the
            metric attribute. Defaults to “max”.

        Return
        ----------

        The best estimator of estimator optimized by TuneGridSearch.

        Methods
        -------
        prepare_data()
            Prepare data to be consumed by TuneGridSearch.Pass for TuneGridSearch case.
        optimize()
            Optimize estimator using TuneGridSearch engine.
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
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.refit = refit
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.local_dir = local_dir
        self.name = name
        self.max_iters = max_iters
        self.use_gpu = use_gpu
        self.loggers = loggers
        self.pipeline_auto_early_stop = pipeline_auto_early_stop
        self.stopper = stopper
        self.time_budget_s = time_budget_s
        self.mode = mode

        self.tune_gridsearch = None
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
    def early_stopping(self):
        return self._early_stopping

    @early_stopping.setter
    def early_stopping(self, value):
        self._early_stopping = value

    @property
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, value):
        self._scoring = value

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
    def refit(self):
        return self._refit

    @refit.setter
    def refit(self, value):
        self._refit = value

    @property
    def error_score(self):
        return self._error_score

    @error_score.setter
    def error_score(self, value):
        self._error_score = value

    @property
    def return_train_score(self):
        return self._return_train_score

    @return_train_score.setter
    def return_train_score(self, value):
        self._return_train_score = value

    @property
    def local_dir(self):
        return self._local_dir

    @local_dir.setter
    def local_dir(self, value):
        self._local_dir = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def max_iters(self):
        return self._max_iters

    @max_iters.setter
    def max_iters(self, value):
        self._max_iters = value

    @property
    def use_gpu(self):
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, value):
        self._use_gpu = value

    @property
    def loggers(self):
        return self._loggers

    @loggers.setter
    def loggers(self, value):
        self._loggers = value

    @property
    def pipeline_auto_early_stop(self):
        return self._pipeline_auto_early_stop

    @pipeline_auto_early_stop.setter
    def pipeline_auto_early_stop(self, value):
        self._pipeline_auto_early_stop = value

    @property
    def stopper(self):
        return self._stopper

    @stopper.setter
    def stopper(self, value):
        self._stopper = value

    @property
    def time_budget_s(self):
        return self._time_budget_s

    @time_budget_s.setter
    def time_budget_s(self, value):
        self._time_budget_s = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    @property
    def tune_gridsearch(self):
        return self._tune_gridsearch

    @tune_gridsearch.setter
    def tune_gridsearch(self, value):
        self._tune_gridsearch = value

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
        Optimize estimator using TuneGridSearch engine.
        """
        self.tune_gridsearch = TuneGridSearchCV(
            estimator=self.estimator,
            param_grid=self.estimator_params,
            early_stopping=self.early_stopping,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv,
            refit=self.refit,
            verbose=self.verbose,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
            local_dir=self.local_dir,
            name=self.name,
            max_iters=self.max_iters,
            use_gpu=self.use_gpu,
            loggers=self.loggers,
            pipeline_auto_early_stop=self.pipeline_auto_early_stop,
            stopper=self.stopper,
            time_budget_s=self.time_budget_s,
            mode=self.mode,
        )

        if self.fit_params is not None:
            self.tune_gridsearch.fit(self.X, self.y, **self.fit_params)
        else:
            self.tune_gridsearch.fit(self.X, self.y)
        self.best_estimator = self.tune_gridsearch.best_estimator_
        return self

    def get_best_estimator(self, *args, **kwargs):
        """
        Get the best estimator after invoking fit on it.
        """

        if self.scoring is not None:
            scoring = self.scoring
        else:
            scoring = self.calc_metric.calc_make_scorer(self.measure_of_accuracy)
        logger.info(f"The optimization will be based on {scoring} metric!")

        if self.optimize.has_been_called and self.best_estimator is not None:
            return self.best_estimator
        else:
            self.optimize(
                estimator=self.estimator,
                param_grid=self.estimator_params,
                early_stopping=self.early_stopping,
                n_jobs=self.n_jobs,
                cv=self.cv,
                refit=self.refit,
                verbose=self.verbose,
                error_score=self.error_score,
                return_train_score=self.return_train_score,
                local_dir=self.local_dir,
                name=self.name,
                max_iters=self.max_iters,
                use_gpu=self.use_gpu,
                loggers=self.loggers,
                pipeline_auto_early_stop=self.pipeline_auto_early_stop,
                stopper=self.stopper,
                time_budget_s=self.time_budget_s,
                mode=self.mode,
                scoring=scoring,
            )
            if self.optimize.has_been_called and self.best_estimator is not None:
                return self.best_estimator
            else:
                raise NotImplementedError(
                    "TuneGridSearchCV has not been implemented \
                    or best_estomator is null"
                )

    def get_optimized_object(self, *args, **kwargs):
        """
        Get the best estimator after invoking fit on it.
        """
        if self.optimize.has_been_called and self.tune_gridsearch is not None:
            return self.tune_gridsearch
        else:
            raise NotImplementedError(
                "TuneGridSearchCV has not been implemented \
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
        estimator,
        estimator_params,
        fit_params,
        measure_of_accuracy,
        verbose,
        early_stopping,
        scoring,
        n_jobs,
        cv,
        n_trials,
        refit,
        random_state,
        error_score,
        return_train_score,
        local_dir,
        name,
        max_iters,
        search_optimization,
        use_gpu,
        loggers,
        pipeline_auto_early_stop,
        stopper,
        time_budget_s,
        mode,
        search_kwargs,
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
        measure_of_accuracy : object of type make_scorer
            see documentation in
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        early_stopping: (bool, str or TrialScheduler, optional)
            Option to stop fitting to a hyperparameter configuration if it performs poorly. Possible inputs are:
            If True, defaults to ASHAScheduler. A string corresponding to the name of a Tune Trial Scheduler (i.e.,
            “ASHAScheduler”). To specify parameters of the scheduler, pass in a scheduler object instead of a string.
            Scheduler for executing fit with early stopping. Only a subset of schedulers are currently supported.
            The scheduler will only be used if the estimator supports partial fitting If None or False,
            early stopping will not be used.
        scoring : str, list/tuple, dict, or None)
            A single string or a callable to evaluate the predictions on the test set.
            See https://scikit-learn.org/stable/modules/model_evaluation.html #scoring-parameter
            for all options. For evaluating multiple metrics, either give a list/tuple of (unique)
            strings or a dict with names as keys and callables as values. If None, the estimator’s
            score method is used. Defaults to None.
        n_jobs : int
            Number of jobs to run in parallel. None or -1 means using all processors. Defaults to None.
            If set to 1, jobs will be run using Ray’s ‘local mode’. This can lead to significant speedups
            if the model takes < 10 seconds to fit due to removing inter-process communication overheads.
        cv : int, cross-validation generator or iterable :
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
            None, to use the default 5-fold cross validation, integer, to specify the number
            of folds in a (Stratified)KFold, An iterable yielding (train, test) splits as arrays
            of indices. For integer/None inputs, if the estimator is a classifier and y is either
            binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
            Defaults to None.
        n_trials : int
            Number of parameter settings that are sampled. n_trials trades off runtime vs
            quality of the solution. Defaults to 10.
        refit : bool or str
            Refit an estimator using the best found parameters on the whole dataset.
            For multiple metric evaluation, this needs to be a string denoting the scorer
            that would be used to find the best parameters for refitting the estimator at the end.
            The refitted estimator is made available at the best_estimator_ attribute and permits using predict
            directly on this GridSearchCV instance. Also for multiple metric evaluation,
            the attributes best_index_, best_score_ and best_params_ will only be available if
            refit is set and all of them will be determined w.r.t this specific scorer.
            If refit not needed, set to False. See scoring parameter to know more about multiple
            metric evaluation. Defaults to True.
        random_state:int or RandomState:
            Pseudo random number generator state used for random uniform sampling from lists
             of possible values instead of scipy.stats distributions. If int, random_state
             is the seed used by the random number generator; If RandomState instance, a seed
              is sampled from random_state; If None, the random number generator is the RandomState
               instance used by np.random and no seed is provided. Defaults to None. Ignored when
               using BOHB.
        verbose : int
            Controls the verbosity: 0 = silent, 1 = only status updates, 2 = status and trial results.
            Defaults to 0.
        error_score : 'raise' or int or float
            Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’,
            the error is raised. If a numeric value is given, FitFailedWarning is raised. This parameter
            does not affect the refit step, which will always raise the error. Defaults to np.nan.
        return_train_score :bool
            If False, the cv_results_ attribute will not include training scores. Defaults to False.
            Computing training scores is used to get insights on how different parameter settings
            impact the overfitting/underfitting trade-off. However computing the scores on the training
            set can be computationally expensive and is not strictly required to select the parameters
            that yield the best generalization performance.
        local_dir : str
            A string that defines where checkpoints will be stored. Defaults to “~/ray_results”.
        name : str
            Name of experiment (for Ray Tune)
        max_iters : int
            Indicates the maximum number of epochs to run for each hyperparameter configuration sampled.
            This parameter is used for early stopping. Defaults to 1. Depending on the classifier
            type provided, a resource parameter (resource_param = max_iter or n_estimators)
            will be detected. The value of resource_param will be treated as a “max resource value”,
            and all classifiers will be initialized with max resource value // max_iters, where max_iters
            is this defined parameter. On each epoch, resource_param (max_iter or n_estimators) is
            incremented by max resource value // max_iters.
        search_optimization: "hyperopt" (search_optimization ("random" or "bayesian" or "bohb" or
        “optuna” or ray.tune.search.Searcher instance): Randomized search is invoked with
        search_optimization set to "random" and behaves like scikit-learn’s RandomizedSearchCV.
            Bayesian search can be invoked with several values of search_optimization.
            "bayesian" via https://scikit-optimize.github.io/stable/
            "bohb" via http://github.com/automl/HpBandSter
            Tree-Parzen Estimators search is invoked with search_optimization set to "hyperopt"
            via HyperOpt: http://hyperopt.github.io/hyperopt
            All types of search aside from Randomized search require parent libraries to be installed.
            Alternatively, instead of a string, a Ray Tune Searcher instance can be used, which
            will be passed to tune.run().
        use_gpu : bool
            Indicates whether to use gpu for fitting. Defaults to False. If True, training will start
            processes with the proper CUDA VISIBLE DEVICE settings set. If a Ray cluster has been initialized,
            all available GPUs will be used.
        loggers : list
            A list of the names of the Tune loggers as strings to be used to log results. Possible
            values are “tensorboard”, “csv”, “mlflow”, and “json”
        pipeline_auto_early_stop : bool
            Only relevant if estimator is Pipeline object and early_stopping is enabled/True. If
            True, early stopping will be performed on the last stage of the pipeline (which must
            support early stopping). If False, early stopping will be determined by
            ‘Pipeline.warm_start’ or ‘Pipeline.partial_fit’ capabilities, which are by default
            not supported by standard SKlearn. Defaults to True.
        stopper : ray.tune.stopper.Stopper
            Stopper objects passed to tune.run().
        time_budget_s : |float|datetime.timedelta
            Global time budget in seconds after which all trials are stopped. Can also be a
            datetime.timedelta object.
        mode : str
            One of {min, max}. Determines whether objective is minimizing or maximizing the
            metric attribute. Defaults to “max”.
        search_kwargs : dict
            Additional arguments to pass to the SearchAlgorithms (tune.suggest) objects.

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

        self.X = X
        self.y = y
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.fit_params = fit_params
        self.measure_of_accuracy = measure_of_accuracy
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.n_trials = n_trials
        self.refit = refit
        self.random_state = random_state
        self.search_optimization = search_optimization
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.local_dir = local_dir
        self.name = name
        self.max_iters = max_iters
        self.use_gpu = use_gpu
        self.loggers = loggers
        self.pipeline_auto_early_stop = pipeline_auto_early_stop
        self.stopper = stopper
        self.time_budget_s = time_budget_s
        self.mode = mode
        self.search_kwargs = search_kwargs

        self.tune_search = None
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
    def early_stopping(self):
        return self._early_stopping

    @early_stopping.setter
    def early_stopping(self, value):
        self._early_stopping = value

    @property
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, value):
        self._scoring = value

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
    def n_trials(self):
        return self._n_trials

    @n_trials.setter
    def n_trials(self, value):
        self._n_trials = value

    @property
    def refit(self):
        return self._refit

    @refit.setter
    def refit(self, value):
        self._refit = value

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        self._random_state = value

    @property
    def search_optimization(self):
        return self._search_optimization

    @search_optimization.setter
    def search_optimization(self, value):
        self._search_optimization = value

    @property
    def error_score(self):
        return self._error_score

    @error_score.setter
    def error_score(self, value):
        self._error_score = value

    @property
    def return_train_score(self):
        return self._return_train_score

    @return_train_score.setter
    def return_train_score(self, value):
        self._return_train_score = value

    @property
    def local_dir(self):
        return self._local_dir

    @local_dir.setter
    def local_dir(self, value):
        self._local_dir = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def max_iters(self):
        return self._max_iters

    @max_iters.setter
    def max_iters(self, value):
        self._max_iters = value

    @property
    def use_gpu(self):
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, value):
        self._use_gpu = value

    @property
    def loggers(self):
        return self._loggers

    @loggers.setter
    def loggers(self, value):
        self._loggers = value

    @property
    def pipeline_auto_early_stop(self):
        return self._pipeline_auto_early_stop

    @pipeline_auto_early_stop.setter
    def pipeline_auto_early_stop(self, value):
        self._pipeline_auto_early_stop = value

    @property
    def stopper(self):
        return self._stopper

    @stopper.setter
    def stopper(self, value):
        self._stopper = value

    @property
    def time_budget_s(self):
        return self._time_budget_s

    @time_budget_s.setter
    def time_budget_s(self, value):
        self._time_budget_s = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    @property
    def search_kwargs(self):
        return self._search_kwargs

    @search_kwargs.setter
    def search_kwargs(self, value):
        self._search_kwargs = value

    @property
    def tune_search(self):
        return self._tune_search

    @tune_search.setter
    def tune_search(self, value):
        self._tune_search = value

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
        Optimize estimator using TuneSearchCV engine.
        """

        if self.scoring is not None:
            scoring = self.scoring
        else:
            scoring = self.calc_metric.calc_make_scorer(self.measure_of_accuracy)
        logger.info(f"The optimization will be based on {scoring} metric!")

        self.tune_search = TuneSearchCV(
            estimator=self.estimator,
            param_distributions=self.estimator_params,
            early_stopping=self.early_stopping,
            scoring=scoring,
            n_jobs=self.n_jobs,
            cv=self.cv,
            n_trials=self.n_trials,
            refit=self.refit,
            random_state=self.random_state,
            verbose=self.verbose,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
            local_dir=self.local_dir,
            name=self.name,
            max_iters=self.max_iters,
            search_optimization=self.search_optimization,
            use_gpu=self.use_gpu,
            loggers=self.loggers,
            pipeline_auto_early_stop=self.pipeline_auto_early_stop,
            stopper=self.stopper,
            time_budget_s=self.time_budget_s,
            mode=self.mode,
            search_kwargs=self.search_kwargs,
        )

        if self.fit_params is not None:
            self.tune_search.fit(self.X, self.y, **self.fit_params)
        else:
            self.tune_search.fit(self.X, self.y)
        self.best_estimator = self.tune_search.best_estimator_
        return self

    def get_best_estimator(self, *args, **kwargs):
        """
        Get the best estimator after invoking fit on it.
        """
        if self.scoring is not None:
            scoring = self.scoring
        else:
            scoring = self.calc_metric.calc_make_scorer(self.measure_of_accuracy)

        if self.optimize.has_been_called and self.best_estimator is not None:
            return self.best_estimator
        else:
            self.optimize(
                estimator=self.estimator,
                param_grid=self.estimator_params,
                early_stopping=self.early_stopping,
                n_jobs=self.n_jobs,
                cv=self.cv,
                n_trials=self.n_trials,
                refit=self.refit,
                verbose=self.verbose,
                random_state=self.random_state,
                error_score=self.error_score,
                return_train_score=self.return_train_score,
                local_dir=self.local_dir,
                name=self.name,
                max_iters=self.max_iters,
                search_optimization=self.search_optimization,
                use_gpu=self.use_gpu,
                loggers=self.loggers,
                pipeline_auto_early_stop=self.pipeline_auto_early_stop,
                stopper=self.stopper,
                time_budget_s=self.time_budget_s,
                mode=self.mode,
                scoring=scoring,
                search_kwargs=self.search_kwargs,
            )
            if self.optimize.has_been_called and self.best_estimator is not None:
                return self.best_estimator
            else:
                raise NotImplementedError(
                    "TuneSearchCV has not been implemented \
                    or best_estomator is null"
                )

    def get_optimized_object(self, *args, **kwargs):
        """
        Get the best estimator after invoking fit on it.
        """
        if self.optimize.has_been_called and self.tune_gridsearch is not None:
            return self.tune_gridsearch
        else:
            raise NotImplementedError(
                "TuneSearchCV has not been implemented \
                or best_estomator is null"
            )
