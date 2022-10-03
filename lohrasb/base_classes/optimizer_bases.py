import subprocess

import numpy as np
from catboost import *
from imblearn.ensemble import *
from lightgbm import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import *
from sklearn.svm import *
from xgboost import *

from lohrasb.abstracts.optimizers import OptimizerABC
from lohrasb.decorators.decorators import trackcalls
from lohrasb.factories.factories import OptimizerFactory
from lohrasb.utils.helper_funcs import _trail_params_retrive  # maping_mesurements,
from lohrasb.utils.metrics import CalcMetrics


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
        # grid search and random search
        measure_of_accuracy,
        add_extra_args_for_measure_of_accuracy,
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
            add_extra_args_for_measure_of_accuracy : boolean
                True if the user wants to add extra arguments for measure_of_accuracy
                False otherwise.
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
        """

        self.X = X
        self.y = y
        self.verbose = verbose
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        # grid search and random search
        self.measure_of_accuracy = measure_of_accuracy
        self.add_extra_args_for_measure_of_accuracy = (
            add_extra_args_for_measure_of_accuracy
        )
        self.n_jobs = n_jobs
        # optuna params
        self.test_size = test_size
        self.with_stratified = with_stratified
        # number_of_trials=100,
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
        self.calc_metric = CalcMetrics(
            y_true=None,
            y_pred=None,
            metric=None,
        )
        self.func_str = None
        self.func_params = None

    def prepare_data(self):
        """
        Prepare data to be consumed by the optimizer.
        """

        if self.with_stratified:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=self.test_size,
                stratify=self.y[self.y.columns.to_list()[0]],
                random_state=self.random_state,
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )

        # prepare metrics arguments
        self.calc_metric.metric = self.measure_of_accuracy
        self.calc_metric.change_default_args_of_metric = (
            self.add_extra_args_for_measure_of_accuracy
        )
        self.func_str = self.calc_metric.get_func_string_if_any()
        self.func_params = self.calc_metric.get_default_params_if_any()

        return self

    def optimize(self):
        """
        Optimize estimator using Optuna engine.
        """

        def objective(trial):

            params = _trail_params_retrive(trial, self.estimator_params)
            print(params)
            est = eval(
                self.estimator.__class__.__name__
                + "(**params)"
                + ".fit(self.X_train, self.y_train)"
            )
            preds = est.predict(self.X_test)
            pred_labels = np.rint(preds)

            if self.measure_of_accuracy in [
                "accuracy_score",
                "auc",
                "precision_recall_curve",
                "balanced_accuracy_score",
                "cohen_kappa_score",
                "dcg_score",
                "det_curve",
                "f1_score",
                "hamming_loss",
                "fbeta_score",
                "jaccard_score",
                "matthews_corrcoef",
                "ndcg_score",
                "precision_score",
                "recall_score",
                "roc_auc_score",
                "roc_curve",
                "top_k_accuracy_score",
                "zero_one_loss",
                # customs
                "tn",
                "tp",
                "tn_score",
                "tp_score",
                "f1_plus_tp",
                "f1_plus_tn",
                "specificity",
                "roc_plus_f1",
                "auc_plus_f1",
                "precision_recall_curve",
                "precision_recall_fscore_support",
            ]:
                self.calc_metric.y_pred = pred_labels
                self.calc_metric.y_true = self.y_test
                metric = self.calc_metric.metric
                y_true = self.calc_metric.y_true
                y_pred = self.calc_metric.y_pred
                params = self.func_params
                accr = self.calc_metric.get_simple_metric(
                    metric, y_true, y_pred, params
                )
            elif self.measure_of_accuracy in [
                "explained_variance_score",
                "max_error",
                "mean_absolute_error",
                "mean_squared_log_error",
                "mean_absolute_percentage_error",
                "mean_squared_log_error",
                "median_absolute_error",
                "mean_absolute_percentage_error",
                "r2_score",
                "mean_poisson_deviance",
                "mean_gamma_deviance",
                "mean_tweedie_deviance",
                "d2_tweedie_score",
                "mean_pinball_loss",
                "d2_pinball_score",
                "d2_absolute_error_score",
            ]:
                self.calc_metric.y_pred = preds
                self.calc_metric.y_true = self.y_test
                print(self.calc_metric.metric)
                metric = self.calc_metric.metric
                y_true = self.calc_metric.y_true
                y_pred = self.calc_metric.y_pred
                params = self.func_params
                accr = self.calc_metric.get_simple_metric(
                    metric, y_true, y_pred, params
                )

            return accr

        # study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
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
        measure_of_accuracy,
        add_extra_args_for_measure_of_accuracy,
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

            add_extra_args_for_measure_of_accuracy : boolean
                True if the user wants to add extra arguments for measure_of_accuracy
                False otherwise.

            verbose: int
                Controls the verbosity across all objects: the higher, the more messages.
            n_jobs: int
                The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
                ``-1`` means using all processors. (default -1)
        Return
        ----------

        The best estimator of estimator optimized by GridSearchCV.

        """
        self.X = X
        self.y = y
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.measure_of_accuracy = measure_of_accuracy
        self.add_extra_args_for_measure_of_accuracy = (
            add_extra_args_for_measure_of_accuracy
        )
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
        self.calc_metric.change_default_args_of_metric = (
            self.add_extra_args_for_measure_of_accuracy
        )

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
            self.estimator,
            param_grid=self.estimator_params,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring=self.calc_metric.calc_make_scorer(self.measure_of_accuracy),
            verbose=self.verbose,
        )
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
            self.best_estimator, self.random_search = self.optimize(
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
        return False

    def get_optimized_object(self, *args, **kwargs):
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
        measure_of_accuracy,
        add_extra_args_for_measure_of_accuracy,
        verbose,
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

        add_extra_args_for_measure_of_accuracy : boolean
            True if the user wants to add extra arguments for measure_of_accuracy
            False otherwise.

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

        """

        self.X = X
        self.y = y
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.measure_of_accuracy = measure_of_accuracy
        self.add_extra_args_for_measure_of_accuracy = (
            add_extra_args_for_measure_of_accuracy
        )
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.cv = cv
        self.random_search = None
        self.best_estimator = None
        self.calc_metric = CalcMetrics(
            y_true=y,
            y_pred=None,
            metric=self.measure_of_accuracy,
        )
        self.calc_metric.change_default_args_of_metric = (
            self.add_extra_args_for_measure_of_accuracy
        )

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
            self.best_estimator, self.random_search = self.optimize(
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
        return False

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


class GridSearchFactory(OptimizerFactory):
    """Factory for building GridSeachCv."""

    def __init__(
        self,
        X,
        y,
        estimator,
        estimator_params,
        measure_of_accuracy,
        add_extra_args_for_measure_of_accuracy,
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

        add_extra_args_for_measure_of_accuracy : boolean
            True if the user wants to add extra arguments for measure_of_accuracy
            False otherwise.

        verbose: int
            Controls the verbosity across all objects: the higher, the more messages.
        n_jobs: int
            The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
            ``-1`` means using all processors. (default -1)
        """
        self.X = X
        self.y = y
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.measure_of_accuracy = measure_of_accuracy
        self.add_extra_args_for_measure_of_accuracy = (
            add_extra_args_for_measure_of_accuracy
        )
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.cv = cv

    def optimizer_builder(self):
        print("Initializing GridSEarchCV")
        return GridSearch(
            self.X,
            self.y,
            self.estimator,
            self.estimator_params,
            self.measure_of_accuracy,
            self.add_extra_args_for_measure_of_accuracy,
            self.verbose,
            self.n_jobs,
            self.cv,
        )


class OptunaFactory(OptimizerFactory):
    """Factory for building Optuna engine."""

    def __init__(
        self,
        X,
        y,
        verbose,
        random_state,
        estimator,
        estimator_params,
        # grid search and random search
        measure_of_accuracy,
        add_extra_args_for_measure_of_accuracy,
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

            add_extra_args_for_measure_of_accuracy : boolean
                True if the user wants to add extra arguments for measure_of_accuracy
                False otherwise.

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
        """

        self.X = X
        self.y = y
        self.verbose = verbose
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        # grid search and random search
        self.measure_of_accuracy = measure_of_accuracy
        self.add_extra_args_for_measure_of_accuracy = (
            add_extra_args_for_measure_of_accuracy
        )
        self.n_jobs = n_jobs
        # optuna params
        self.test_size = test_size
        self.with_stratified = with_stratified
        # number_of_trials=100,
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

    def optimizer_builder(self):
        """
        Return a OptunaSearch instance.

        """
        print("Initializing Optuna")
        return OptunaSearch(
            self.X,
            self.y,
            self.verbose,
            self.random_state,
            self.estimator,
            self.estimator_params,
            # grid search and random search
            self.measure_of_accuracy,
            self.add_extra_args_for_measure_of_accuracy,
            self.n_jobs,
            # optuna params
            self.test_size,
            self.with_stratified,
            # number_of_trials=100,
            # optuna study init params
            self.study,
            # optuna optimization params
            self.study_optimize_objective,
            self.study_optimize_objective_n_trials,
            self.study_optimize_objective_timeout,
            self.study_optimize_n_jobs,
            self.study_optimize_catch,
            self.study_optimize_callbacks,
            self.study_optimize_gc_after_trial,
            self.study_optimize_show_progress_bar,
        )


class RandomSearchFactory(OptimizerFactory):
    """Factory for building GridSeachCv."""

    def __init__(
        self,
        X,
        y,
        estimator,
        estimator_params,
        measure_of_accuracy,
        add_extra_args_for_measure_of_accuracy,
        verbose,
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

        add_extra_args_for_measure_of_accuracy : boolean
            True if the user wants to add extra arguments for measure_of_accuracy
            False otherwise.

        verbose: int
            Controls the verbosity across all objects: the higher, the more messages.
        n_jobs: int
            The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
            ``-1`` means using all processors. (default -1)
        n_iter : int
            Only it means full in Random Search. It is several parameter
            settings that are sampled. n_iter trades off runtime vs. quality of the solution.
        """

        self.X = X
        self.y = y
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.measure_of_accuracy = measure_of_accuracy
        self.add_extra_args_for_measure_of_accuracy = (
            add_extra_args_for_measure_of_accuracy
        )
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.cv = cv

    def optimizer_builder(self):
        """
        Return a RandomSeachCV instance.

        """

        print("Initializing RandomSeachCV")
        return RandomSearch(
            self.X,
            self.y,
            self.estimator,
            self.estimator_params,
            self.measure_of_accuracy,
            self.add_extra_args_for_measure_of_accuracy,
            self.verbose,
            self.n_jobs,
            self.n_iter,
            self.cv,
        )
