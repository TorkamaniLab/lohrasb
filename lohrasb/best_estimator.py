import logging
from abc import ABCMeta
from pickletools import optimize

import numpy as np
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator

from lohrasb.abstracts.estimators import AbstractEstimator
from lohrasb.base_classes.optimizer_bases import GridSearch, OptunaSearch, RandomSearch


class OptunaBestEstimator(AbstractEstimator):
    """
    BestModel estimation using optuna optimization.
    ...

    Parameters
    ----------

    estimator: object
        An unfitted estimator that has fit and predicts methods.
    estimator_params: dict
        Parameters were passed to find the best estimator using the optimization
        method.
    hyper_parameter_optimization_method : str
        Use ``optuna`` to set for using Optuna.
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
        Get optuna trail object after optimization.
    Notes
    -----
    It is recommended to use available factories
    to create a new instance of this class.

    """

    def __init__(
        self,
        # general argument setting
        hyper_parameter_optimization_method="optuna",
        verbose=None,
        random_state=None,
        estimator=None,
        estimator_params=None,
        # grid search and random search
        measure_of_accuracy=None,
        add_extra_args_for_measure_of_accuracy=None,
        n_jobs=None,
        with_stratified=None,
        test_size=None,
        # optuna study init params
        study=None,
        # optuna optimization params
        study_optimize_objective=None,
        study_optimize_objective_n_trials=None,
        study_optimize_objective_timeout=None,
        study_optimize_n_jobs=None,
        study_optimize_catch=None,
        study_optimize_callbacks=None,
        study_optimize_gc_after_trial=None,
        study_optimize_show_progress_bar=None,
    ):
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
        self.verbose = verbose
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.measure_of_accuracy = measure_of_accuracy
        self.add_extra_args_for_measure_of_accuracy = (
            add_extra_args_for_measure_of_accuracy
        )
        self.n_jobs = n_jobs
        # optuna params
        self.with_stratified = with_stratified
        self.test_size = test_size
        # number_of_trials=100,
        # optuna study init params
        self.study = study
        # optuna optimization params
        self.study_optimize_objective = study_optimize_objective
        self.study_optimize_show_progress_bar = study_optimize_show_progress_bar
        self.study_optimize_objective_n_trials = study_optimize_objective_n_trials
        self.study_optimize_objective_timeout = study_optimize_objective_timeout
        self.study_optimize_n_jobs = study_optimize_n_jobs
        self.study_optimize_catch = study_optimize_catch
        self.study_optimize_callbacks = study_optimize_callbacks
        self.study_optimize_gc_after_trial = study_optimize_gc_after_trial
        self.best_estimator = None
        self.search_optimization = None
        self.optimized_object = None

    @property
    def best_estimator(self):
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, value):
        self._best_estimator = value

    @property
    def search_optimization(self):
        return self._search_optimization

    @search_optimization.setter
    def search_optimization(self, value):
        self._search_optimization = value

    @property
    def optimized_object(self):
        return self._optimized_object

    @optimized_object.setter
    def optimized_object(self, value):
        self._optimized_object = value

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
    def hyper_parameter_optimization_method(self):
        return self._hyper_parameter_optimization_method

    @hyper_parameter_optimization_method.setter
    def hyper_parameter_optimization_method(self, value):
        if value.lower() == "optuna":
            self._hyper_parameter_optimization_method = value
        else:
            raise ValueError(
                f"error occures during selecting optimization_method, {value} is \
                     not supported. The omptimizing engine should be \
                     optuna, grid or random."
            )

    @property
    def measure_of_accuracy(self):
        return self._measure_of_accuracy

    @measure_of_accuracy.setter
    def measure_of_accuracy(self, value):
        self._measure_of_accuracy = value

    @property
    def add_extra_args_for_measure_of_accuracy(self):
        return self._add_extra_args_for_measure_of_accuracy

    @add_extra_args_for_measure_of_accuracy.setter
    def add_extra_args_for_measure_of_accuracy(self, value):
        self._add_extra_args_for_measure_of_accuracy = value

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    @property
    def with_stratified(self):
        return self._with_stratified

    @with_stratified.setter
    def with_stratified(self, value):
        self._with_stratified = value

    @property
    def test_size(self):
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        self._test_size = value

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
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    @property
    def number_of_trials(self):
        return self._number_of_trials

    @number_of_trials.setter
    def number_of_trials(self, value):
        self._number_of_trials = value

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, value):
        self._sampler = value

    @property
    def pruner(self):
        return self._pruner

    @pruner.setter
    def pruner(self, value):
        self._pruner = value

    @property
    def best_estimator(self):
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, value):
        self._best_estimator = value

    def fit(self, X, y):
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
        self.cols = X.columns
        self.search_optimization = OptunaSearch(
            X,
            y,
            verbose=self.verbose,
            random_state=self.random_state,
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            # grid search and random search
            measure_of_accuracy=self.measure_of_accuracy,
            add_extra_args_for_measure_of_accuracy=self.add_extra_args_for_measure_of_accuracy,
            n_jobs=self.n_jobs,
            # optuna params
            test_size=self.test_size,
            with_stratified=self.with_stratified,
            # number_of_trials=100,
            # optuna study init params
            study=self.study,
            # optuna optimization params
            study_optimize_objective=self.study_optimize_objective,
            study_optimize_show_progress_bar=self.study_optimize_show_progress_bar,
            study_optimize_objective_n_trials=self.study_optimize_objective_n_trials,
            study_optimize_objective_timeout=self.study_optimize_objective_timeout,
            study_optimize_n_jobs=self.study_optimize_n_jobs,
            study_optimize_catch=self.study_optimize_catch,
            study_optimize_callbacks=self.study_optimize_callbacks,
            study_optimize_gc_after_trial=self.study_optimize_gc_after_trial,
        )
        self.optimized_object = self.search_optimization.prepare_data().optimize()
        self.best_estimator = self.optimized_object.get_best_estimator()

    def get_optimized_object(self):
        """
        Get optuna trail object after optimization.
        """
        return self.optimized_object.study

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
        """Predict class probabilities using the best estimator model.
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


class GridBestEstimator(AbstractEstimator):
    """
    BestModel estimation using gridseachcv optimization.
    ...

    Parameters
    ----------
    estimator: object
        An unfitted estimator that has fit and predicts methods.
    estimator_params: dict
        Parameters were passed to find the best estimator using the optimization
        method.
    hyper_parameter_optimization_method : str
        Use ``grid`` to set for Grid Search.
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

    cv: int
        cross-validation generator or an iterable.
        Determines the cross-validation splitting strategy. Possible inputs
        for cv are: None, to use the default 5-fold cross-validation,
        int, to specify the number of folds in a (Stratified)KFold,
        CV splitter, An iterable yielding (train, test) splits
        as arrays of indices. For int/None inputs, if the estimator
        is a classifier, and y is either binary or multiclass,
        StratifiedKFold is used. In all other cases, Fold is used.
        These splitters are instantiated with shuffle=False, so the splits
        will be the same across calls. It is only used when hyper_parameter_optimization_method
        is grid or random.
    verbose: int
        Controls the verbosity across all objects: the higher, the more messages.
    random_state: int
        Random number seed.
    n_jobs: int
        The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
        ``-1`` means using all processors. (default -1)
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
        # general argument setting
        hyper_parameter_optimization_method="grid",
        verbose=None,
        random_state=None,
        estimator=None,
        estimator_params=None,
        # grid search and random search
        measure_of_accuracy=None,
        add_extra_args_for_measure_of_accuracy=None,
        n_jobs=None,
        cv=None,
    ):
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
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
        self.cv = cv
        self.best_estimator = None
        self.search_optimization = None
        self.optimized_object = None

    @property
    def best_estimator(self):
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, value):
        self._best_estimator = value

    @property
    def search_optimization(self):
        return self._search_optimization

    @search_optimization.setter
    def search_optimization(self, value):
        self._search_optimization = value

    @property
    def optimized_object(self):
        return self._optimized_object

    @optimized_object.setter
    def optimized_object(self, value):
        self._optimized_object = value

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
    def hyper_parameter_optimization_method(self):
        return self._hyper_parameter_optimization_method

    @hyper_parameter_optimization_method.setter
    def hyper_parameter_optimization_method(self, value):
        if value.lower() == "grid":
            self._hyper_parameter_optimization_method = value
        else:
            raise ValueError(
                f"error occures during selecting optimization_method, {value} is \
                     not supported. The omptimizing engine should be \
                     optuna, grid or random."
            )

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
    def add_extra_args_for_measure_of_accuracy(self):
        return self._add_extra_args_for_measure_of_accuracy

    @add_extra_args_for_measure_of_accuracy.setter
    def add_extra_args_for_measure_of_accuracy(self, value):
        self._add_extra_args_for_measure_of_accuracy = value

    @property
    def test_size(self):
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        self._test_size = value

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, value):
        self._cv = value

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
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    def fit(self, X, y):
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
        self.cols = X.columns
        self.search_optimization = GridSearch(
            X,
            y,
            verbose=self.verbose,
            random_state=self.random_state,
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            # grid search
            measure_of_accuracy=self.measure_of_accuracy,
            add_extra_args_for_measure_of_accuracy=self.add_extra_args_for_measure_of_accuracy,
            n_jobs=self.n_jobs,
            cv=self.cv,
        )
        self.optimized_object = self.search_optimization.optimize()
        self.best_estimator = self.optimized_object.get_best_estimator()

    def get_optimized_object(self):
        """
        Get GridSearchCV  object after optimization.
        """
        return self.optimized_object.grid_search

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


class RandomBestEstimator(AbstractEstimator):
    """
     BestModel estimation using optuna optimization.
     ...

    BestModel estimation using gridseachcv optimization.
     ...

     Parameters
     ----------
     estimator: object
         An unfitted estimator that has fit and predicts methods.
     estimator_params: dict
         Parameters were passed to find the best estimator using the optimization
         method.
     hyper_parameter_optimization_method : str
        Use ``random for Random Search.
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
     cv: int
         cross-validation generator or an iterable.
         Determines the cross-validation splitting strategy. Possible inputs
         for cv are: None, to use the default 5-fold cross-validation,
         int, to specify the number of folds in a (Stratified)KFold,
         CV splitter, An iterable yielding (train, test) splits
         as arrays of indices. For int/None inputs, if the estimator
         is a classifier, and y is either binary or multiclass,
         StratifiedKFold is used. In all other cases, Fold is used.
         These splitters are instantiated with shuffle=False, so the splits
         will be the same across calls. It is only used when hyper_parameter_optimization_method
         is grid or random.
     verbose: int
         Controls the verbosity across all objects: the higher, the more messages.
     random_state: int
         Random number seed.
     n_jobs: int
         The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
         ``-1`` means using all processors. (default -1)
     n_iter : int
         Only it means full in Random Search. It is several parameter
         settings that are sampled. n_iter trades off runtime vs. quality of the solution.

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
         Get RandomizedSearchCV object after optimization.
     Notes
     -----
     It is recommended to use available factories
     to create a new instance of this class.

    """

    def __init__(
        self,
        # general argument setting
        hyper_parameter_optimization_method="random",
        verbose=None,
        random_state=None,
        estimator=None,
        estimator_params=None,
        # grid search and random search
        measure_of_accuracy=None,
        add_extra_args_for_measure_of_accuracy=None,
        n_jobs=None,
        n_iter=None,
        cv=None,
    ):
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
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
        self.n_iter = n_iter
        self.cv = cv
        self.best_estimator = None
        self.search_optimization = None
        self.optimized_object = None

    @property
    def best_estimator(self):
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, value):
        self._best_estimator = value

    @property
    def search_optimization(self):
        return self._search_optimization

    @search_optimization.setter
    def search_optimization(self, value):
        self._search_optimization = value

    @property
    def optimized_object(self):
        return self._optimized_object

    @optimized_object.setter
    def optimized_object(self, value):
        self._optimized_object = value

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
    def hyper_parameter_optimization_method(self):
        return self._hyper_parameter_optimization_method

    @hyper_parameter_optimization_method.setter
    def hyper_parameter_optimization_method(self, value):
        if value.lower() == "random":
            self._hyper_parameter_optimization_method = value
        else:
            raise ValueError(
                f"error occures during selecting optimization_method, {value} is \
                    not supported. The omptimizing engine should be \
                    optuna, grid or random."
            )

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
    def add_extra_args_for_measure_of_accuracy(self):
        return self._add_extra_args_for_measure_of_accuracy

    @add_extra_args_for_measure_of_accuracy.setter
    def add_extra_args_for_measure_of_accuracy(self, value):
        self._add_extra_args_for_measure_of_accuracy = value

    @property
    def test_size(self):
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        self._test_size = value

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, value):
        self._cv = value

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
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    @property
    def best_estimator(self):
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, value):
        self._best_estimator = value

    def fit(self, X, y):
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
        self.cols = X.columns
        self.search_optimization = RandomSearch(
            X,
            y,
            verbose=self.verbose,
            random_state=self.random_state,
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            # random search
            measure_of_accuracy=self.measure_of_accuracy,
            add_extra_args_for_measure_of_accuracy=self.add_extra_args_for_measure_of_accuracy,
            n_jobs=self.n_jobs,
            n_iter=self.n_iter,
            cv=self.cv,
        )
        self.optimized_object = self.search_optimization.optimize()
        self.best_estimator = self.optimized_object.get_best_estimator()

    def get_optimized_object(self):
        """
        Get RandomizedSearchCV  object after optimization.
        """
        return self.optimized_object.random_search

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
        return


class BaseModel(BaseEstimator, metaclass=ABCMeta):
    """
        AutoML with Hyperparameter optimization capabilities.
    ...

    Parameters
    ----------
    estimator: object
        An unfitted estimator that has fit and predicts methods.
    estimator_params: dict
        Parameters were passed to find the best estimator using the optimization
        method.
    hyper_parameter_optimization_method : str
        Type of method for hyperparameter optimization of the estimator.
        Supported methods are Grid Search, Random Search, and Optional.
        Use ``grid`` to set for Grid Search, ``random for Random Search,
        and ``optuna`` for Optuna.
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
    cv: int
        cross-validation generator or an iterable.
        Determines the cross-validation splitting strategy. Possible inputs
        for cv are: None, to use the default 5-fold cross-validation,
        int, to specify the number of folds in a (Stratified)KFold,
        CV splitter, An iterable yielding (train, test) splits
        as arrays of indices. For int/None inputs, if the estimator
        is a classifier, and y is either binary or multiclass,
        StratifiedKFold is used. In all other cases, Fold is used.
        These splitters are instantiated with shuffle=False, so the splits
        will be the same across calls. It is only used when hyper_parameter_optimization_method
        is grid or random.

    with_stratified: bool
        Set True if you want data split in a stratified fashion. (default ``True``)
    verbose: int
        Controls the verbosity across all objects: the higher, the more messages.
    random_state: int
        Random number seed.
    n_jobs: int
        The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
        ``-1`` means using all processors. (default -1)
    n_iter : int
        Only it means full in Random Search. It is several parameter
        settings that are sampled. n_iter trades off runtime vs. quality of the solution.
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

    Methods
    -------
    fit(X, y)
        Fit the feature selection estimator by the best parameters extracted
        from optimization methods.
    predict(X)
        Predict using the best estimator model.
    get_best_estimator()
        Return best estimator, if aleardy fitted.
    Notes
    -----
    It is recommended to use available factories
    to create a new instance of this class.

    """

    def __init__(
        self,
        # general argument setting
        hyper_parameter_optimization_method=None,
        verbose=None,
        random_state=None,
        estimator=None,
        estimator_params=None,
        # grid search and random search
        measure_of_accuracy=None,
        add_extra_args_for_measure_of_accuracy=None,
        n_jobs=None,
        n_iter=None,
        cv=None,
        # optuna params
        test_size=None,
        with_stratified=None,
        # number_of_trials=100,
        # optuna study init params
        study=None,
        # optuna optimization params
        study_optimize_objective=None,
        study_optimize_objective_n_trials=None,
        study_optimize_objective_timeout=None,
        study_optimize_n_jobs=None,
        study_optimize_catch=None,
        study_optimize_callbacks=None,
        study_optimize_gc_after_trial=None,
        study_optimize_show_progress_bar=None,
    ):
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
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
        self.n_iter = n_iter
        self.cv = cv
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

    @property
    def logging_basicConfig(self):
        return self._logging_basicConfig

    @logging_basicConfig.setter
    def logging_basicConfig(self, value):
        self._logging_basicConfig = value

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
    def hyper_parameter_optimization_method(self):
        return self._hyper_parameter_optimization_method

    @hyper_parameter_optimization_method.setter
    def hyper_parameter_optimization_method(self, value):
        self._hyper_parameter_optimization_method = value

    @property
    def measure_of_accuracy(self):
        return self._measure_of_accuracy

    @measure_of_accuracy.setter
    def measure_of_accuracy(self, value):
        self._measure_of_accuracy = value

    @property
    def add_extra_args_for_measure_of_accuracy(self):
        return self._add_extra_args_for_measure_of_accuracy

    @add_extra_args_for_measure_of_accuracy.setter
    def add_extra_args_for_measure_of_accuracy(self, value):
        self._add_extra_args_for_measure_of_accuracy = value

    @property
    def test_size(self):
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        self._test_size = value

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, value):
        self._cv = value

    @property
    def with_stratified(self):
        return self._with_stratified

    @with_stratified.setter
    def with_stratified(self, value):
        self._with_stratified = value

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
    def number_of_trials(self):
        return self._number_of_trials

    @number_of_trials.setter
    def number_of_trials(self, value):
        self._number_of_trials = value

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, value):
        self._sampler = value

    @property
    def pruner(self):
        return self._pruner

    @pruner.setter
    def pruner(self, value):
        self._pruner = value

    @property
    def best_estimator(self):
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, value):
        self._best_estimator = value

    @classmethod
    def optimize_by_gridsearchcv(
        self,
        # general argument setting
        hyper_parameter_optimization_method="grid",
        verbose=None,
        random_state=None,
        estimator=None,
        estimator_params=None,
        # grid search and random search
        measure_of_accuracy=None,
        add_extra_args_for_measure_of_accuracy=None,
        n_jobs=None,
        cv=None,
    ):
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
        self.verbose = verbose
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.n_jobs = n_jobs
        # grid search
        self.measure_of_accuracy = measure_of_accuracy
        self.add_extra_args_for_measure_of_accuracy = (
            add_extra_args_for_measure_of_accuracy
        )
        self.cv = cv

        gse = GridBestEstimator(
            hyper_parameter_optimization_method=self.hyper_parameter_optimization_method,
            verbose=self.verbose,
            random_state=self.random_state,
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            # grid search
            measure_of_accuracy=self.measure_of_accuracy,
            add_extra_args_for_measure_of_accuracy=self.add_extra_args_for_measure_of_accuracy,
            cv=self.cv,
        )
        return gse

    def optimize_by_randomsearchcv(
        self,
        # general argument setting
        hyper_parameter_optimization_method="random",
        verbose=None,
        random_state=None,
        estimator=None,
        estimator_params=None,
        # random search
        measure_of_accuracy=None,
        add_extra_args_for_measure_of_accuracy=None,
        n_jobs=None,
        n_iter=None,
        cv=None,
    ):
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
        self.verbose = verbose
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        # random search
        self.measure_of_accuracy = measure_of_accuracy
        self.add_extra_args_for_measure_of_accuracy = (
            add_extra_args_for_measure_of_accuracy
        )
        self.cv = cv

        rse = RandomBestEstimator(
            hyper_parameter_optimization_method=self.hyper_parameter_optimization_method,
            verbose=self.verbose,
            random_state=self.random_state,
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            # random search
            n_iter=self.n_iter,
            measure_of_accuracy=self.measure_of_accuracy,
            add_extra_args_for_measure_of_accuracy=self.add_extra_args_for_measure_of_accuracy,
            cv=self.cv,
        )

        return rse

    def optimize_by_optuna(
        self,
        # general argument setting
        hyper_parameter_optimization_method="optuna",
        verbose=None,
        random_state=None,
        estimator=None,
        estimator_params=None,
        # grid search and random search
        measure_of_accuracy=None,
        add_extra_args_for_measure_of_accuracy=None,
        n_jobs=None,
        with_stratified=None,
        test_size=None,
        # number_of_trials=100,
        # optuna study init params
        study=None,
        # optuna optimization params
        study_optimize_objective=None,
        study_optimize_objective_n_trials=None,
        study_optimize_objective_timeout=None,
        study_optimize_n_jobs=None,
        study_optimize_catch=None,
        study_optimize_callbacks=None,
        study_optimize_gc_after_trial=None,
        study_optimize_show_progress_bar=None,
    ):

        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
        self.verbose = verbose
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.measure_of_accuracy = measure_of_accuracy
        self.add_extra_args_for_measure_of_accuracy = (
            add_extra_args_for_measure_of_accuracy
        )
        # optuna params
        self.n_jobs = n_jobs
        self.with_stratified = with_stratified
        self.test_size = test_size
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

        ope = OptunaBestEstimator(
            # general argument setting
            hyper_parameter_optimization_method=self.hyper_parameter_optimization_method,
            verbose=self.verbose,
            random_state=self.random_state,
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            measure_of_accuracy=self.measure_of_accuracy,
            add_extra_args_for_measure_of_accuracy=self.add_extra_args_for_measure_of_accuracy,
            # optuna params
            n_jobs=self.n_jobs,
            with_stratified=self.with_stratified,
            test_size=self.test_size,
            # optuna study init params
            study=self.study,
            # optuna optimization params
            study_optimize_objective=self.study_optimize_objective,
            study_optimize_objective_n_trials=self.study_optimize_objective_n_trials,
            study_optimize_objective_timeout=self.study_optimize_objective_timeout,
            study_optimize_n_jobs=self.study_optimize_n_jobs,
            study_optimize_catch=self.study_optimize_catch,
            study_optimize_callbacks=self.study_optimize_callbacks,
            study_optimize_gc_after_trial=self.study_optimize_gc_after_trial,
            study_optimize_show_progress_bar=self.study_optimize_show_progress_bar,
        )
        return ope

    def fit(self, X, y, *args, **kwargs):
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
        self.cols = X.columns
        if self.hyper_parameter_optimization_method == "grid":
            return self.optimize_by_gridsearchcv(
                # general argument setting
                self.hyper_parameter_optimization_method,
                self.verbose,
                self.random_state,
                self.estimator,
                self.estimator_params,
                # grid search
                self.measure_of_accuracy,
                self.add_extra_args_for_measure_of_accuracy,
                self.n_jobs,
                self.cv,
            ).fit(X, y)

        if self.hyper_parameter_optimization_method == "random":
            return self.optimize_by_randomsearchcv(
                # general argument setting
                self.hyper_parameter_optimization_method,
                self.verbose,
                self.random_state,
                self.estimator,
                self.estimator_params,
                # random search
                self.measure_of_accuracy,
                self.add_extra_args_for_measure_of_accuracy,
                self.n_jobs,
                self.n_iter,
                self.cv,
            ).fit(X, y)
        if self.hyper_parameter_optimization_method == "optuna":
            return self.optimize_by_optuna(
                # general argument setting
                self.hyper_parameter_optimization_method,
                self.verbose,
                self.random_state,
                self.estimator,
                self.estimator_params,
                self.measure_of_accuracy,
                self.add_extra_args_for_measure_of_accuracy,
                self.n_jobs,
                self.with_stratified,
                self.test_size,
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
            ).fit(X, y)

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
