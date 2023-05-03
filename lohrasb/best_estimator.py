from abc import ABCMeta
from pickletools import optimize

import ray
from sklearn.base import BaseEstimator

from lohrasb import logger
from lohrasb.abstracts.estimators import AbstractEstimator
from lohrasb.base_classes.optimizer_bases import (
    GridSearch,
    OptunaSearch,
    RandomSearch,
    TuneGridSearch,
    TuneSearch,
)


class OptunaBestEstimator(AbstractEstimator):
    """BestModel estimation using Optuna optimization.
    ...

    Parameters
    ----------

    estimator: object
        An unfitted estimator that has fit and predicts methods.
    estimator_params: dict
        Parameters were passed to find the best estimator using the optimization
        method.
    fit_params: dict
        A dictionary of parameters that passes to fit the method of the estimator.
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
        fit_params=None,
        # grid search and random search
        measure_of_accuracy=None,
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
        self.fit_params = fit_params
        self.measure_of_accuracy = measure_of_accuracy
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
    def fit_params(self):
        return self._fit_params

    @fit_params.setter
    def fit_params(self, value):
        self._fit_params = value

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
                     optuna, grid, random, raytunegrid"
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
            fit_params=self.fit_params,
            # grid search and random search
            measure_of_accuracy=self.measure_of_accuracy,
            n_jobs=self.n_jobs,
            # optuna params
            test_size=self.test_size,
            with_stratified=self.with_stratified,
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
    """BestModel estimation using GridSearchCV optimization.

    ...

    Parameters
    ----------
    estimator: object
        An unfitted estimator that has fit and predicts methods.
    estimator_params: dict
        Parameters were passed to find the best estimator using the optimization
        method.
    fit_params: dict
        A dictionary of parameters that passes to fit the method of the estimator.
    hyper_parameter_optimization_method : str
        Use ``grid`` to set for Grid Search.
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
        fit_params=None,
        estimator_params=None,
        # grid search and random search
        measure_of_accuracy=None,
        scoring=None,
        n_jobs=None,
        cv=None,
    ):
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
        self.verbose = verbose
        self.random_state = random_state
        self.estimator = estimator
        self.fit_params = fit_params
        self.estimator_params = estimator_params
        # grid search and random search
        self.measure_of_accuracy = measure_of_accuracy
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.best_estimator = None
        self.search_optimization = None
        self.optimized_object = None

    @property
    def fit_params(self):
        return self._fit_params

    @fit_params.setter
    def fit_params(self, value):
        self._fit_params = value

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
                     optuna, grid, random, raytunegrid."
            )

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
        self.search_optimization = GridSearch.remote(
            X,
            y,
            verbose=self.verbose,
            random_state=self.random_state,
            fit_params=self.fit_params,
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            # grid search
            measure_of_accuracy=self.measure_of_accuracy,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv,
        )

        self.optimized_object = ray.get(self.search_optimization.optimize.remote())
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
    """BestModel estimation using RandomizedSearchCV optimization.

     ...

     Parameters
     ----------
     estimator: object
         An unfitted estimator that has fit and predicts methods.
     estimator_params: dict
         Parameters were passed to find the best estimator using the optimization
         method.
    fit_params: dict
        A dictionary of parameters that passes to fit the method of the estimator.
     hyper_parameter_optimization_method : str
        Use ``random`` for Random Search.
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
        If None, the estimator’s score method is used.     cv: int
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
        fit_params=None,
        # grid search and random search
        measure_of_accuracy=None,
        scoring=None,
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
        self.fit_params = fit_params
        # grid search and random search
        self.measure_of_accuracy = measure_of_accuracy
        self.scoring = scoring
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
    def fit_params(self):
        return self._fit_params

    @fit_params.setter
    def fit_params(self, value):
        self._fit_params = value

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
                    optuna, grid, random, raytunegrid or raytunesearch."
            )

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
        ...

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
        self.search_optimization = RandomSearch.remote(
            X,
            y,
            verbose=self.verbose,
            random_state=self.random_state,
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            fit_params=self.fit_params,
            # random search
            measure_of_accuracy=self.measure_of_accuracy,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            n_iter=self.n_iter,
            cv=self.cv,
        )
        self.optimized_object = ray.get(self.search_optimization.optimize.remote())
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


class TuneGridBestEstimator(AbstractEstimator):
    """BestModel estimation using TuneGridSearchCV optimization.

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
            If None, the estimator’s score method is used.       hyper_parameter_optimization_method : str
           Use ``raytunegrid`` for Tune Grid Search.
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
        hyper_parameter_optimization_method="raytunegrid",
        estimator=None,
        estimator_params=None,
        fit_params=None,
        early_stopping=None,
        scoring=None,
        n_jobs=None,
        cv=None,
        refit=None,
        verbose=None,
        error_score=None,
        return_train_score=None,
        local_dir=None,
        name=None,
        max_iters=None,
        use_gpu=None,
        loggers=None,
        pipeline_auto_early_stop=None,
        stopper=None,
        time_budget_s=None,
        mode=None,
        measure_of_accuracy=None,
    ):
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.fit_params = fit_params

        self.early_stopping = early_stopping
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.refit = refit
        self.verbose = verbose
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
        self.measure_of_accuracy = measure_of_accuracy
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
    def fit_params(self):
        return self._fit_params

    @fit_params.setter
    def fit_params(self, value):
        self._fit_params = value

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
    def hyper_parameter_optimization_method(self):
        return self._hyper_parameter_optimization_method

    @hyper_parameter_optimization_method.setter
    def hyper_parameter_optimization_method(self, value):
        if value.lower() == "raytunegrid":
            self._hyper_parameter_optimization_method = value
        else:
            raise ValueError(
                f"error occures during selecting optimization_method, {value} is \
                    not supported. The omptimizing engine should be \
                    optuna, grid, random, raytunegrid, or raytunesearch."
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
    def time_budget_s(self):
        return self._time_budget_s

    @time_budget_s.setter
    def time_budget_s(self, value):
        self._time_budget_s = value

    @property
    def stopper(self):
        return self._stopper

    @stopper.setter
    def stopper(self, value):
        self._stopper = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

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
        self.search_optimization = TuneGridSearch(
            X,
            y,
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            fit_params=self.fit_params,
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
            measure_of_accuracy=self.measure_of_accuracy,
        )
        self.optimized_object = self.search_optimization.optimize()
        self.best_estimator = self.optimized_object.get_best_estimator()

    def get_optimized_object(self):
        """
        Get TuneGridSearch  object after optimization.
        """
        return self.optimized_object.tune_gridsearch

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


class TuneSearchBestEstimator(AbstractEstimator):
    """BestModel estimation using TuneSearchCV optimization.

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

    """

    def __init__(
        self,
        # general argument setting
        hyper_parameter_optimization_method="raytunesearch",
        estimator=None,
        estimator_params=None,
        fit_params=None,
        measure_of_accuracy=None,
        verbose=None,
        early_stopping=None,
        scoring=None,
        n_jobs=None,
        cv=None,
        n_trials=None,
        refit=None,
        random_state=None,
        error_score=None,
        return_train_score=None,
        local_dir=None,
        name=None,
        max_iters=None,
        search_optimization=None,
        use_gpu=None,
        loggers=None,
        pipeline_auto_early_stop=None,
        stopper=None,
        time_budget_s=None,
        mode=None,
        search_kwargs=None,
    ):
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
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
    def hyper_parameter_optimization_method(self):
        return self._hyper_parameter_optimization_method

    @hyper_parameter_optimization_method.setter
    def hyper_parameter_optimization_method(self, value):
        if value.lower() == "raytunesearch":
            self._hyper_parameter_optimization_method = value
        else:
            raise ValueError(
                f"error occures during selecting optimization_method, {value} is \
                    not supported. The omptimizing engine should be \
                    optuna, grid, random, raytunegrid, or raytunesearch."
            )

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
        self.search_optimization_obj = TuneSearch(
            X,
            y,
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            fit_params=self.fit_params,
            measure_of_accuracy=self.measure_of_accuracy,
            verbose=self.verbose,
            early_stopping=self.early_stopping,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv,
            n_trials=self.n_trials,
            refit=self.refit,
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
            search_kwargs=self.search_kwargs,
        )
        self.optimized_object = self.search_optimization_obj.optimize()
        self.best_estimator = self.optimized_object.get_best_estimator()

    def get_optimized_object(self):
        """
        Get TuneSearch  object after optimization.
        """
        return self.optimized_object.tune_search

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
    fit_params: dict
        A dictionary of parameters that passes to fit the method of the estimator.
    hyper_parameter_optimization_method : str
        Type of method for hyperparameter optimization of the estimator.
        Supported methods are Grid Search, Random Search, and Optional.
        Use ``grid`` to set for Grid Search, ``random for Random Search,
        and ``optuna`` for Optuna.
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
    early_stopping: (bool, str or TrialScheduler, optional)
        Option to stop fitting to a hyperparameter configuration if it performs poorly. Possible inputs are:
        If True, defaults to ASHAScheduler. A string corresponding to the name of a Tune Trial Scheduler (i.e.,
        “ASHAScheduler”). To specify parameters of the scheduler, pass in a scheduler object instead of a string.
        Scheduler for executing fit with early stopping. Only a subset of schedulers are currently supported.
        The scheduler will only be used if the estimator supports partial fitting If None or False,
        early stopping will not be used.
    scoring : str, list/tuple, dict, or None)
        (For TuneSearch) A single string or a callable to evaluate the predictions on the test set.
        See https://scikit-learn.org/stable/modules/model_evaluation.html #scoring-parameter
        for all options. For evaluating multiple metrics, either give a list/tuple of (unique)
        strings or a dict with names as keys and callables as values. If None, the estimator’s
        score method is used. Defaults to None.
    n_jobs : int
        (For TuneSearch) Number of jobs to run in parallel. None or -1 means using all processors. Defaults to None.
        If set to 1, jobs will be run using Ray’s ‘local mode’. This can lead to significant speedups
        if the model takes < 10 seconds to fit due to removing inter-process communication overheads.
    cv : int, cross-validation generator or iterable :
        (For TuneSearch) Determines the cross-validation splitting strategy. Possible inputs for cv are:
        None, to use the default 5-fold cross validation, integer, to specify the number
        of folds in a (Stratified)KFold, An iterable yielding (train, test) splits as arrays
        of indices. For integer/None inputs, if the estimator is a classifier and y is either
        binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
        Defaults to None.
    refit : bool or str
        (For TuneSearch) Refit an estimator using the best found parameters on the whole dataset.
        For multiple metric evaluation, this needs to be a string denoting the scorer
        that would be used to find the best parameters for refitting the estimator at the end.
        The refitted estimator is made available at the best_estimator_ attribute and permits using predict
        directly on this GridSearchCV instance. Also for multiple metric evaluation,
        the attributes best_index_, best_score_ and best_params_ will only be available if
        refit is set and all of them will be determined w.r.t this specific scorer.
        If refit not needed, set to False. See scoring parameter to know more about multiple
        metric evaluation. Defaults to True.
    verbose : int
        (For TuneSearch) Controls the verbosity: 0 = silent, 1 = only status updates, 2 = status and trial results.
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
    n_trials : int
        Number of parameter settings that are sampled. n_trials trades
        off runtime vs quality of the solution. Defaults to 10.
        Note: only for TuneSearchCV.




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
        n_jobs=None,
        n_iter=None,
        cv=None,
        n_trials=None,
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
        # tune grid search optimization params
        early_stopping=None,
        scoring=None,
        refit=None,
        error_score=None,
        return_train_score=None,
        local_dir=None,
        name=None,
        max_iters=None,
        use_gpu=None,
        loggers=None,
        pipeline_auto_early_stop=None,
        stopper=None,
        time_budget_s=None,
        mode=None,
        # tune search optimization params
        search_optimization=None,
        search_kwargs=None,
    ):
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
        self.verbose = verbose
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        # grid search and random search
        self.measure_of_accuracy = measure_of_accuracy
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.cv = cv
        self.n_trials = n_trials
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
        # tune grid search optimization params
        self.early_stopping = early_stopping
        self.scoring = scoring
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
        # tune  search optimization params
        self.search_optimization = search_optimization
        self.search_kwargs = search_kwargs

    @property
    def fit_params(self):
        return self._fit_params

    @fit_params.setter
    def fit_params(self, value):
        self._fit_params = value

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
    def n_trials(self):
        return self._n_trials

    @n_trials.setter
    def n_trials(self, value):
        self._n_trials = value

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
    def time_budget_s(self):
        return self._time_budget_s

    @time_budget_s.setter
    def time_budget_s(self, value):
        self._time_budget_s = value

    @property
    def stopper(self):
        return self._stopper

    @stopper.setter
    def stopper(self, value):
        self._stopper = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    @property
    def search_optimization(self):
        return self._search_optimization

    @search_optimization.setter
    def search_optimization(self, value):
        self._search_optimization = value

    @property
    def search_kwargs(self):
        return self._search_kwargs

    @search_kwargs.setter
    def search_kwargs(self, value):
        self._search_kwargs = value

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
        fit_params=None,
        # grid search and random search
        measure_of_accuracy=None,
        scoring=None,
        n_jobs=None,
        cv=None,
    ):
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
        self.verbose = verbose
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.fit_params = fit_params
        self.n_jobs = n_jobs
        # grid search
        self.measure_of_accuracy = measure_of_accuracy
        self.scoring = scoring
        self.cv = cv

        gse = GridBestEstimator(
            hyper_parameter_optimization_method=self.hyper_parameter_optimization_method,
            verbose=self.verbose,
            random_state=self.random_state,
            estimator=self.estimator,
            fit_params=self.fit_params,
            estimator_params=self.estimator_params,
            # grid search
            measure_of_accuracy=self.measure_of_accuracy,
            scoring=self.scoring,
            cv=self.cv,
        )
        return gse

    @classmethod
    def optimize_by_randomsearchcv(
        self,
        # general argument setting
        hyper_parameter_optimization_method="random",
        verbose=None,
        random_state=None,
        estimator=None,
        estimator_params=None,
        fit_params=None,
        # random search
        measure_of_accuracy=None,
        scoring=None,
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
        self.fit_params = fit_params
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        # random search
        self.measure_of_accuracy = measure_of_accuracy
        self.scoring = scoring
        self.cv = cv

        rse = RandomBestEstimator(
            hyper_parameter_optimization_method=self.hyper_parameter_optimization_method,
            verbose=self.verbose,
            random_state=self.random_state,
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            fit_params=self.fit_params,
            # random search
            n_iter=self.n_iter,
            measure_of_accuracy=self.measure_of_accuracy,
            scoring=self.scoring,
            cv=self.cv,
        )

        return rse

    @classmethod
    def optimize_by_tunegridsearchcv(
        self,
        # general argument setting
        hyper_parameter_optimization_method="raytunegrid",
        estimator=None,
        estimator_params=None,
        fit_params=None,
        early_stopping=None,
        scoring=None,
        n_jobs=None,
        cv=None,
        refit=None,
        verbose=None,
        error_score=None,
        return_train_score=None,
        local_dir=None,
        name=None,
        max_iters=None,
        use_gpu=None,
        loggers=None,
        pipeline_auto_early_stop=None,
        stopper=None,
        time_budget_s=None,
        mode=None,
        measure_of_accuracy=None,
    ):
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.fit_params = fit_params
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.refit = refit
        self.verbose = verbose
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
        self.measure_of_accuracy = measure_of_accuracy

        tge = TuneGridBestEstimator(
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            fit_params=self.fit_params,
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
            measure_of_accuracy=self.measure_of_accuracy,
        )

        return tge

    @classmethod
    def optimize_by_tunesearchcv(
        self,
        # general argument setting
        hyper_parameter_optimization_method="raytunesearch",
        estimator=None,
        estimator_params=None,
        fit_params=None,
        measure_of_accuracy=None,
        verbose=None,
        early_stopping=None,
        scoring=None,
        n_jobs=None,
        cv=None,
        n_trials=None,
        refit=None,
        random_state=None,
        error_score=None,
        return_train_score=None,
        local_dir=None,
        name=None,
        max_iters=None,
        search_optimization=None,
        use_gpu=None,
        loggers=None,
        pipeline_auto_early_stop=None,
        stopper=None,
        time_budget_s=None,
        mode=None,
        search_kwargs=None,
    ):
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
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
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.local_dir = local_dir
        self.name = name
        self.max_iters = max_iters
        self.search_optimization = search_optimization
        self.use_gpu = use_gpu
        self.loggers = loggers
        self.pipeline_auto_early_stop = pipeline_auto_early_stop
        self.stopper = stopper
        self.time_budget_s = time_budget_s
        self.mode = mode
        self.search_kwargs = search_kwargs

        te = TuneSearchBestEstimator(
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            fit_params=self.fit_params,
            measure_of_accuracy=self.measure_of_accuracy,
            verbose=self.verbose,
            early_stopping=self.early_stopping,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv,
            n_trials=self.n_trials,
            refit=self.refit,
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
            search_kwargs=self.search_kwargs,
        )

        return te

    def optimize_by_optuna(
        self,
        # general argument setting
        hyper_parameter_optimization_method="optuna",
        verbose=None,
        random_state=None,
        estimator=None,
        estimator_params=None,
        fit_params=None,
        # grid search and random search
        measure_of_accuracy=None,
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
        self.fit_params = fit_params
        self.measure_of_accuracy = measure_of_accuracy
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
            fit_params=self.fit_params,
            measure_of_accuracy=self.measure_of_accuracy,
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
                self.fit_params,
                # grid search
                self.measure_of_accuracy,
                self.scoring,
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
                self.fit_params,
                # random search
                self.measure_of_accuracy,
                self.scoring,
                self.n_jobs,
                self.n_iter,
                self.cv,
            ).fit(X, y)
        if self.hyper_parameter_optimization_method == "raytunegrid":
            return self.optimize_by_tunegridsearchcv(
                # general argument setting
                self.estimator,
                self.estimator_params,
                self.fit_params,
                self.early_stopping,
                self.scoring,
                self.n_jobs,
                self.cv,
                self.refit,
                self.verbose,
                self.error_score,
                self.return_train_score,
                self.local_dir,
                self.name,
                self.max_iters,
                self.use_gpu,
                self.loggers,
                self.pipeline_auto_early_stop,
                self.stopper,
                self.time_budget_s,
                self.mode,
                self.measure_of_accuracy,
            ).fit(X, y)
        if self.hyper_parameter_optimization_method == "raytunesearch":
            return self.optimize_by_tunesearchcv(
                # general argument setting
                self.estimator,
                self.estimator_params,
                self.fit_params,
                self.measure_of_accuracy,
                self.verbose,
                self.early_stopping,
                self.scoring,
                self.n_jobs,
                self.cv,
                self.refit,
                self.random_state,
                self.error_score,
                self.return_train_score,
                self.local_dir,
                self.name,
                self.max_iters,
                self.search_optimization,
                self.use_gpu,
                self.loggers,
                self.pipeline_auto_early_stop,
                self.stopper,
                self.time_budget_s,
                self.mode,
                self.search_kwargs,
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
