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
        
        self.optimized_object = OptunaSearch(X, y, *self.args, **self.kwargs).\
            prepare_data().optimize()
    
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
        Get GridSearchCV  object after optimization.
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

class GridBestEstimator(AbstractEstimator):
    """BestModel estimation using GridSearchCV optimization.
    ...

    Parameters
    ----------
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
        self.optimized_object = GridSearch(X, y, *self.args, **self.kwargs).\
            optimize()
        #self.best_estimator = self.optimized_object.get_best_estimator()
    
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
        Get GridSearchCV  object after optimization.
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
    """BestModel estimation using OptunaSearchCV optimization.
    ...

    Parameters
    ----------
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


class RandomBestEstimator(AbstractEstimator):
    """BestModel estimation using RandomizedSearchCV optimization.

     ...

     Parameters
     ----------
     estimator: object
         An unfitted estimator that has fit and predicts methods.

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
        self.optimized_object = RandomSearch(X, y, *self.args, **self.kwargs).\
            optimize()
        #self.best_estimator = self.optimized_object.get_best_estimator()
    
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
        Get randomSearchCV  object after optimization.
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


class TuneBestEstimator(AbstractEstimator):
    """BestModel estimation using RandomizedSearchCV optimization.

     ...

     Parameters
     ----------
     estimator: object
         An unfitted estimator that has fit and predicts methods.

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
        self.optimized_object = TuneCV(X, y, *self.args, **self.kwargs).\
            optimize()
        #self.best_estimator = self.optimized_object.get_best_estimator()
    
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
        Get randomSearchCV  object after optimization.
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

class TuneGridBestEstimator(AbstractEstimator):
    """BestModel estimation using TuneGridSearchCV optimization.

    Parameters
    ----------
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
        self.optimized_object = TuneGridSearch(X, y, *self.args, **self.kwargs).\
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
        Get GridSearchCV  object after optimization.
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

class TuneSearchBestEstimator(AbstractEstimator):
    """BestModel estimation using TuneGridSearchCV optimization.

    Parameters
    ----------
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
        self.optimized_object = TuneSearch(X, y, *self.args, **self.kwargs).\
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
        Get GridSearchCV  object after optimization.
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


class BaseModel(BaseEstimator, metaclass=ABCMeta):
    """
        AutoML with Hyperparameter optimization capabilities.
    ...

    Parameters
    ----------
    kwargs : kwargs of the method

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
        # grid for test
        **kwargs,
    ):  

        self.kwargs=kwargs

    @classmethod
    def optimize_by_gridsearchcv(
        self,
        # general argument setting
        *args,
        **grid_search_kwargs,

    ):
        # general argument setting
        self.grid_search_kwargs = grid_search_kwargs
        gse = GridBestEstimator(**self.grid_search_kwargs)
        return gse

    @classmethod
    def optimize_by_optunasearchcv(
        self,
        # general argument setting
        *args,
        **newoptuna_search_kwargs,

    ):
        # general argument setting
        self.newoptuna_search_kwargs = newoptuna_search_kwargs
        noe = NewOptunaBestEstimator(**self.newoptuna_search_kwargs)
        return noe 

    @classmethod
    def optimize_by_randomsearchcv(
        self,
        # general argument setting
        *args,
        **random_search_kwargs,

    ):
        # general argument setting
        self.random_search_kwargs = random_search_kwargs
        rse = RandomBestEstimator(**self.random_search_kwargs)
        return rse

    @classmethod
    def optimize_by_tunegridsearchcv(
        self,
        # general argument setting
        *args,
        **tunegrid_search_kwargs,

    ):
        # general argument setting
        self.tunegrid_search_kwargs = tunegrid_search_kwargs
        tge = TuneGridBestEstimator(**self.tunegrid_search_kwargs)
        return tge

    @classmethod
    def optimize_by_tunesearchcv(
        self,
        # general argument setting
        *args,
        **tune_search_kwargs,

    ):
        # general argument setting
        self.tune_search_kwargs = tune_search_kwargs
        tse = TuneSearchBestEstimator(**self.tune_search_kwargs)
        return tse

    @classmethod
    def optimize_by_optuna(
        self,
        # general argument setting
        *args,
        **optuna_search_kwargs,

    ):
        # general argument setting
        self.optuna_search_kwargs = optuna_search_kwargs 
        obe = OptunaBestEstimator(**self.optuna_search_kwargs)
        return obe

    @classmethod
    def optimize_by_tune(
        self,
        # general argument setting
        *args,
        **tune_search_kwargs,

    ):
        # general argument setting
        self.tune_search_kwargs = tune_search_kwargs
        tbe = TuneBestEstimator(**self.tune_search_kwargs)
        return tbe

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
        pass

    def predict(self, X):
        """Predict using the best estimator model.
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection
            step of the pipeline.
        """
        pass

    def get_best_estimator(self):
        """Return best estimator if model already fitted."""
        pass
