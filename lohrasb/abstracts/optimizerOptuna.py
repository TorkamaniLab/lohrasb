

from abc import ABCMeta, abstractmethod


class OptimizerOptuna(metaclass=ABCMeta):
    """Internal function for returning best estimator using
    assigned parameters by Optuna."""
    
    def __init__(self,*args,**kwargs):
                
        """Internal function for returning best estimator using
            assigned parameters by Optuna.
            Parameters
            ----------
            X : Pandas DataFrame
                Training data. Must fulfill input requirements of the feature selection
                step of the pipeline.
            y : Pandas DataFrame or Pandas series
                Training targets. Must fulfill label requirements of the feature selection
                step of the pipeline.
            estimator: object
                An unfitted estimator. 
            measure_of_accuracy : str
                Measurement of performance for classification and
                regression estimator during hyperparameter optimization while
                estimating best estimator. Classification-supported measurments are
                f1, f1_score, acc, accuracy_score, pr, precision_score,
                recall, recall_score, roc, roc_auc_score, roc_auc,
                tp, true positive, tn, true negative. Regression supported
                measurements are r2, r2_score, explained_variance_score,
                max_error, mean_absolute_error, mean_squared_error,
                median_absolute_error, and mean_absolute_percentage_error.    ----------
            test_size : float or int
                If float, it should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the train split during estimating the best estimator
                by optimization method. If int represents the
                absolute number of train samples. If None, the value is automatically
                set to the complement of the test size.
            random_state : int
                Random number seed.
            eval_metric : str
                An evaluation metric name for pruning. For xgboost.XGBClassifier it is
                ``auc``, for catboost.CatBoostClassifier it is ``AUC`` for catboost.CatBoostRegressor
                it is ``RMSE``.
            number_of_trials : int
                The number of trials. If this argument is set to None,
                there is no limitation on the number of trials.
            sampler : object
                optuna.samplers. For more information, see:
                ``https://optuna.readthedocs.io/en/stable/reference/samplers.html#module-optuna.samplers``.
            pruner : object
                optuna.pruners. For more information, see:
                ``https://optuna.readthedocs.io/en/stable/reference/pruners.html``.
            with_stratified : bool
                Set True if you want data split in a stratified fashion. (default ``True``).
            """
        pass
    @abstractmethod
    def train_test_split(self,*args,**kwargs):
        """
        Split X, y, into X_train and y_train, X_test, y_test.
        ...

        Attributes 
        ----------
        *args: list
            A list of possible argumnets
        **kwargs: dict
            A dict of possible argumnets
        """
        pass

    @abstractmethod
    def define_objective_func(self,*args,**kwargs):
        """
        Define objective function that required for Optuna
        ...

        Attributes 
        ----------
        *args: list
            A list of possible argumnets
        **kwargs: dict
            A dict of possible argumnets
        """
        pass
    
    @abstractmethod
    def get_optimized_object(self,*args,**kwargs):
        """
        Return whole object, a product of Search CV .
        ...

        Attributes 
        ----------
        *args: list
            A list of possible argumnets
        **kwargs: dict
            A dict of possible argumnets
        """
        pass


