

from abc import ABCMeta, abstractmethod


class OptimizerCV(metaclass=ABCMeta):
    """Internal function for returning best estimator using
    assigned parameters by Search CV, e.g., GridSearchCV, RandomizedSearchCV, etc."""
    
    def __init__(self,*args,**kwargs):
        
        """
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection
            step of the pipeline.
        y : Pandas DataFrame or Pandas series
            Training targets. Must fulfill label requirements of the feature selection
            step of the pipeline.
        estimator: object
            An unfitted estimator. For now, only tree-based estimators. Supported
            methods are, "XGBRegressor",
            ``XGBClassifier``, ``RandomForestClassifier``,``RandomForestRegressor``,
            ``CatBoostClassifier``,``CatBoostRegressor``,
            ``BalancedRandomForestClassifier``,
            ``LGBMClassifier``, and ``LGBMRegressor``.
        estimator_params: dict
            Parameters passed to find the best estimator using optimization
            method. 
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
        verbose : int
            Controls the verbosity across all objects: the higher, the more messages.
        n_jobs : int
            Number of jobs to run in parallel for Grid Search, Random Search, and Optuna.
            ``-1`` means using all processors. (default -1)
        cv : int
                cross-validation generator or an iterable.
                Determines the cross-validation splitting strategy. Possible inputs
                for cv are: None, to use the default 5-fold cross-validation,
                int, to specify the number of folds in a (Stratified)KFold,
                CV splitter, An iterable yielding (train, test) splits
                as arrays of indices. For int/None inputs, if the estimator
                is a classifier and y is either binary or multiclass,
                StratifiedKFold is used. In all other cases, Fold is used.
                These splitters are instantiated with shuffle=False, so the splits
                will be the same across calls.
        """
        pass
    @abstractmethod
    def optimize(self,*args,**kwargs):
        """
        Optimize estimator using params using a search cv algorithms
        e.g., GridSearchCV. 
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
    def get_best_estimator(self,*args,**kwargs):
        """
        Return a best_estimator, aproduct of Search CV.
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


