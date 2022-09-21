# https://www.geeksforgeeks.org/inspect-module-in-python/

from abc import ABCMeta
from pickletools import optimize
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
import numpy as np
from lohrasb.model_conf import SUPPORTED_MODELS
from xgboost import XGBClassifier, XGBRegressor

from lohrasb.utils.helper_funcs import install_and_import
from lohrasb.base_classes.optimizer_bases import (
    GridSearchFactory,
    OptunaFactory,
    RandomSearchFactory
)


class BaseModel(BaseEstimator, metaclass=ABCMeta):
    """
        Feature Selector class using shap values. It is extended from scikit-learn
        BaseEstimator and TransformerMixin.
    ...

    Attributes
    ----------
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
    hyper_parameter_optimization_method : str
        Type of method for hyperparameter optimization of the estimator.
        Supported methods are: Grid Search, Random Search, and Optuna.
        use ``grid`` to set for Grid Search, ``random`` to set for Random Search,
        and ``optuna`` for Optuna method. (default ``optuna``)
    measure_of_accuracy : str
        Measurement of performance for classification and
        regression estimator during hyperparameter optimization while
        estimating best estimator. Classification-supported measurments are
        f1, f1_score, acc, accuracy_score, pr, precision_score,
        recall, recall_score, roc, roc_auc_score, roc_auc,
        tp, true positive, tn, true negative. Regression supported
        measurements are r2, r2_score, explained_variance_score,
        max_error, mean_absolute_error, mean_squared_error,
        median_absolute_error, and mean_absolute_percentage_error.
    test_size : float or int
        If float, it should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the train split during estimating the best estimator
        by optimization method. If int represents the
        absolute number of train samples. If None, the value is automatically
        set to the complement of the test size.
    cv : int
        cross-validation generator or an iterable.
        Determines the cross-validation splitting strategy. Possible inputs
        for cv are: None, to use the default 5-fold cross-validation,
        int, to specify the number of folds in a (Stratified)KFold,
        CV splitter, An iterable yielding (train, test) splits
        as arrays of indices. For int/None inputs, if the estimator
        are a classifier, and y is either binary or multiclass,
        StratifiedKFold is used. In all other cases, Fold is used.
        These splitters are instantiated with shuffle=False, so the splits
         will be the same across calls.
    with_stratified: bool
        Set True if you want data split in a stratified fashion. (default ``True``)
    verbose : int
        Controls the verbosity across all objects: the higher, the more messages.
    random_state : int
        Random number seed.
    n_jobs : int
        Number of jobs to run in parallel for Grid Search, Random Search, and Optuna.
        ``-1`` means using all processors. (default -1)
    n_iter : int
        Only it means full in Random Search. it is several parameter
        settings that are sampled. n_iter trades off runtime vs quality of the solution.
    eval_metric : str
        An evaluation metric name for pruning. For xgboost.XGBClassifier it is
        ``auc``, for catboost.CatBoostClassifier it is ``AUC`` for catboost.CatBoostRegressor
        it is ``RMSE``.
    number_of_trials : int
        The number of trials. If this argument is set to None,
        there is no limitation on the number of trials. (default 20)
    sampler : object
        optuna.samplers. For more information, see:
        ``https://optuna.readthedocs.io/en/stable/reference/samplers.html#module-optuna.samplers``.
        (default TPESampler())
    pruner : object
        optuna.pruners. For more information, see:
        ``https://optuna.readthedocs.io/en/stable/reference/pruners.html``.
        (default HyperbandPruner())

    Methods
    -------
    fit(X, y)
        Fit the feature selection estimator by best parameters extracted
        from optimization methods.
    predict(X)
        Predict using the best estimator model.
    """

    def __init__(
        self,
        # general argument setting
        hyper_parameter_optimization_method=None,
        verbose=0,
        random_state=0,
        estimator=None,
        estimator_params=None,
        # grid search and random search
        measure_of_accuracy=None,
        n_jobs=None,
        n_iter=None,
        cv=None,
        # optuna params
        test_size=0.33,
        with_stratified=False,
        # number_of_trials=100,
        # optuna study init params
        study=optuna.create_study(
            storage=None,
            sampler=TPESampler(),
            pruner=HyperbandPruner(),
            study_name=None,
            direction="maximize",
            load_if_exists=False,
            directions=None,
        ),
        # optuna optimization params
        study_optimize_objective=None,
        study_optimize_objective_n_trials=100,
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs=-1,
        study_optimize_catch=(),
        study_optimize_callbacks=None,
        study_optimize_gc_after_trial=False,
        study_optimize_show_progress_bar=False,
    ):
        """
        Parameters
        ----------
        n_features : int
            The number of features seen during term:`fit`. Only defined if the
            underlying estimator exposes such an attribute when fitted.
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
        hyper_parameter_optimization_method : str
            Type of method for hyperparameter optimization of the estimator.
            Supported methods are: Grid Search, Random Search, and Optuna.
            use ``grid`` to set for Grid Search, ``random`` to set for Random Search,
            and ``optuna`` for Optuna method. (default ``optuna``)
        shap_version : str
            FastTreeSHAP algorithms. Supported version ``v0``,
            ``v1``, and ``v2``. Check this paper
            `` https://arxiv.org/abs/2109.09847 ``
        measure_of_accuracy : str
            Measurement of performance for classification and
            regression estimator during hyperparameter optimization while
            estimating best estimator. Classification-supported measurments are
            f1, f1_score, acc, accuracy_score, pr, precision_score,
            recall, recall_score, roc, roc_auc_score, roc_auc,
            tp, true positive, tn, true negative. Regression supported
            measurements are r2, r2_score, explained_variance_score,
            max_error, mean_absolute_error, mean_squared_error,
            median_absolute_error, and mean_absolute_percentage_error.
        list_of_obligatory_features : [str]
            A list of strings (columns names of feature set pandas data frame)
            that should be among selected features. No matter if they have high or
            low shap values will be selected at the end of feature selection
            step.
        test_size : float or int
            If float, it should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the train split during estimating the best estimator
            by optimization method. If int represents the
            absolute number of train samples. If None, the value is automatically
            set to the complement of the test size.
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
        with_shap_summary_plot : bool
            Set True if you want to see a shap summary plot of
            selected features. (default ``False``)
        with_stratified : bool
            Set True if you want data split in a stratified fashion. (default ``True``)
        verbose : int
            Controls the verbosity across all objects: the higher, the more messages.
        random_state : int
            Random number seed.
        n_jobs : int
            The number of jobs to run in parallel for Grid Search, Random Search, and Optuna.
            ``-1`` means using all processors. (default -1)
        n_iter : int
            Only it means full in Random Search. it is a number of parameter
            settings that are sampled. n_iter trades off runtime vs quality of the solution.
        eval_metric : str
            An evaluation metric name for pruning. For xgboost.XGBClassifier it is
            ``auc``, for catboost.CatBoostClassifier it is ``AUC`` for catboost.CatBoostRegressor
            it is ``RMSE``.
        number_of_trials : int
            The number of trials. If this argument is set to None,
            there is no limitation on the number of trials. (default 20)
        sampler : object
            optuna.samplers. For more information, see:
            ``https://optuna.readthedocs.io/en/stable/reference/samplers.html#module-optuna.samplers``.
            (default TPESampler())
        pruner : object
            optuna.pruners. For more information, see:
            ``https://optuna.readthedocs.io/en/stable/reference/pruners.html``.
            (default HyperbandPruner())
        """

         # general argument setting
        self.hyper_parameter_optimization_method=hyper_parameter_optimization_method
        self.verbose=verbose
        self.random_state=random_state
        self.estimator=estimator
        self.estimator_params=estimator_params
        # grid search and random search
        self.measure_of_accuracy=measure_of_accuracy
        self.n_jobs=n_jobs
        self.n_iter=n_iter
        self.cv=cv
        # optuna params
        self.test_size=test_size
        self.with_stratified=with_stratified
        # number_of_trials=100,
        # optuna study init params
        self.study=study
        # optuna optimization params
        self.study_optimize_objective=study_optimize_objective
        self.study_optimize_objective_n_trials=study_optimize_objective_n_trials
        self.study_optimize_objective_timeout=study_optimize_objective_timeout
        self.study_optimize_n_jobs=study_optimize_n_jobs
        self.study_optimize_catch=study_optimize_catch
        self.study_optimize_callbacks=study_optimize_callbacks
        self.study_optimize_gc_after_trial=study_optimize_gc_after_trial
        self.study_optimize_show_progress_bar=study_optimize_show_progress_bar


    @property
    def estimator(self):
        print("Getting value for estimator")
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        print("Setting value for estimator")
        self._estimator = value

    @property
    def estimator_params(self):
        print("Getting value for estimator_params")
        return self._estimator_params

    @estimator_params.setter
    def estimator_params(self, value):
        print("Setting value for  estimator params")
        self._estimator_params = value

    @property
    def hyper_parameter_optimization_method(self):
        print("Getting value for hyper_parameter_optimization_method")
        return self._hyper_parameter_optimization_method

    @hyper_parameter_optimization_method.setter
    def hyper_parameter_optimization_method(self, value):
        print("Setting value for hyper_parameter_optimization_method")
        if (
            value.lower() == "optuna"
            or value.lower() == "grid"
            or value.lower() == "random"
        ):
            self._hyper_parameter_optimization_method = value
        else:
            raise ValueError(
                f"error occures during selecting optimization_method, {value} is \
                     not supported."
            )

    @property
    def measure_of_accuracy(self):
        print("Getting value for measure_of_accuracy")
        return self._measure_of_accuracy

    @measure_of_accuracy.setter
    def measure_of_accuracy(self, value):
        print("Setting value for measure_of_accuracy")
        self._measure_of_accuracy = value

    @property
    def test_size(self):
        print("Getting value for test_size")
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        print("Setting value for test_size")
        self._test_size = value

    @property
    def cv(self):
        print("Getting value for Cross Validation object")
        return self._cv

    @cv.setter
    def cv(self, value):
        print("Setting value for Cross Validation object")
        self._cv = value

    @property
    def with_stratified(self):
        print("Getting value for with_stratified")
        return self._with_stratified

    @with_stratified.setter
    def with_stratified(self, value):
        print("Setting value for with_stratified")
        self._with_stratified = value

    @property
    def verbose(self):
        print("Getting value for verbose")
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        print("Setting value for verbose")
        self._verbose = value

    @property
    def random_state(self):
        print("Getting value for random_state")
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        print("Setting value for random_state")
        self._random_state = value

    @property
    def n_jobs(self):
        print("Getting value for n_jobs")
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        print("Setting value for n_jobs")
        self._n_jobs = value

    @property
    def n_iter(self):
        print("Getting value for n_iter")
        return self._n_iter

    @n_iter.setter
    def n_iter(self, value):
        print("Setting value for n_iter")
        self._n_iter = value

    @property
    def eval_metric(self):
        print("Getting value for eval_metric")
        return self._eval_metric

    @eval_metric.setter
    def eval_metric(self, value):
        print("Setting value for eval_metric")
        self._eval_metric = value

    @property
    def number_of_trials(self):
        print("Getting value for number_of_trials")
        return self._number_of_trials

    @number_of_trials.setter
    def number_of_trials(self, value):
        print("Setting value for number_of_trials")
        self._number_of_trials = value

    @property
    def sampler(self):
        print("Getting value for sampler")
        return self._sampler

    @sampler.setter
    def sampler(self, value):
        print("Setting value for sampler")
        self._sampler = value

    @property
    def pruner(self):
        print("Getting value for pruner")
        return self._pruner

    @pruner.setter
    def pruner(self, value):
        print("Setting value for pruner")
        self._pruner = value

    @property
    def best_estimator(self):
        print("Getting value for best_estimator")
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, value):
        print("Setting value for best_estimator")
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
        self.best_estimator  = BestEstimatorFactory(
            type_engine=self.hyper_parameter_optimization_method,
            X=X,
            y=y,
            estimator = self.estimator,
            estimator_params=self.estimator_params,
            measure_of_accuracy=self.measure_of_accuracy,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=self.random_state,
            # optuna params
            test_size=self.test_size,
            with_stratified=self.with_stratified,
            # number_of_trials=100,
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
            ).return_engine()
    def predict(self, X):
        """Predict using the best estimator model.
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection
            step of the pipeline.
        """
        return self.best_estimator.predict(X)

    class BestModelFactory:
        def using_optuna(
                self,
                hyper_parameter_optimization_method='optuna',
                verbose=0,
                random_state=0,
                estimator=None,
                estimator_params=None,
                # grid search and random search
                measure_of_accuracy=None,
                n_jobs=None,
                # optuna params
                test_size=0.33,
                with_stratified=False,
                # number_of_trials=100,
                # optuna study init params
                study=optuna.create_study(
                    storage=None,
                    sampler=TPESampler(),
                    pruner=HyperbandPruner(),
                    study_name=None,
                    direction="maximize",
                    load_if_exists=False,
                    directions=None,
                ),
                # optuna optimization params
                study_optimize_objective=None,
                study_optimize_objective_n_trials=100,
                study_optimize_objective_timeout=600,
                study_optimize_n_jobs=-1,
                study_optimize_catch=(),
                study_optimize_callbacks=None,
                study_optimize_gc_after_trial=False,
                study_optimize_show_progress_bar=False,
            ):
            best_model = BaseModel(hyper_parameter_optimization_method='optuna')
            best_model.verbose=verbose
            best_model.random_state=random_state
            best_model.estimator=estimator
            best_model.estimator_params=estimator_params
            best_model.measure_of_accuracy=measure_of_accuracy
            best_model.n_jobs=n_jobs
            # optuna params
            best_model.test_size=test_size
            best_model.with_stratified=with_stratified
            # number_of_trials=100,
            # optuna study init params
            best_model.study=study
            # optuna optimization params
            best_model.study_optimize_objective=study_optimize_objective
            best_model.study_optimize_objective_n_trials=study_optimize_objective_n_trials
            best_model.study_optimize_objective_timeout=study_optimize_objective_timeout
            best_model.study_optimize_n_jobs=study_optimize_n_jobs
            best_model.study_optimize_catch=study_optimize_catch
            best_model.study_optimize_callbacks=study_optimize_callbacks
            best_model.study_optimize_gc_after_trial=study_optimize_gc_after_trial
            best_model.study_optimize_show_progress_bar=study_optimize_show_progress_bar
            return best_model

        def using_gridsearch(
                self,
                hyper_parameter_optimization_method='grid',
                verbose=0,
                random_state=0,
                estimator=None,
                estimator_params=None,
                # grid search and random search
                measure_of_accuracy=None,
                n_jobs=None,
                cv=None,

                
            ):
            best_model = BaseModel(hyper_parameter_optimization_method='grid')
            best_model.hyper_parameter_optimization_method='grid'
            best_model.verbose=verbose
            best_model.random_state=random_state
            best_model.estimator=estimator
            best_model.estimator_params=estimator_params
            best_model.measure_of_accuracy=measure_of_accuracy
            best_model.n_jobs=n_jobs
            best_model.cv = cv
            return best_model

        def using_randomsearch(
                self,
                hyper_parameter_optimization_method='random',
                verbose=0,
                random_state=0,
                estimator=None,
                estimator_params=None,
                # grid search and random search
                measure_of_accuracy=None,
                n_jobs=None,
                cv=None,
                n_iter = None,

                
            ):
            best_model = BaseModel(hyper_parameter_optimization_method='random')
            best_model.hyper_parameter_optimization_method='random'
            best_model.verbose=verbose
            best_model.random_state=random_state
            best_model.estimator=estimator
            best_model.estimator_params=estimator_params
            best_model.measure_of_accuracy=measure_of_accuracy
            best_model.n_jobs=n_jobs
            best_model.cv = cv
            best_model.n_iter = n_iter
            return best_model
        
    bestmodel_factory = BestModelFactory()


class BestEstimatorFactory:
    def __init__(
        self,
        type_engine,
        X,
        y,
        estimator,
        estimator_params,
        measure_of_accuracy,
        verbose,
        n_jobs,
        n_iter,
        cv,
        random_state,
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
            self.type_engine=type_engine
            self.X = X
            self.y= y
            self.estimator=estimator
            self.estimator_params = estimator_params
            self.measure_of_accuracy=measure_of_accuracy
            self.verbose=verbose
            self.n_jobs=n_jobs
            self.n_iter=n_iter
            self.cv=cv
            self.random_state=random_state
            # optuna params
            self.test_size=test_size
            self.with_stratified=with_stratified
            # number_of_trials=100,
            # optuna study init params
            self.study=study
            # optuna optimization params
            self.study_optimize_objective=study_optimize_objective
            self.study_optimize_objective_n_trials=study_optimize_objective_n_trials
            self.study_optimize_objective_timeout=study_optimize_objective_timeout
            self.study_optimize_n_jobs=study_optimize_n_jobs
            self.study_optimize_catch=study_optimize_catch
            self.study_optimize_callbacks=study_optimize_callbacks
            self.study_optimize_gc_after_trial=study_optimize_gc_after_trial
            self.study_optimize_show_progress_bar=study_optimize_show_progress_bar

    def using_randomsearch(self):
        return RandomSearchFactory().optimizer_builder(
                self.X,
                self.y,
                self.estimator,
                self.estimator_params,
                self.measure_of_accuracy,
                self.verbose,
                self.n_jobs,
                self.n_iter,
                self.cv,
            ).optimize().get_best_estimator()
    def using_gridsearch(self):
        return GridSearchFactory().optimizer_builder(
                self.X,
                self.y,
                self.estimator,
                self.estimator_params,
                self.measure_of_accuracy,
                self.verbose,
                self.n_jobs,
                self.cv,
            ).optimize().get_best_estimator()
    def using_optunasearch(self):
        return OptunaFactory().optimizer_builder(
                self.X,
                self.y,
                self.verbose,
                self.random_state,
                self.estimator,
                self.estimator_params,
                # grid search and random search
                self.measure_of_accuracy,
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
            ).prepare_data().optimize().get_best_estimator()
    def return_engine(self):
        if self.type_engine == 'grid':
            return self.using_gridsearch()
        if self.type_engine == 'random':
            return self.using_randomsearch()
        if self.type_engine == 'optuna':
            print(self.using_optunasearch())
            return self.using_optunasearch()
        else:
            return None





        



            





