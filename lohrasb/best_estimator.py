from abc import ABCMeta

import xgboost
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator

from lohrasb.model_conf import (
    BLF_CLASSIFICATION_PARAMS_DEFAULT,
    CATBOOST_CLASSIFICATION_PARAMS_DEFAULT,
    CATBOOST_REGRESSION_PARAMS_DEFAULT,
    LGB_CLASSIFICATION_PARAMS_DEFAULT,
    LGB_REGRESSION_PARAMS_DEFAULT,
    LINEARREGRESSION_REGRESSION_PARAMS_DEFAULT,
    LOGISTICREGRESSION_CLASSIFICATION_PARAMS_DEFAULT,
    RANDOMFOREST_CLASSIFICATION_PARAMS_DEFAULT,
    RANDOMFOREST_REGRESSION_PARAMS_DEFAULT,
    SUPPORTED_MODELS,
    SVC_CLASSIFICATION_PARAMS_DEFAULT,
    XGBOOST_CLASSIFICATION_PARAMS_DEFAULT,
    XGBOOST_REGRESSION_PARAMS_DEFAULT,
)
from lohrasb.utils.helper_funcs import (
    _calc_best_estimator_grid_search,
    _calc_best_estimator_optuna_univariate,
    _calc_best_estimator_random_search,
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
        method. For CATBOOST_CLASSIFICATION_PARAMS_DEFAULT are :     "nan_mode": "Min",
        "eval_metric","iterations","sampling_frequency","leaf_estimation_method",
        "grow_policy","penalties_coefficient","boosting_type","model_shrink_mode",
        "feature_border_type","bayesian_matrix_reg","force_unit_auto_pair_weights",
        "l2_leaf_reg","random_strength","rsm","boost_from_average","model_size_reg",
        "pool_metainfo_options","subsample","use_best_model","class_names",
        "random_seed","depth","posterior_sampling","border_count",
        "classes_count","auto_class_weights","sparse_features_conflict_fraction",
        "leaf_estimation_backtracking","best_model_min_trees","model_shrink_rate",
        "min_data_in_leaf","loss_function","learning_rate","score_function",
        "task_type","leaf_estimation_iterations","bootstrap_type","max_leaves"

        For CATBOOST_REGRESSION_PARAMS_DEFAULT are :
        "nan_mode","eval_metric","iterations","sampling_frequency","leaf_estimation_method",
        "grow_policy","penalties_coefficient","boosting_type","model_shrink_mode",
        "feature_border_type","bayesian_matrix_reg","force_unit_auto_pair_weights",
        "l2_leaf_reg","random_strength","rsm","boost_from_average","model_size_reg",
        "pool_metainfo_options","subsample","use_best_model","random_seed","depth",
        "posterior_sampling","border_count","classes_count","auto_class_weights",
        "sparse_features_conflict_fraction","leaf_estimation_backtracking",
        "best_model_min_trees","model_shrink_rate","min_data_in_leaf",
        "loss_function","learning_rate","score_function","task_type",
        "leaf_estimation_iterations","bootstrap_type","max_leaves"

        For XGBOOST_CLASSIFICATION_PARAMS_DEFAULT are :
        "objective","use_label_encoder","base_score","booster",
        "callbacks","colsample_bylevel","colsample_bynode","colsample_bytree",
        "early_stopping_rounds","enable_categorical","eval_metric","gamma",
        "gpu_id","grow_policy","importance_type","interaction_constraints",
        "learning_rate","max_bin","max_cat_to_onehot","max_delta_step",
        "max_depth","max_leaves","min_child_weight","missing","monotone_constraints",
        "n_estimators","n_jobs","num_parallel_tree","predictor","random_state",
        "reg_alpha","reg_lambda","sampling_method","scale_pos_weight","subsample",
        "tree_method","validate_parameters","verbosity"

        For XGBOOST_REGRESSION_PARAMS_DEFAULT are :
        "objective","base_score","booster","callbacks","colsample_bylevel","colsample_bynode",
        "colsample_bytree","early_stopping_rounds","enable_categorical","eval_metric",
        "gamma","gpu_id","grow_policy","importance_type","interaction_constraints",
        "learning_rate","max_bin","max_cat_to_onehot","max_delta_step","max_depth",
        "max_leaves","min_child_weight","missing","monotone_constraints","n_estimators",
        "n_jobs","num_parallel_tree","predictor","random_state","reg_alpha","reg_lambda",
        "sampling_method","scale_pos_weight","subsample","tree_method","validate_parameters",
        "verbosity"

        For RANDOMFOREST_CLASSIFICATION_PARAMS_DEFAULT are :
        "n_estimators","criterion","max_depth","min_samples_split",
        "min_samples_leaf","min_weight_fraction_leaf","max_features",
        "max_leaf_nodes","min_impurity_decrease","bootstrap","oob_score",
        "n_jobs","random_state","verbose","warm_start","class_weight",
        "ccp_alpha","max_samples"

        For RANDOMFOREST_REGRESSION_PARAMS_DEFAULT are :
        "n_estimators","criterion","max_depth","min_samples_split",
        "min_samples_leaf","min_weight_fraction_leaf","max_features",
        "max_leaf_nodes","min_impurity_decrease","bootstrap","oob_score",
        "n_jobs","random_state","verbose","warm_start","ccp_alpha","max_samples"

        For BLF_CLASSIFICATION_PARAMS_DEFAULT are :
        "n_estimators","criterion"","max_depth","min_samples_split","min_samples_leaf",
        "min_weight_fraction_leaf","max_features","max_leaf_nodes","min_impurity_decrease",
        "bootstrap","oob_score","sampling_strategy","replacement","n_jobs","random_state",
        "verbose","warm_start","class_weight","ccp_alpha","max_samples"

        For LGB_CLASSIFICATION_PARAMS_DEFAULT are:
        "boosting_type","num_leaves","max_depth","learning_rate","n_estimators",
        "subsample_for_bin","objective","class_weight","min_split_gain","min_child_weight",
        "min_child_samples","subsample","subsample_freq","colsample_bytree","reg_alpha",
        "reg_lambda","random_state","n_jobs","silent","importance_type"

        For LGB_REGRESSION_PARAMS_DEFAULT are:
        "boosting_type","num_leaves","max_depth","learning_rate",
        "n_estimators","subsample_for_bin","objective","class_weight",
        "min_split_gain","min_child_weight","min_child_samples","subsample",
        "subsample_freq","colsample_bytree","reg_alpha","reg_lambda","random_state",
        "n_jobs","silent","importance_type"

        For SVC_CLASSIFICATION_PARAMS_DEFAULT are :
        "C", "kernel", "degree", "gamma", "coef0", "shrinking","probability",
        "tol", "cache_size", "class_weight", "verbose", "max_iter","decision_function_shape"

        For LINEARREGRESSION_REGRESSION_PARAMS_DEFAULT are :
        "fit_intercept", "normalize", "copy_X", "n_jobs",
        "positive"

        For LOGISTICREGRESSION_CLASSIFICATION_PARAMS_DEFAULT are :
        "penalty", "dual", "tol","C", "fit_intercept","intercept_scaling",
        "class_weightt", "random_state", "solver", "max_iter","multi_class",
        "verbose","warm_start", "n_jobs", "l1_ratio"

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
        estimator=None,
        estimator_params=None,
        hyper_parameter_optimization_method="optuna",
        measure_of_accuracy=None,
        test_size=0.33,
        cv=3,
        with_stratified=True,
        verbose=1,
        random_state=0,
        n_jobs=-1,
        n_iter=20,
        eval_metric="auc",
        number_of_trials=100,
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
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
            method. For CATBOOST_CLASSIFICATION_PARAMS_DEFAULT are :     "nan_mode": "Min",
            "eval_metric","iterations","sampling_frequency","leaf_estimation_method",
            "grow_policy","penalties_coefficient","boosting_type","model_shrink_mode",
            "feature_border_type","bayesian_matrix_reg","force_unit_auto_pair_weights",
            "l2_leaf_reg","random_strength","rsm","boost_from_average","model_size_reg",
            "pool_metainfo_options","subsample","use_best_model","class_names",
            "random_seed","depth","posterior_sampling","border_count",
            "classes_count","auto_class_weights","sparse_features_conflict_fraction",
            "leaf_estimation_backtracking","best_model_min_trees","model_shrink_rate",
            "min_data_in_leaf","loss_function","learning_rate","score_function",
            "task_type","leaf_estimation_iterations","bootstrap_type","max_leaves"

            For CATBOOST_REGRESSION_PARAMS_DEFAULT are :
            "nan_mode","eval_metric","iterations","sampling_frequency","leaf_estimation_method",
            "grow_policy","penalties_coefficient","boosting_type","model_shrink_mode",
            "feature_border_type","bayesian_matrix_reg","force_unit_auto_pair_weights",
            "l2_leaf_reg","random_strength","rsm","boost_from_average","model_size_reg",
            "pool_metainfo_options","subsample","use_best_model","random_seed","depth",
            "posterior_sampling","border_count","classes_count","auto_class_weights",
            "sparse_features_conflict_fraction","leaf_estimation_backtracking",
            "best_model_min_trees","model_shrink_rate","min_data_in_leaf",
            "loss_function","learning_rate","score_function","task_type",
            "leaf_estimation_iterations","bootstrap_type","max_leaves"

            For XGBOOST_CLASSIFICATION_PARAMS_DEFAULT are :
            "objective","use_label_encoder","base_score","booster",
            "callbacks","colsample_bylevel","colsample_bynode","colsample_bytree",
            "early_stopping_rounds","enable_categorical","eval_metric","gamma",
            "gpu_id","grow_policy","importance_type","interaction_constraints",
            "learning_rate","max_bin","max_cat_to_onehot","max_delta_step",
            "max_depth","max_leaves","min_child_weight","missing","monotone_constraints",
            "n_estimators","n_jobs","num_parallel_tree","predictor","random_state",
            "reg_alpha","reg_lambda","sampling_method","scale_pos_weight","subsample",
            "tree_method","validate_parameters","verbosity"

            For XGBOOST_REGRESSION_PARAMS_DEFAULT are :
            "objective","base_score","booster","callbacks","colsample_bylevel","colsample_bynode",
            "colsample_bytree","early_stopping_rounds","enable_categorical","eval_metric",
            "gamma","gpu_id","grow_policy","importance_type","interaction_constraints",
            "learning_rate","max_bin","max_cat_to_onehot","max_delta_step","max_depth",
            "max_leaves","min_child_weight","missing","monotone_constraints","n_estimators",
            "n_jobs","num_parallel_tree","predictor","random_state","reg_alpha","reg_lambda",
            "sampling_method","scale_pos_weight","subsample","tree_method","validate_parameters",
            "verbosity"

            For RANDOMFOREST_CLASSIFICATION_PARAMS_DEFAULT are :
            "n_estimators","criterion","max_depth","min_samples_split",
            "min_samples_leaf","min_weight_fraction_leaf","max_features",
            "max_leaf_nodes","min_impurity_decrease","bootstrap","oob_score",
            "n_jobs","random_state","verbose","warm_start","class_weight",
            "ccp_alpha","max_samples"

            For RANDOMFOREST_REGRESSION_PARAMS_DEFAULT are :
            "n_estimators","criterion","max_depth","min_samples_split",
            "min_samples_leaf","min_weight_fraction_leaf","max_features",
            "max_leaf_nodes","min_impurity_decrease","bootstrap","oob_score",
            "n_jobs","random_state","verbose","warm_start","ccp_alpha","max_samples"

            For BLF_CLASSIFICATION_PARAMS_DEFAULT are :
            "n_estimators","criterion"","max_depth","min_samples_split","min_samples_leaf",
            "min_weight_fraction_leaf","max_features","max_leaf_nodes","min_impurity_decrease",
            "bootstrap","oob_score","sampling_strategy","replacement","n_jobs","random_state",
            "verbose","warm_start","class_weight","ccp_alpha","max_samples"

            For LGB_CLASSIFICATION_PARAMS_DEFAULT are:
            "boosting_type","num_leaves","max_depth","learning_rate","n_estimators",
            "subsample_for_bin","objective","class_weight","min_split_gain","min_child_weight",
            "min_child_samples","subsample","subsample_freq","colsample_bytree","reg_alpha",
            "reg_lambda","random_state","n_jobs","silent","importance_type"

            For LGB_REGRESSION_PARAMS_DEFAULT are:
            "boosting_type","num_leaves","max_depth","learning_rate",
            "n_estimators","subsample_for_bin","objective","class_weight",
            "min_split_gain","min_child_weight","min_child_samples","subsample",
            "subsample_freq","colsample_bytree","reg_alpha","reg_lambda","random_state",
            "n_jobs","silent","importance_type"
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

        self.estimator = estimator
        self.estimator_params = estimator_params
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
        self.measure_of_accuracy = measure_of_accuracy
        self.test_size = test_size
        self.cv = cv
        self.with_stratified = with_stratified
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.eval_metric = eval_metric
        self.number_of_trials = number_of_trials
        self.sampler = sampler
        self.pruner = pruner
        self.best_estimator = None
        self.importance_df = None

    @property
    def estimator(self):
        print("Getting value for estimator")
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        print("Setting value for estimator")
        if value.__class__.__name__ not in SUPPORTED_MODELS:

            raise TypeError(
                f"{value.__class__.__name__} \
                 model is not supported yet"
            )
        self._estimator = value

    @property
    def estimator_params(self):
        print("Getting value for estimator_params")
        return self._estimator_params

    @estimator_params.setter
    def estimator_params(self, value):
        print(self.estimator)
        # get parameters for SVC and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "SVC":
            if value.keys() <= SVC_CLASSIFICATION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for LinearRegression and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "LinearRegression":
            if value.keys() <= LINEARREGRESSION_REGRESSION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for LogisticRegression and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "LogisticRegression":
            if value.keys() <= LOGISTICREGRESSION_CLASSIFICATION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for lightgbm.LGBMRegressor and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "LGBMRegressor":
            if value.keys() <= LGB_REGRESSION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for lightgbm.LGBMClassifier and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "LGBMClassifier":
            if value.keys() <= LGB_CLASSIFICATION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for XGBRegressor and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "BalancedRandomForestClassifier":
            if value.keys() <= BLF_CLASSIFICATION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for XGBRegressor and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "RandomForestRegressor":
            if value.keys() <= RANDOMFOREST_REGRESSION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for XGBRegressor and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "RandomForestClassifier":
            if value.keys() <= RANDOMFOREST_CLASSIFICATION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )

        # get parameters for XGBRegressor and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "XGBRegressor":
            if value.keys() <= XGBOOST_REGRESSION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for XGBClassifier and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "XGBClassifier":
            if value.keys() <= XGBOOST_CLASSIFICATION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )

        # get parameters for CatBoostClassifier and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "CatBoostClassifier":
            if value.keys() <= CATBOOST_CLASSIFICATION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for CatBoostRegressor and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "CatBoostRegressor":
            if value.keys() <= CATBOOST_REGRESSION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )

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
        if self.hyper_parameter_optimization_method.lower() == "grid":
            self.best_estimator = _calc_best_estimator_grid_search(
                X,
                y,
                self.estimator,
                self.estimator_params,
                self.measure_of_accuracy,
                self.verbose,
                self.n_jobs,
                self.cv,
            )
        if self.hyper_parameter_optimization_method.lower() == "random":
            self.best_estimator = _calc_best_estimator_random_search(
                X,
                y,
                self.estimator,
                self.estimator_params,
                self.measure_of_accuracy,
                self.verbose,
                self.n_jobs,
                self.n_iter,
                self.cv,
            )
        if self.hyper_parameter_optimization_method.lower() == "optuna":
            self.best_estimator = _calc_best_estimator_optuna_univariate(
                X,
                y,
                self.estimator,
                self.measure_of_accuracy,
                self.estimator_params,
                self.verbose,
                self.test_size,
                self.random_state,
                self.eval_metric,
                self.number_of_trials,
                self.sampler,
                self.pruner,
                self.with_stratified,
            )

        return self

    def predict(self, X):
        """Predict using the best estimator model.
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection
            step of the pipeline.
        """
        if (
            self.estimator.__class__.__name__ == "XGBRegressor"
            or self.estimator.__class__.__name__ == "XGBClassifier"
            and self.hyper_parameter_optimization_method == "optuna"
        ):
            X = xgboost.DMatrix(X)
        return self.best_estimator.predict(X)
