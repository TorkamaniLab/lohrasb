import subprocess
from lohrasb.utils.helper_funcs import (
    _trail_params_retrive,
    _calc_metric_for_single_output_classification,
    _calc_metric_for_single_output_regression,
    install_and_import,
    import_from,
    maping_mesurements,
    )
from lohrasb.decorators.decorators import trackcalls
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,train_test_split
from sklearn.metrics import (
    make_scorer,
)
from lohrasb.abstracts.optimizers import OptimizerABC
from lohrasb.factories.factories import OptimizerFactory
import numpy as np
from sklearn.linear_model import *
from sklearn.svm import *
from xgboost import *
from sklearn.linear_model import *
from catboost import *
from lightgbm import *
from sklearn.neural_network import *
from imblearn.ensemble import *
from sklearn.ensemble import *

class OptunaSearch(OptimizerABC):
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
        
        self.X = X
        self.y = y
        self.verbose =verbose
        self.random_state=random_state
        self.estimator=estimator
        self.estimator_params=estimator_params
        # grid search and random search
        self.measure_of_accuracy=measure_of_accuracy
        self.n_jobs=n_jobs
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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.objective = None
        self.trial = None

    def prepare_data(self):
        if self.with_stratified:
             self.X_train,  self.X_test, self.y_train, self.y_test = \
                 train_test_split(self.X, self.y, test_size=self.test_size,\
                     stratify=self.y[self.y.columns.to_list()[0]],\
                        random_state=self.random_state)
        else:
             self.X_train,  self.X_test, self.y_train, self.y_test = \
                 train_test_split(self.X, self.y, test_size=self.test_size,\
                        random_state=self.random_state)

        return self
    def optimize(self):
        def objective(trial):
            
            params = _trail_params_retrive(trial, self.estimator_params)
            print(params)
            est = eval(self.estimator.__class__.__name__+'(**params)'+'.fit(self.X_train, self.y_train)')
            preds = est.predict(self.X_test)
            pred_labels = np.rint(preds)
        
            if self.measure_of_accuracy in [
                     "f1" , "f1_score", "acc", "accuracy_score", "accuracy","pr", 
                     "precision_score", "precision","recall", "recall_score","recall",
                      "roc", "roc_auc_score","roc_auc","tp" , "true possitive","tn" 
                      ,"true negative"
                      ]:
                        accr = _calc_metric_for_single_output_classification(
                        self.y_test, pred_labels, self.measure_of_accuracy)
            elif self.measure_of_accuracy in [
                      "r2", "r2_score", "explained_variance_score", 
                      "max_error", "mean_absolute_error", "mean_squared_error", 
                      "median_absolute_error","mean_absolute_percentage_error"
                      ]:
                        accr = _calc_metric_for_single_output_regression(self.y_test, \
                            preds, self.measure_of_accuracy)

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
        return self.study.best_trial

    def get_best_estimator(self):
        self.estimator = eval(self.estimator.__class__.__name__+'(**self.trial.params)')
        self.best_estimator = self.estimator.fit(self.X_train, self.y_train)
        return self.best_estimator


class GridSearch(OptimizerABC):
    def __init__(
        self,
        X,
        y,
        estimator,
        estimator_params,
        measure_of_accuracy,
        verbose,
        n_jobs,
        cv,
    ):
        self.X = X
        self.y = y
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.measure_of_accuracy = measure_of_accuracy
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.cv = cv
        self.grid_search = None
        self.best_estimator = None

    def prepare_data(self):
        pass

    @trackcalls
    def optimize(self):
        self.grid_search = GridSearchCV(
            self.estimator,
            param_grid=self.estimator_params,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring=make_scorer(maping_mesurements[self.measure_of_accuracy]),
            verbose=self.verbose,
        )
        self.grid_search.fit(self.X, self.y)
        self.best_estimator = self.grid_search.best_estimator_
        return self
    @trackcalls
    def get_best_estimator(self, *args, **kwargs):
        if self.optimize.has_been_called and self.best_estimator is not None:
            return self.best_estimator
        else:
            self.best_estimator, self.random_search = self.optimize(
            self.estimator,
            param_grid=self.estimator_params,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring=make_scorer(maping_mesurements[self.measure_of_accuracy]),
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
    def __init__(
        self,
        X,
        y,
        estimator,
        estimator_params,
        measure_of_accuracy,
        verbose,
        n_jobs,
        n_iter,
        cv,
    ):
        self.X = X
        self.y = y
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.measure_of_accuracy = measure_of_accuracy
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.cv = cv
        self.random_search = None
        self.best_estimator = None

    def prepare_data(self):
        pass
    @trackcalls
    def optimize(self):
        self.random_search = RandomizedSearchCV(
            self.estimator,
            param_distributions=self.estimator_params,
            cv=self.cv,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
            scoring=make_scorer(maping_mesurements[self.measure_of_accuracy]),
            verbose=self.verbose,
        )

        self.random_search.fit(self.X, self.y)
        self.best_estimator = self.random_search.best_estimator_
        return self

    def get_best_estimator(self, *args, **kwargs):
        if self.optimize.has_been_called and self.best_estimator is not None:
            return self.best_estimator
        else:
            self.best_estimator, self.random_search = self.optimize(
            self.estimator,
            param_distributions=self.estimator_params,
            cv=self.cv,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
            scoring=make_scorer(maping_mesurements[self.measure_of_accuracy]),
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
                "RandomSearch has not been implemented \
                or best_estomator is null"
            )


class GridSearchFactory(OptimizerFactory):
    """Factory for building GridSeachCv."""

    def optimizer_builder(
        self,
        X,
        y,
        estimator,
        estimator_params,
        measure_of_accuracy,
        verbose,
        n_jobs,
        cv,
    ):
        print("Initializing GridSEarchCV")
        return GridSearch(
            X,
            y,
            estimator,
            estimator_params,
            measure_of_accuracy,
            verbose,
            n_jobs,
            cv,
        )

class OptunaFactory(OptimizerFactory):
    """Factory for building Optuna."""

    def optimizer_builder(
        self,
        X,
        y,
        verbose,
        random_state,
        estimator,
        estimator_params,
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
        print("Initializing Optuna")
        return OptunaSearch(
            X,
            y,
            verbose,
            random_state,
            estimator,
            estimator_params,
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
        )


class RandomSearchFactory(OptimizerFactory):
    """Factory for building GridSeachCv."""

    def optimizer_builder(
        self,
        X,
        y,
        estimator,
        estimator_params,
        measure_of_accuracy,
        verbose,
        n_jobs,
        n_iter,
        cv,
    ):
        print("Initializing RandomSeachCV")
        return RandomSearch(
            X,
            y,
            estimator,
            estimator_params,
            measure_of_accuracy,
            verbose,
            n_jobs,
            n_iter,
            cv,
        )
