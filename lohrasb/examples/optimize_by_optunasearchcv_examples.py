import optuna
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold, train_test_split

from lohrasb.best_estimator import BaseModel

# Define hyperparameters for the classifiers and regressors
adb_params = {
    "n_estimators": optuna.distributions.IntDistribution(50, 200),
    "learning_rate": optuna.distributions.FloatDistribution(0.001, 0.1),
}
ridge_params_reg = {
    "fit_intercept": optuna.distributions.CategoricalDistribution(choices=[True, False])
}


def using_tune_classification(estimator, params):
    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=3,
        n_redundant=10,
        n_classes=3,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the estimator and create a model with the specified hyperparameters
    est = estimator()
    obj = BaseModel().optimize_by_optunasearchcv(
        kwargs={
            "fit_newoptuna_kwargs": {"sample_weight": None},
            "newoptuna_search_kwargs": {
                "estimator": est,
                "param_distributions": params,
                "scoring": "f1_micro",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
            "main_newoptuna_kwargs": {},
        }
    )

    # Fit the model and make predictions
    obj.fit(X_train, y_train)
    y_pred = obj.predict(X_test)

    # Evaluate and print the model performance
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"f1_score is {f1}")


def using_tune_regression(estimator, params):
    # Create synthetic regression dataset
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1
    )

    # Initialize the estimator and create a model with the specified hyperparameters
    est = estimator()
    obj = BaseModel().optimize_by_optunasearchcv(
        kwargs={
            "fit_newoptuna_kwargs": {"sample_weight": None},
            "newoptuna_search_kwargs": {
                "estimator": est,
                "param_distributions": params,
                "scoring": "r2",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
            "main_newoptuna_kwargs": {},
        }
    )

    # Fit the model and make predictions
    obj.fit(X, y)
    predictions = obj.predict(X)

    # Evaluate and print the model performance
    r2 = r2_score(y, predictions)
    print(f"r2_score is {r2}")


# Run regression examples
using_tune_regression(Ridge, ridge_params_reg)

# Run classification examples
using_tune_classification(AdaBoostClassifier, adb_params)
