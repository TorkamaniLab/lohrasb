from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, r2_score
from lohrasb.best_estimator import BaseModel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Ridge

# Define hyperparameters for the classifiers and regressors
adb_params = {"n_estimators": [50, 100, 200], "learning_rate": [0.001, 0.01, 0.1]}
ridge_params_reg = {"fit_intercept": [True, False]}


def using_tune_classification(estimator, params):
    # Create synthetic dataset
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

    # Initialize the estimator
    est = estimator()

    # Create the model with the chosen hyperparameters
    obj = BaseModel().optimize_by_randomsearchcv(
        kwargs={
            "fit_random_kwargs": {
                "sample_weight": None,
            },
            "random_search_kwargs": {
                "estimator": est,
                "param_distributions": params,
                "scoring": "f1_micro",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
                "n_iter": 10,
            },
            "main_random_kwargs": {},
        }
    )

    # Fit the model and make predictions
    obj.fit(X_train, y_train)
    y_pred = obj.predict(X_test)

    # Evaluate the model performance
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"f1_score is {f1}")


def using_tune_regression(estimator, params):
    # Create synthetic regression dataset
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=1
    )

    # Initialize the estimator
    est = estimator()

    # Create the model with the chosen hyperparameters
    obj = BaseModel().optimize_by_randomsearchcv(
        kwargs={
            "fit_random_kwargs": {
                "sample_weight": None,
            },
            "random_search_kwargs": {
                "estimator": est,
                "param_distributions": params,
                "scoring": "r2",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
                "n_iter": 10,
            },
            "main_random_kwargs": {},
        }
    )

    # Fit the model and make predictions
    obj.fit(X, y)
    predictions = obj.predict(X)

    # Evaluate the model performance
    r2 = r2_score(y, predictions)
    print(f"r2_score is {r2}")


# Regression examples
using_tune_regression(Ridge, ridge_params_reg)

# Classification examples
using_tune_classification(AdaBoostClassifier, adb_params)
