from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, r2_score
from lohrasb.best_estimator import BaseModel
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier

# Define hyperparameters for the classifiers and regressors
cat_params_reg = {"n_estimators": [50, 100, 200], "learning_rate": [0.001, 0.01, 0.1]}
lgbm_params = {"max_depth": [5, 6, 7, 10], "gamma": [0.01, 0.1, 1, 1.2]}


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
    obj = BaseModel().optimize_by_tunegridsearchcv(
        kwargs={
            "fit_tunegrid_kwargs": {"sample_weight": None},
            "tunegrid_search_kwargs": {
                "estimator": est,
                "param_grid": params,
                "scoring": "f1_micro",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
            "main_tunegrid_kwargs": {},
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
    obj = BaseModel().optimize_by_tunegridsearchcv(
        kwargs={
            "fit_tunegrid_kwargs": {"sample_weight": None},
            "tunegrid_search_kwargs": {
                "estimator": est,
                "param_grid": params,
                "scoring": "r2",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
            "main_tunegrid_kwargs": {},
        }
    )

    # Fit the model and make predictions
    obj.fit(X, y)
    predictions = obj.predict(X)

    # Evaluate and print the model performance
    r2 = r2_score(y, predictions)
    print(f"r2_score is {r2}")


# Run regression examples
using_tune_regression(CatBoostRegressor, cat_params_reg)

# Run classification examples
using_tune_classification(LGBMClassifier, lgbm_params)
