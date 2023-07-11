# Import necessary libraries
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor

from lohrasb.best_estimator import BaseModel

# Define hyperparameters for the MLPRegressor and LGBMClassifier
# These will be the values that the hyperparameter search function will iterate through.
mlp_params_reg = {
    "hidden_layer_sizes": [(5, 5, 5), (5, 10, 5), (10,)],
    "activation": ["tanh", "relu"],
    "solver": ["sgd", "adam"],
    "alpha": [0.0001, 0.05],
    "learning_rate": ["constant", "adaptive"],
}
lgbm_params = {"max_depth": [5, 6, 7, 10]}


# Function for training and evaluating a classification model
def using_tune_classification(estimator, params):
    # Create a synthetic classification dataset with 1000 samples, 20 features, and 3 classes
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=3,
        n_redundant=10,
        n_classes=3,
        random_state=42,
    )
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the estimator
    est = estimator()

    # Use the hyperparameter search function provided by the BaseModel class to find the best parameters
    obj = BaseModel().optimize_by_tunesearchcv(
        kwargs={
            "fit_tune_kwargs": {"sample_weight": None},
            "tune_search_kwargs": {
                "estimator": est,
                "param_distributions": params,
                "scoring": "f1_micro",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
            "main_tune_kwargs": {},
        }
    )

    # Fit the model to the training data
    obj.fit(X_train, y_train)
    # Predict the labels for the test data
    y_pred = obj.predict(X_test)

    # Compute the F1 score of the model
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"f1_score is {f1}")


# Function for training and evaluating a regression model
def using_tune_regression(estimator, params):
    # Create a synthetic regression dataset with 1000 samples and 10 features
    X, y = make_regression(
        n_samples=1000, n_features=10, n_informative=5, n_targets=1, random_state=1
    )

    # Initialize the estimator
    est = estimator()

    # Use the hyperparameter search function provided by the BaseModel class to find the best parameters
    obj = BaseModel().optimize_by_tunesearchcv(
        kwargs={
            "fit_tune_kwargs": {},
            "tune_search_kwargs": {
                "estimator": est,
                "param_distributions": params,
                "scoring": "r2",
                "verbose": 3,
                "n_jobs": -1,
                "cv": KFold(2),
            },
            "main_tune_kwargs": {},
        }
    )

    # Fit the model to the data
    obj.fit(X, y)
    # Predict the targets for the data
    predictions = obj.predict(X)

    # Compute the R2 score of the model
    r2 = r2_score(y, predictions)
    print(f"r2_score is {r2}")


# Run the regression function using the MLPRegressor and the specified parameters
using_tune_regression(MLPRegressor, mlp_params_reg)

# Run the classification function using the LGBMClassifier and the specified parameters
using_tune_classification(LGBMClassifier, lgbm_params)
