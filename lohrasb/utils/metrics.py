import inspect
import os

import numpy as np
from sklearn.metrics import *

from lohrasb import logger

# for load Environment Variables
# True there will not be default args for metric


def f1_plus_tp(y_true, y_pred):
    """Return f1_score+True Positive

    This function calculates the F1 score and adds the count of true positives to it.

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        The true values.
    y_pred : Pandas DataFrame or Pandas Series
        The predicted values.
    """

    # Calculate the F1 score using the f1_score function
    f1 = f1_score(y_true, y_pred)

    # Compute the confusion matrix using the confusion_matrix function
    # The labels argument specifies the order of the labels in the matrix
    # Here, it is set to [0, 1] indicating that label 0 corresponds to the first row, and label 1 corresponds to the second row
    _, _, _, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Add the count of true positives to the F1 score
    return f1 + tp


def f1_plus_tn(y_true, y_pred):
    """Return f1_score + True Negative

    This function calculates the F1 score and adds the count of true negatives to it.

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        The true values.
    y_pred : Pandas DataFrame or Pandas Series
        The predicted values.
    """

    # Calculate the F1 score using the f1_score function
    f1 = f1_score(y_true, y_pred)

    # Compute the confusion matrix using the confusion_matrix function
    # The labels argument specifies the order of the labels in the matrix
    # Here, it is set to [0, 1] indicating that label 0 corresponds to the first row, and label 1 corresponds to the second row
    tn, _, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Add the count of true negatives to the F1 score
    return f1 + tn


def specificity(y_true, y_pred):
    """Return Specificity

    This function calculates the specificity, which is the true negative rate.

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        The true values.
    y_pred : Pandas DataFrame or Pandas Series
        The predicted values.
    """

    # Compute the confusion matrix using the confusion_matrix function
    # The labels argument specifies the order of the labels in the matrix
    # Here, it is set to [0, 1] indicating that label 0 corresponds to the first row, and label 1 corresponds to the second row
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Calculate specificity by dividing true negatives by the sum of true negatives and false positives
    return tn / (tn + fp)


def tn_score(y_true, y_pred):
    """Return True Negative (TN) Score

    This function calculates the count of true negatives.

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        The true values.
    y_pred : Pandas DataFrame or Pandas Series
        The predicted values.
    """

    # Compute the confusion matrix using the confusion_matrix function
    # The labels argument specifies the order of the labels in the matrix
    # Here, it is set to [0, 1] indicating that label 0 corresponds to the first row, and label 1 corresponds to the second row
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Return the count of true negatives
    return tn


def tn(y_true, y_pred):
    """Return True Negative (TN)

    This function calculates the count of true negatives.

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        The true values.
    y_pred : Pandas DataFrame or Pandas Series
        The predicted values.
    """

    # Compute the confusion matrix using the confusion_matrix function
    # The labels argument specifies the order of the labels in the matrix
    # Here, it is set to [0, 1] indicating that label 0 corresponds to the first row, and label 1 corresponds to the second row
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Return the count of true negatives
    return tn


def tp_score(y_true, y_pred):
    """Return True Positive (TP) Score

    This function calculates the count of true positives.

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        The true values.
    y_pred : Pandas DataFrame or Pandas Series
        The predicted values.
    """

    # Compute the confusion matrix using the confusion_matrix function
    # The labels argument specifies the order of the labels in the matrix
    # Here, it is set to [0, 1] indicating that label 0 corresponds to the first row, and label 1 corresponds to the second row
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Return the count of true positives
    return tp


def tp(y_true, y_pred):
    """Return True Positive (TP)

    This function calculates the count of true positives.

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        The true values.
    y_pred : Pandas DataFrame or Pandas Series
        The predicted values.
    """

    # Compute the confusion matrix using the confusion_matrix function
    # The labels argument specifies the order of the labels in the matrix
    # Here, it is set to [0, 1] indicating that label 0 corresponds to the first row, and label 1 corresponds to the second row
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Return the count of true positives
    return tp


def roc_plus_f1(y_true, y_pred):
    """Return ROC + F1 Score

    This function calculates the sum of the F1 score and the ROC curve.

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        The true values.
    y_pred : Pandas DataFrame or Pandas Series
        The predicted values.
    """

    # Calculate the F1 score using the f1_score function
    f1 = f1_score(y_true, y_pred)

    # Compute the ROC curve using the roc_curve function
    roc = roc_curve(y_true, y_pred)

    # Return the sum of the F1 score and the ROC curve
    return np.sum(f1 + roc)


def auc_plus_f1(y_true, y_pred):
    """Return AUC + F1 Score

    This function calculates the sum of the F1 score and the Area Under the ROC Curve (AUC).

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        The true values.
    y_pred : Pandas DataFrame or Pandas Series
        The predicted values.
    """

    # Calculate the F1 score using the f1_score function
    f1 = f1_score(y_true, y_pred)

    # Compute the false positive rate (fpr), true positive rate (tpr), and thresholds for the ROC curve using the roc_curve function
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Calculate the AUC using the auc function
    auc_result = auc(fpr, tpr)

    # Return the sum of the F1 score and the AUC
    return f1 + auc_result


def det_curve_ret(y_true, y_pred):
    """Return DET Curve

    This function calculates the sum of the false positive rates (fpr) and false negative rates (fnr) for the Detection Error Tradeoff (DET) curve.

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        The true values.
    y_pred : Pandas DataFrame or Pandas Series
        The predicted values.
    """

    # Compute the false positive rate (fpr), false negative rate (fnr), and thresholds for the DET curve using the det_curve function
    fpr, fnr, thresholds = det_curve(y_true, y_pred)

    # Return the sum of the false positive rates (fpr) and false negative rates (fnr)
    return np.sum(fpr) + np.sum(fnr)


def precision_recall_curve_ret(y_true, y_pred):
    """Return Precision-Recall Curve

    This function calculates the sum of the precision and recall values for the Precision-Recall curve.

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        The true values.
    y_pred : Pandas DataFrame or Pandas Series
        The predicted values.
    """

    # Compute the precision, recall, and thresholds for the Precision-Recall curve using the precision_recall_curve function
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    # Return the sum of the precision and recall values
    return np.sum(precision) + np.sum(recall)


def precision_recall_fscore_support_ret(y_true, y_pred):
    """Return Precision, Recall, F1-Score, and Support

    This function calculates the sum of the values returned by the precision_recall_fscore_support function.

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        The true values.
    y_pred : Pandas DataFrame or Pandas Series
        The predicted values.
    """

    # Compute the precision, recall, F1-score, and support using the precision_recall_fscore_support function
    output = precision_recall_fscore_support(y_true, y_pred)

    # Return the sum of the values returned by precision_recall_fscore_support
    return np.sum(output)


def roc_curve_ret(y_true, y_pred):
    """Return ROC Curve

    This function calculates the difference between the sum of the true positive rates (tpr) and the sum of the false positive rates (fpr) for the ROC curve.

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        The true values.
    y_pred : Pandas DataFrame or Pandas Series
        The predicted values.
    """

    # Compute the false positive rate (fpr), true positive rate (tpr), and thresholds for the ROC curve using the roc_curve function
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Return the difference between the sum of the true positive rates (tpr) and the sum of the false positive rates (fpr)
    return np.sum(tpr) - np.sum(fpr)


class CalcMetrics:
    """
    Class for calculating metrics.

    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values.
    y_pred : Pandas DataFrame or Pandas Series
        Predicted values.
    metric : str
        Name of the metric function.
    d : dict, optional
        A mapping dictionary that maps a metric function's string name to its representative function, by default None.
    change_default_args_of_metric : bool, optional
        Flag to change the default arguments of the metric function, by default False.
    *args, **kwargs
        Additional arguments to be passed to the metric function.

    """

    def __init__(
        self,
        y_true,
        y_pred,
        metric,
        d={
            "accuracy_score": accuracy_score,  # normal
            "auc": auc,  # normal
            "precision_recall_curve": precision_recall_curve,  # normal
            "balanced_accuracy_score": balanced_accuracy_score,  # normal
            "cohen_kappa_score": cohen_kappa_score,  # normal
            "dcg_score": dcg_score,  # normal
            "det_curve": det_curve_ret,  # normal minimize
            "f1_score": f1_score,  # normal
            "fbeta_score": fbeta_score,  # normal
            "hamming_loss": hamming_loss,  # normal minimize
            "jaccard_score": jaccard_score,  # normal
            "matthews_corrcoef": matthews_corrcoef,  # normal
            "ndcg_score": ndcg_score,  # normal
            "precision_score": precision_score,  # normal
            "recall_score": recall_score,  # normal
            "recall": recall_score,  # normal
            "roc_auc_score": roc_auc_score,  # normal
            "roc_curve": roc_curve_ret,  # normal
            "top_k_accuracy_score": top_k_accuracy_score,  # normal
            "zero_one_loss": zero_one_loss,  # normal minimize
            # customs
            "tn": tn,  # custom
            "tp": tp,  # custom
            "tn_score": tn_score,  # custom
            "tp_score": tp_score,  # custom
            "f1_plus_tp": f1_plus_tp,  # custom
            "f1_plus_tn": f1_plus_tn,  # custom
            "specificity": specificity,  # custom
            "roc_plus_f1": roc_plus_f1,  # custom
            "auc_plus_f1": auc_plus_f1,  # custom
            "precision_recall_curve_ret": precision_recall_curve_ret,  # custom
            "precision_recall_fscore_support": precision_recall_fscore_support_ret,  # custom
            # regression
            "explained_variance_score": explained_variance_score,
            "max_error": max_error,
            "mean_absolute_error": mean_absolute_error,
            "mean_squared_log_error": mean_squared_log_error,
            "mean_absolute_percentage_error": mean_absolute_percentage_error,
            "median_absolute_error": median_absolute_error,
            "r2_score": r2_score,
            "mean_poisson_deviance": mean_poisson_deviance,
            "mean_gamma_deviance": mean_gamma_deviance,
            "mean_tweedie_deviance": mean_tweedie_deviance,
            "d2_tweedie_score": d2_tweedie_score,
            "mean_pinball_loss": mean_pinball_loss,
            "d2_pinball_score": d2_pinball_score,
            "d2_absolute_error_score": d2_absolute_error_score,
        },
        change_default_args_of_metric=False,
        *args,
        **kwargs,
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.metric = metric
        self.d = d
        self.change_default_args_of_metric = change_default_args_of_metric
        self.args = args
        self.kwargs = kwargs

    def resolve_name(self):
        """
        Resolve the metric function based on the given metric name.
        """
        if self.metric in self.d:
            return self.d[self.metric]

    def get_func_args(self):
        """
        Get the arguments of the metric function.
        """
        args_of_func = inspect.signature(self.resolve_name()).parameters
        return args_of_func

    def get_func_default_args(self):
        """
        Get the default arguments of the metric function.
        """
        args_of_func = self.get_func_args()
        d = {}
        for name, value in args_of_func.items():
            value_str = str(value)
            if "=" in value_str and value not in [
                "y_true",
                "y_score",
                "y_pred",
                "y_prob",
            ]:
                d[name] = value
        func_default_args = d
        return func_default_args

    def get_transformed_default_args(self):
        """
        Get the transformed default arguments of the metric function.
        """
        func_default_args = self.get_func_default_args()
        d = {}
        for name, value in func_default_args.items():
            value_str = str(value)
            if "None" in value_str:
                d[name] = None
            elif "True" in value_str:
                d[name] = True
            elif "False" in value_str:
                d[name] = False
            elif "'" in value_str and "=" in value_str:
                start = end = "'"
                d[name] = value_str.split(start)[1].split(end)[0]
            elif "'" not in value_str and "=" in value_str:
                start = end = "="
                d[name] = value_str.split(start)[1].split(end)[0]
            for k, v in func_default_args.items():
                try:
                    start = end = "="
                    str_var = str(v).split(start)[1].split(end)[0]
                    float_str = float(str_var)
                    d[k] = float_str
                except ValueError:
                    logger.warning(f"Warning! {k} is not used in {self.metric}")
                try:
                    start = end = "="
                    str_var = str(v).split(start)[1].split(end)[0]
                    int_str = int(str_var)
                    d[k] = int_str
                except ValueError:
                    logger.warning(f"Warning! {k} is not used in {self.metric}")

        transformed_default_args = d
        return transformed_default_args

    def assign_default(self):
        """
        Provide a way for users to change the default values of the metric function's default arguments.
        """
        metric = self.resolve_name()
        transformed_default_args = self.get_transformed_default_args()
        if self.change_default_args_of_metric:
            if len(transformed_default_args) > 0:
                ans = input(
                    f"Do you want to change the default arguments for {str(metric.__name__)}? (Y/N): "
                )
                if ans.lower() in ["y", "yes"]:
                    self.change_default_args_of_metric = True
                else:
                    self.change_default_args_of_metric = False
                if self.change_default_args_of_metric:
                    for t, v in transformed_default_args.items():
                        value_input = input(
                            f"Set a value for {t} of {str(metric.__name__)}: "
                        )
                        if value_input == "":
                            transformed_default_args[t] = v
                        elif value_input.lower() in ["true", "false"]:
                            transformed_default_args[t] = value_input.lower() == "true"
                        elif value_input.lower() == "none":
                            transformed_default_args[t] = None
                        else:
                            transformed_default_args[t] = str(value_input)

                    for t, v in transformed_default_args.items():
                        if not isinstance(v, bool):
                            try:
                                float_v = float(v)
                                transformed_default_args[t] = float_v
                            except Exception as e:
                                logger.warning(
                                    f"Warning! {t} is not used in {self.metric}"
                                )
                            try:
                                int_v = int(v)
                                transformed_default_args[t] = int_v
                            except Exception as e:
                                logger.warning(
                                    f"Warning! {t} is not used in {self.metric}"
                                )

        return transformed_default_args

    def get_default_params_if_any(self):
        """
        Get the default parameters of the metric function, if any.
        """
        assign_default = self.assign_default()
        return assign_default

    def get_func_string_if_any(self):
        """
        Get the metric function as a string, if any.
        """
        metric = self.resolve_name()
        func = str(metric.__name__)
        return func

    def get_metric_func(self):
        """
        Get the metric function and its default assigned arguments.
        """
        assign_default = self.assign_default()
        metric = self.resolve_name()
        func = str(metric.__name__)
        f_str = func + "(" + "self.y_true, self.y_pred, **assign_default" + ")"
        return eval(f_str)

    def calc_make_scorer(self, metric):
        """
        Calculate the function body of a function using make_scorer.
        See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html

        Parameters
        ----------
        metric : str
            String representation of a metric function.
        """
        metrics_list_for_maximizing = [
            "accuracy_score",
            "auc",
            "precision_recall_curve",
            "balanced_accuracy_score",
            "cohen_kappa_score",
            "dcg_score",
            "f1_score",
            "fbeta_score",
            "fbeta_score",
            "jaccard_score",
            "matthews_corrcoef",
            "ndcg_score",
            "precision_score",
            "recall_score",
            "roc_auc_score",
            "roc_curve",
            "top_k_accuracy_score",
            # customs
            "tn",
            "tn_score",
            "tp",
            "tp_score",
            "f1_plus_tp",
            "f1_plus_tn",
            "specificity",
            "roc_plus_f1",
            "auc_plus_f1",
            "precision_recall_curve",
            "precision_recall_fscore_support",
            "max_error",
            "r2_score",
        ]
        metrics_list_for_minimizing = [
            "det_curve",
            "hamming_loss",
            "zero_one_loss",
            "explained_variance_score",
            "mean_absolute_error",
            "mean_squared_log_error",
            "mean_absolute_percentage_error",
            "mean_squared_log_error",
            "median_absolute_error",
            "mean_absolute_percentage_error",
            "mean_poisson_deviance",
            "mean_gamma_deviance",
            "mean_tweedie_deviance",
            "d2_tweedie_score",
            "mean_pinball_loss",
            "d2_pinball_score",
            "d2_absolute_error_score",
        ]

        if (
            metric.__class__.__name__ == "_BaseScorer"
            or metric.__class__.__name__ == "_ProbaScorer"
            or metric.__class__.__name__ == "_PredictScorer"
            or metric.__class__.__name__ == "_ThresholdScorer"
        ):
            return metric
        elif isinstance(metric, str):
            if "(" in metric:
                metric = metric.replace("y_true", "self.y_test")
                return metric
            else:
                raise TypeError(
                    f"The selected metric {metric} is not proper. Read the examples and documentation!"
                )
        else:
            raise TypeError(
                f"The selected metric {metric} is not proper. Read the examples and documentation!"
            )

    def get_simple_metric(self, metric, y_true, y_pred, params=None):
        """
        Get the simple metric value for y_true and y_pred.

        Parameters
        ----------
        metric : str
            String representation of a metric function.
        y_true : Pandas DataFrame or Pandas Series
            True values.
        y_pred : Pandas DataFrame or Pandas Series
            Predicted values.
        params : dict, optional
            Additional parameters for the metric function, by default None.
        """
        if params is None:
            f_str = metric + "(" + "y_true, y_pred" + ")"
            return eval(f_str)
        if isinstance(params, dict) and len(params) == 0:
            f_str = metric + "(" + "y_true, y_pred" + ")"
            return eval(f_str)
        if isinstance(params, dict) and len(params) > 0:
            f_str = metric + "(" + "y_true, y_pred, **params" + ")"
            return eval(f_str)
        return None


if __name__ == "__main__":
    # Test metrics
    calcmetric = CalcMetrics(
        y_true=np.array([0, 0, 1, 0]),
        y_pred=np.array([0, 1, 1, 0]),
        metric="d2_absolute_error_score",
    )
    calcmetric.get_metric_func()
    func = calcmetric.calc_make_scorer("d2_absolute_error_score")
