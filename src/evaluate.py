from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error

def rmse( y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Computes the root mean squared error (RMSE) between true and predicted values.

    Args:
        y_true (np.ndarray): Given target values of shape (n_samples, 1).
        y_predicted (np.ndarray): Predicted target values of shape (n_samples, 1).

    Returns:
        float: Root mean squared error.
    """
    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)

    return sqrt(mean_squared_error(y_true, y_predicted))