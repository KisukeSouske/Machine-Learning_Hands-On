import numpy as np

def calculate_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate loss between true and predicted values.

    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.

    Returns:
    float: The sum of absolute differences between true and predicted values.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    return np.sum(abs(y_true - y_pred))

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.

    Returns:
    float: The mean of absolute differences between true and predicted values.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    return calculate_loss(y_true, y_pred) / y_true.size