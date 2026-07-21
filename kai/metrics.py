import numpy as np

def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
    return loss(y_true, y_pred) / y_true.size

def squared_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Squared Loss between true and predicted values.

    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.

    Returns:
    float: The sum of squared differences between true and predicted values.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    return np.sum((y_true - y_pred) ** 2)

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.

    Returns:
    float: The mean of squared differences between true and predicted values.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    return squared_loss(y_true, y_pred) / y_true.size

# Different from mean_squared_error, this function calculates the partial derivatives of the MSE with respect to weight and bias. Furthermore, 
# the subtraction of y_pred and y_true is reversed in the derivation, as it is used to calculate the gradients for weight and bias updates during training.
def mean_squared_error_derivation(y_true: np.ndarray, y_pred: np.ndarray, x_true: np.ndarray) -> tuple[float, float]:
    """
    Calculate the partial derivatives of the Mean Squared Error (MSE)
    with respect to the weight and the bias.

    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.
    x_true (np.ndarray): True feature values.

    Returns:
    tuple[float, float]: The (weight_derivation, bias_derivation) gradients.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    if y_true.shape != x_true.shape:
        raise ValueError("Shapes of y_true and x_true must be the same.")
    weight_derivation = np.sum((y_pred - y_true) * 2 * x_true) / y_true.size
    bias_derivation = np.sum((y_pred - y_true) * 2) / y_true.size
    return weight_derivation, bias_derivation