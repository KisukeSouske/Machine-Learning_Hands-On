import numpy as np

def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Total absolute error: sums the magnitudes of the residuals, showing the overall distance between predictions and observations.
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
    # Mean Absolute Error (MAE): the average absolute residual, representing the typical error in the same units as the target variable.
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
    # Squared loss (L2): sums squared residuals, giving more weight to larger mistakes than absolute-error metrics.
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
    # Mean Squared Error (MSE): the average squared residual, commonly used to assess regression fit and to optimize linear models.
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

# Mean Squared Error derivative: the gradient of the MSE with respect to the model parameters, used to update weights and bias during gradient descent.
def mean_squared_error_derivation(y_true: np.ndarray, y_pred: np.ndarray, x_true: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Calculate the partial derivatives of the Mean Squared Error (MSE)
    with respect to the weights and the bias.

    Parameters:
    y_true (np.ndarray): True values, shape (n_samples,).
    y_pred (np.ndarray): Predicted values, shape (n_samples,).
    x_true (np.ndarray): Feature values, shape (n_samples, n_features).

    Returns:
    tuple[np.ndarray, float]: The (weight_derivation, bias_derivation) gradients.
        weight_derivation has shape (n_features,).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    if x_true.shape[0] != y_true.shape[0]:
        raise ValueError("x_true must have the same number of samples as y_true.")

    error = y_pred - y_true
    weight_derivation = (x_true.T @ error) * 2 / y_true.size
    bias_derivation = np.sum(error * 2) / y_true.size
    return weight_derivation, bias_derivation

def total_sum_of_squares(y_true: np.ndarray) -> float:
    # Total Sum of Squares (TSS): measures the total variability of the target values around their mean, used as the baseline for R-squared.
    """
    Calculate the Total Sum of Squares (TSS) for true values.

    Parameters:
    y_true (np.ndarray): True values.

    Returns:
    float: The total sum of squares.
    """
    return np.sum((y_true - np.mean(y_true)) ** 2)

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # R-squared: the proportion of variance explained by the model; values near 1 indicate a good fit, while values below 0 mean the model is worse than simply predicting the mean.
    """
    Calculate the R-squared (coefficient of determination) between true and predicted values.

    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.

    Returns:
    float: The R-squared value.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    rss = squared_loss(y_true, y_pred)
    tss = total_sum_of_squares(y_true)
    return 1 - (rss / tss)

def adjusted_r_squared(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    # Adjusted R-squared: penalizes the addition of predictors so the metric rewards explanatory power without inflating fit just by adding complexity.
    """
    Calculate the Adjusted R-squared between true and predicted values.

    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.
    n_features (int): Number of features used in the model.

    Returns:
    float: The Adjusted R-squared value.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    n_samples = y_true.size
    r2 = r_squared(y_true, y_pred)
    return 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))

def f_statistic(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    # F-statistic: compares the regression model against a baseline model, testing whether the explanatory variables significantly improve the fit.
    """
    Calculate the F-statistic for the regression model.

    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.
    n_features (int): Number of features used in the model.

    Returns:
    float: The F-statistic value.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    n_samples = y_true.size
    rss = squared_loss(y_true, y_pred)
    tss = total_sum_of_squares(y_true)
    return ((tss - rss) / n_features) / (rss / (n_samples - n_features - 1))