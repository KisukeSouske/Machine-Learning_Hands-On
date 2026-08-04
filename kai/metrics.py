import numpy as np

GAMMA_ETA_CLIP = 30.0

def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    # Accepts any array-like (list, tuple, Series, ndarray) and validates shapes,
    # so every public metric shares the same input contract.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    return y_true, y_pred

def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Total absolute error: sums the magnitudes of the residuals, showing the overall distance between predictions and observations.
    """
    Calculate loss between true and predicted values.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    float: The sum of absolute differences between true and predicted values.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
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
    y_true, y_pred = _as_arrays(y_true, y_pred)
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
    y_true, y_pred = _as_arrays(y_true, y_pred)
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
    y_true, y_pred = _as_arrays(y_true, y_pred)
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
    y_true, y_pred = _as_arrays(y_true, y_pred)
    x_true = np.asarray(x_true, dtype=float)
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
    y_true = np.asarray(y_true, dtype=float)
    return np.sum((y_true - np.mean(y_true)) ** 2)

def gamma_log_inverse_link(eta: np.ndarray) -> np.ndarray:
    """μ = exp(η), com clipping para evitar overflow durante a busca."""
    return np.exp(np.clip(eta, -GAMMA_ETA_CLIP, GAMMA_ETA_CLIP))

def gamma_log_nll(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
        Mean Gamma negative log-likelihood up to additive constants in beta.
        Here, y_pred is mu (already passed through the inverse link), not eta.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    if np.any(y_pred <= 0):
        raise ValueError("Predicted values must be positive for the Gamma distribution.")
    return np.sum(y_true / y_pred + np.log(y_pred))

def gamma_log_nll_derivation(y_true: np.ndarray, y_pred: np.ndarray, x_true: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Calculate the partial derivatives of the Gamma negative log-likelihood
    with respect to the weights and the bias.

    Parameters:
    y_true (np.ndarray): True values, shape (n_samples,).
    y_pred (np.ndarray): Predicted values (mu), shape (n_samples,).
    x_true (np.ndarray): Feature values, shape (n_samples, n_features).

    Returns:
    tuple[np.ndarray, float]: The (weight_derivation, bias_derivation) gradients.
        weight_derivation has shape (n_features,).
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    x_true = np.asarray(x_true, dtype=float)
    if x_true.shape[0] != y_true.shape[0]:
        raise ValueError("x_true must have the same number of samples as y_true.")

    error = (y_pred - y_true) / y_pred
    weight_derivation = -(x_true.T @ error) / y_true.size
    bias_derivation = float(np.sum(error) / y_true.size)
    return weight_derivation, bias_derivation

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # R-squared: the proportion of variance explained by the model; values near 1 indicate a good fit, while values below 0 mean the model is worse than simply predicting the mean.
    """
    Calculate the R-squared (coefficient of determination) between true and predicted values.

    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.

    Returns:
    float: The R-squared value.

    Raises:
    ValueError: If y_true is constant (TSS = 0), since R² is undefined
        when the target has no variance to explain.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    rss = squared_loss(y_true, y_pred)
    tss = total_sum_of_squares(y_true)
    if tss == 0:
        raise ValueError("R² is undefined when y_true is constant (TSS = 0).")
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

    Raises:
    ValueError: If n_samples <= n_features + 1, where the adjustment
        denominator (n - p - 1) is zero or negative and the statistic
        is undefined.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    n_samples = y_true.size
    if n_samples - n_features - 1 <= 0:
        raise ValueError(
            f"Adjusted R² requires n_samples > n_features + 1 "
            f"(got n_samples={n_samples}, n_features={n_features})."
        )
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

    Raises:
    ValueError: If the fit is perfect (RSS = 0) or if
        n_samples <= n_features + 1; the statistic is undefined in both cases.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    n_samples = y_true.size
    if n_samples - n_features - 1 <= 0:
        raise ValueError(
            f"F-statistic requires n_samples > n_features + 1 "
            f"(got n_samples={n_samples}, n_features={n_features})."
        )
    rss = squared_loss(y_true, y_pred)
    tss = total_sum_of_squares(y_true)
    if rss == 0:
        raise ValueError("F-statistic is undefined for a perfect fit (RSS = 0).")
    return ((tss - rss) / n_features) / (rss / (n_samples - n_features - 1))