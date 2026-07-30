import numpy as np

def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to zero mean and unit standard deviation (Z-score).

    Parameters:
    X (np.ndarray): Feature values, shape (n_samples,) or (n_samples, n_features).

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: (X_standardized, mean, std), where
        mean and std are computed per feature (axis=0) and can be reused to apply
        the same transformation to new data: (new_X - mean) / std.

    Constant features have zero standard deviation; their std is reported as 1 so
    the division is safe and the column simply becomes all zeros.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std
