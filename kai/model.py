import numpy as np
import pandas as pd

from kai.preprocessing import standardize
from kai.regression import fit_gradient_descent


class Model:
    """A CSV-backed linear model.

    This is the stateful adapter around the pure solvers in `kai.regression`:
    it owns data loading, the feature scaling used during training, and the
    fitted parameters, so `predict` can re-apply the same transform. The
    fitting itself is delegated - call `kai.regression.fit_gradient_descent`
    or `fit_ols` directly when you just have arrays and want no state.
    """

    def __init__(self, csv_file: str, label_column: str):
        self.csv_file = csv_file
        self.label_column = label_column
        self._weight = 0
        self._bias = 0
        self._loss_history = []
        self._x_train = np.array([])
        self._y_train = np.array([])
        self._features = []
        self._feature_mean = None
        self._feature_std = None

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def bias(self) -> float:
        return self._bias

    @property
    def loss_history(self) -> list:
        return list(self._loss_history)

    @property
    def x_train(self) -> np.ndarray:
        return self._x_train.copy()

    @property
    def y_train(self) -> np.ndarray:
        return self._y_train.copy()

    def get_data(self, label_column: str, features_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(self.csv_file)
        X = df[features_columns]
        y = df[label_column]
        return X, y

    def start_training(
        self,
        features: list[str],
        learning_rate: float,
        batch_size: int = 100,
        epochs: int = 10_000,
        tolerance: float = 1e-4,
        standardize_features: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Load the CSV columns and fit weights and bias by gradient descent.

        Parameters:
        features (list[str]): Feature column names to use as predictors.
        learning_rate (float): Step size. Must satisfy lr < 2/lambda_max of the
            MSE Hessian or training diverges; standardizing features raises that
            ceiling dramatically (see `standardize_features`).
        batch_size (int): Samples per gradient step. Values >= n_samples make
            this plain full-batch gradient descent.
        epochs (int): Maximum passes over the data; a safety ceiling, not a target.
        tolerance (float): Relative convergence threshold. Training stops when
            ||grad|| <= tolerance * ||grad_initial||. Being relative to the
            starting gradient makes it invariant to the scale of y and of the
            features, unlike an absolute threshold on the loss delta.
        standardize_features (bool): Z-score the predictors before training.
            Recommended whenever features live on different scales: it drives
            the Hessian condition number toward 1, which is what lets a single
            learning rate work for every column (ISLR, p.179).
        random_state (int | None): Seed for the per-epoch batch shuffle. Pass an
            int for reproducible runs; None (default) reshuffles differently
            every call.

        Raises:
        ValueError: If the loss becomes non-finite, i.e. training diverged.
        """
        X, y = self.get_data(self.label_column, features)
        x_raw, y_full = X.to_numpy(dtype=float), y.to_numpy(dtype=float)
        # _x_train always keeps the RAW values; predict() re-applies the same
        # transform, so callers never have to know whether training was scaled
        self._x_train, self._y_train, self._features = x_raw, y_full, features

        if standardize_features:
            x_fit, self._feature_mean, self._feature_std = standardize(x_raw)
        else:
            x_fit = x_raw
            self._feature_mean = self._feature_std = None

        fit = fit_gradient_descent(
            x_fit,
            y_full,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            tolerance=tolerance,
            random_state=random_state,
        )
        self._weight = fit.weights
        self._bias = fit.bias
        self._loss_history = list(fit.loss_history)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict from RAW feature values, re-applying any training scaling."""
        x = np.asarray(x, dtype=float)
        if self._feature_mean is not None:
            x = (x - self._feature_mean) / self._feature_std
        return self._bias + x @ self._weight
