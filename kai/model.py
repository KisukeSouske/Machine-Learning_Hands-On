import numpy as np
import pandas as pd
from kai.metrics import mean_squared_error, mean_squared_error_derivation
from kai.preprocessing import standardize
from kai.visualization import plot_training_results

class Model:
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
        show_plot: bool = True,
        standardize_features: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Fit weights and bias by mini-batch gradient descent.

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
        show_plot (bool): Open the standalone matplotlib dashboard when done.
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
        self._loss_history = []
        weight = np.zeros(len(features))
        bias = 0
        X, y = self.get_data(self.label_column, features)
        x_raw, y_full = X.to_numpy(dtype=float), y.to_numpy(dtype=float)
        # _x_train always keeps the RAW values; predict() re-applies the same
        # transform, so callers never have to know whether training was scaled
        self._x_train, self._y_train, self._features = x_raw, y_full, features

        if standardize_features:
            x_full, self._feature_mean, self._feature_std = standardize(x_raw)
        else:
            x_full = x_raw
            self._feature_mean = self._feature_std = None

        n_samples = len(X)
        rng = np.random.default_rng(random_state)

        # Reference magnitude for the convergence test, measured at the STARTING
        # parameters (before any update). Taking it after the first epoch would
        # be self-defeating: a run that converges immediately would compare its
        # already-tiny gradient against itself and never satisfy the threshold.
        initial_gradient_norm = self._gradient_norm(y_full, bias + x_full @ weight, x_full) or 1.0

        for epoch in range(epochs):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                x_batch = x_full[batch_idx]
                y_batch = y_full[batch_idx]
                y_pred_batch = bias + x_batch @ weight
                weight_slope, bias_slope = mean_squared_error_derivation(y_batch, y_pred_batch, x_batch)
                weight -= (learning_rate * weight_slope)
                bias -= (learning_rate * bias_slope)

            y_pred_full = bias + x_full @ weight
            epoch_loss = mean_squared_error(y_full, y_pred_full)
            if not np.isfinite(epoch_loss):
                raise ValueError(
                    f"Training diverged at epoch {epoch}: loss became {epoch_loss}. "
                    f"Try a smaller learning_rate (current={learning_rate}) or enable "
                    f"standardize_features."
                )
            self._loss_history.append(epoch_loss)

            # Converged when the full-dataset gradient has shrunk to a small
            # FRACTION of where it started. A flat loss delta is not enough on
            # its own: in an ill-conditioned problem the loss can stall while
            # parameters are still far from the optimum. Measuring the gradient
            # relative to its initial magnitude also makes the criterion
            # independent of the units of y and of the features.
            if self._gradient_norm(y_full, y_pred_full, x_full) <= tolerance * initial_gradient_norm:
                break

        self._weight, self._bias = weight, bias

        if show_plot:
            plot_training_results(
                self._loss_history,
                self._y_train,
                self.predict(self._x_train),
                n_features=len(features),
            )

    @staticmethod
    def _gradient_norm(y_true: np.ndarray, y_pred: np.ndarray, x: np.ndarray) -> float:
        """Euclidean norm of the full MSE gradient (weights and bias together)."""
        weight_slope, bias_slope = mean_squared_error_derivation(y_true, y_pred, x)
        return float(np.sqrt(np.sum(weight_slope ** 2) + bias_slope ** 2))

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._feature_mean is not None:
            x = (x - self._feature_mean) / self._feature_std
        return self._bias + x @ self._weight
