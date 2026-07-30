import numpy as np
import pandas as pd
from kai.metrics import mean_squared_error, mean_squared_error_derivation
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
        tolerance: float = 1e-6,
        show_plot: bool = True,
    ) -> None:
        self._loss_history = []
        weight = np.zeros(len(features))
        bias = 0
        X, y = self.get_data(self.label_column, features)
        x_full, y_full = X.to_numpy(), y.to_numpy()
        self._x_train, self._y_train, self._features = x_full, y_full, features
        n_samples = len(X)
        rng = np.random.default_rng()

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
                    f"Try a smaller learning_rate (current={learning_rate}) or normalize the feature values."
                )
            self._loss_history.append(epoch_loss)

            # early stopping on the full-dataset gradient norm: a gradient near
            # zero means there is no direction left that would reduce the loss
            full_weight_slope, full_bias_slope = mean_squared_error_derivation(y_full, y_pred_full, x_full)
            gradient_norm = np.sqrt(np.sum(full_weight_slope ** 2) + full_bias_slope ** 2)
            if gradient_norm < tolerance:
                break

        self._weight, self._bias = weight, bias

        if show_plot:
            plot_training_results(
                self._loss_history,
                self._y_train,
                self.predict(self._x_train),
                n_features=len(features),
            )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._bias + np.asarray(x, dtype=float) @ self._weight
