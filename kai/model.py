import csv
import numpy as np
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
        self._feature = None

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

    def get_data(self, label_column: str, feature_column: str) -> list:
        with open(self.csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return [(row[feature_column], row[label_column]) for row in reader]

    def start_training(
        self,
        feature: str,
        learning_rate: float,
        batch_size: int = 100,
        epochs: int = 10_000,
        tolerance: float = 1e-6,
        show_plot: bool = True,
    ) -> None:
        self._loss_history = []
        weight = 0
        bias = 0
        data = self.get_data(self.label_column, feature)
        x_full = np.array([float(row[0]) for row in data])
        y_full = np.array([float(row[1]) for row in data])
        self._x_train, self._y_train, self._feature = x_full, y_full, feature
        n_samples = len(x_full)
        last_epoch_loss = float('inf')
        rng = np.random.default_rng()

        for epoch in range(epochs):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                x_batch = x_full[batch_idx]
                y_batch = y_full[batch_idx]
                y_pred_batch = bias + weight * x_batch
                weight_slope, bias_slope = mean_squared_error_derivation(y_batch, y_pred_batch, x_batch)
                weight -= (learning_rate * weight_slope)
                bias -= (learning_rate * bias_slope)

            epoch_loss = mean_squared_error(y_full, bias + weight * x_full)
            if not np.isfinite(epoch_loss):
                raise ValueError(
                    f"Training diverged at epoch {epoch}: loss became {epoch_loss}. "
                    f"Try a smaller learning_rate (current={learning_rate}) or normalize the feature values."
                )
            self._loss_history.append(epoch_loss)
            if abs(epoch_loss - last_epoch_loss) < tolerance:
                break
            last_epoch_loss = epoch_loss

        self._weight, self._bias = weight, bias

        if show_plot:
            plot_training_results(
                self._loss_history,
                self._x_train,
                self._y_train,
                self._weight,
                self._bias,
                feature_name=feature,
                label_name=self.label_column,
            )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._bias + self._weight * np.asarray(x, dtype=float)
