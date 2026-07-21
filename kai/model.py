import csv
import numpy as np
from kai.metrics import mean_squared_error, mean_squared_error_derivation

class Model:
    def __init__(self, csv_file: str, label_column: str):
        self.csv_file = csv_file
        self.label_column = label_column
        self._weight = 0
        self._bias = 0
        self._loss_history = []

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def bias(self) -> float:
        return self._bias

    @property
    def loss_history(self) -> list:
        return list(self._loss_history)

    def get_data(self, label_column: str, feature_column: str) -> list:
        with open(self.csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return [(row[feature_column], row[label_column]) for row in reader]

    def start_training(self, feature: str, learning_rate: float, batch_size: int = 100, epochs: int = 10_000, tolerance: float = 1e-6) -> None:
        self._loss_history = []
        weight = 0
        bias = 0
        data = self.get_data(self.label_column, feature)
        x_full = np.array([float(row[0]) for row in data])
        y_full = np.array([float(row[1]) for row in data])
        n_samples = len(x_full)
        last_loss = float('inf')
        rng = np.random.default_rng()

        for _ in range(epochs):
            converged = False
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                x_true = x_full[batch_idx]
                y_true = y_full[batch_idx]
                y_pred = bias + weight * x_true
                current_loss = mean_squared_error(y_true, y_pred)
                self._loss_history.append(current_loss)
                if abs(current_loss - last_loss) < tolerance:
                    converged = True
                    break
                weight_slope, bias_slope = mean_squared_error_derivation(y_true, y_pred, x_true)
                weight -= (learning_rate * weight_slope)
                bias -= (learning_rate * bias_slope)
                last_loss = current_loss
            if converged:
                break

        self._weight, self._bias = weight, bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._bias + self._weight * np.asarray(x, dtype=float)
