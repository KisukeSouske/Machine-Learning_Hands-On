"""A fitted linear model + factories that produce one via a chosen method.

`Model` is now a pure prediction object: it holds the fitted parameters and
the training-time feature scaling, and it knows how to reapply that scaling
when predicting from raw inputs. It does NOT run any training itself.

Two factories build a Model, one per estimation method:

- `Model.fit_gradient_descent(...)` - iterative, has hyperparameters (learning
  rate, batch size, epochs, tolerance, optional Z-scoring) and produces a
  loss history you can plot.
- `Model.fit_ols(...)`             - closed-form ordinary least squares, no
  hyperparameters, exact solution, no loss history.

Each factory returns a `TrainedModel`, a tiny dataclass pairing the fitted
`Model` with any method-specific artifact (the loss history for GD, nothing
for OLS). This is cleaner than an inheritance hierarchy: the two methods
share exactly one thing - the prediction function - and diverge on everything
else, so keeping "artifacts" separate from the predictor keeps the Model API
uniform without pretending they are two flavours of the same class.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kai.preprocessing import standardize
from kai.regression import _FAMILIES, fit_gradient_descent, fit_ols


@dataclass(frozen=True)
class TrainedModel:
    """A Model plus whatever extra artifacts its training method produced.

    `loss_history` is only populated by gradient descent (where iteration
    produces one). OLS has no iterations, so it stays `None`.
    """

    model: Model
    method: str
    loss_history: tuple[float, ...] | None = None


class Model:
    """A fitted linear regressor: parameters + how to reapply training scaling."""

    def __init__(
        self,
        csv_file: str,
        label_column: str,
        features: list[str],
        x_train: np.ndarray,
        y_train: np.ndarray,
        weights: np.ndarray,
        bias: float,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        loss_function: str = "mse",
        loss_function_link: str = "identity",
    ):
        """Callers rarely build a Model directly; use the classmethods below."""
        self.csv_file = csv_file
        self.label_column = label_column
        self._features = list(features)
        self._x_train = np.asarray(x_train, dtype=float)
        self._y_train = np.asarray(y_train, dtype=float)
        self._weight = np.asarray(weights, dtype=float)
        self._bias = float(bias)
        self._feature_mean = None if feature_mean is None else np.asarray(feature_mean, dtype=float)
        self._feature_std = None if feature_std is None else np.asarray(feature_std, dtype=float)
        self._loss_function = loss_function
        self._loss_function_link = loss_function_link

    # ------------------------------------------------------------------ #
    # Read-only accessors
    # ------------------------------------------------------------------ #
    @property
    def weight(self) -> np.ndarray:
        return self._weight

    @property
    def bias(self) -> float:
        return self._bias

    @property
    def x_train(self) -> np.ndarray:
        return self._x_train.copy()

    @property
    def y_train(self) -> np.ndarray:
        return self._y_train.copy()

    @property
    def features(self) -> list[str]:
        return list(self._features)

    @property
    def feature_mean(self) -> np.ndarray | None:
        return None if self._feature_mean is None else self._feature_mean.copy()

    @property
    def feature_std(self) -> np.ndarray | None:
        return None if self._feature_std is None else self._feature_std.copy()

    # ------------------------------------------------------------------ #
    # Prediction (shared by both training methods)
    # ------------------------------------------------------------------ #
    @property
    def loss_function(self) -> str:
        return self._loss_function

    @property
    def loss_function_link(self) -> str:
        return self._loss_function_link

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict from RAW feature values, reapplying any training scaling.

        The result is on the scale of y: the family's inverse link is applied,
        so a Gamma-log model returns mu, not the linear predictor.
        """
        eta = self.linear_predictor(x)
        return _FAMILIES[(self._loss_function, self._loss_function_link)].inverse_link(eta)

    def linear_predictor(self, x: np.ndarray) -> np.ndarray:
        """The linear part eta, from RAW feature values, before the inverse
        link. Equal to predict() only for the default identity link."""
        x = np.asarray(x, dtype=float)
        if self._feature_mean is not None:
            x = (x - self._feature_mean) / self._feature_std
        return self._bias + x @ self._weight

    # ------------------------------------------------------------------ #
    # Data loading (shared)
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_columns(
        csv_file: str,
        label_column: str,
        feature_columns: list[str],
        sep: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read the required columns from a CSV as plain numpy arrays.

        `sep=None` asks pandas to sniff the delimiter (python engine); pass
        the delimiter explicitly (e.g. ";") to skip the sniff.
        """
        engine = "python" if sep is None else None
        df = pd.read_csv(csv_file, sep=sep, engine=engine)
        return df[feature_columns].to_numpy(dtype=float), df[label_column].to_numpy(dtype=float)

    # ------------------------------------------------------------------ #
    # Factories, one per estimation method
    # ------------------------------------------------------------------ #
    @classmethod
    def fit_gradient_descent(
        cls,
        csv_file: str,
        label_column: str,
        features: list[str],
        learning_rate: float,
        batch_size: int = 100,
        epochs: int = 10_000,
        tolerance: float = 1e-4,
        standardize_features: bool = False,
        random_state: int | None = None,
        sep: str | None = None,
        cancel_event: threading.Event | None = None,
        loss_function: str = "mse",
        loss_function_link: str = "identity",
    ) -> TrainedModel:
        """Fit by mini-batch gradient descent.

        Parameters mirror `kai.regression.fit_gradient_descent`. The returned
        TrainedModel carries the per-epoch loss history for plotting. `sep`
        is forwarded to pandas (None = sniff); `cancel_event` is forwarded to
        allow stopping the run early (raises `TrainingCancelled`).

        `loss_function`/`loss_function_link` select the GLM family (e.g.
        "gamma"/"log"). The family is stored on the returned Model so that its
        predict() applies the matching inverse link - a fit made here and a
        prediction made later must not disagree about what scale they are on.
        """
        x_raw, y = cls.load_columns(csv_file, label_column, features, sep=sep)

        if standardize_features:
            x_fit, feature_mean, feature_std = standardize(x_raw)
        else:
            x_fit, feature_mean, feature_std = x_raw, None, None

        fit = fit_gradient_descent(
            x_fit, y,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            tolerance=tolerance,
            random_state=random_state,
            cancel_event=cancel_event,
            loss_function=loss_function,
            loss_function_link=loss_function_link,
        )
        model = cls(
            csv_file=csv_file, label_column=label_column, features=features,
            x_train=x_raw, y_train=y,
            weights=fit.weights, bias=fit.bias,
            feature_mean=feature_mean, feature_std=feature_std,
            loss_function=loss_function, loss_function_link=loss_function_link,
        )
        return TrainedModel(model=model, method="gd", loss_history=fit.loss_history)

    @classmethod
    def fit_ols(
        cls,
        csv_file: str,
        label_column: str,
        features: list[str],
        sep: str | None = None,
    ) -> TrainedModel:
        """Fit exactly, by ordinary least squares via the normal equations.

        No hyperparameters, no scaling: OLS is invariant to affine changes of
        the predictors, and standardization would only complicate the returned
        coefficients without changing the fit itself. The returned TrainedModel
        has no loss_history (there was no iteration). `sep` is forwarded to
        pandas (None = sniff).
        """
        x_raw, y = cls.load_columns(csv_file, label_column, features, sep=sep)
        fit = fit_ols(x_raw, y)
        model = cls(
            csv_file=csv_file, label_column=label_column, features=features,
            x_train=x_raw, y_train=y,
            weights=fit.weights, bias=fit.bias,
        )
        return TrainedModel(model=model, method="ols", loss_history=None)
