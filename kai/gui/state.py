"""Domain state for a training session, decoupled from any widget.

The app collects widget values into these frozen dataclasses before training,
and everything downstream (controller, charts, metrics table, saved report)
reads from them - no more untyped tuples or state scattered across tk.Vars.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np

Method = Literal["gd", "ols"]


@dataclass(frozen=True)
class Hyperparameters:
    """One training configuration.

    Only relevant to gradient descent - OLS has no hyperparameters. When
    `TrainingRequest.method == "ols"` the controller ignores everything here,
    but we keep the fields present (with their defaults) to avoid a second
    dataclass just to represent "no config".

    `tolerance` is relative: training stops once the gradient norm falls to
    this fraction of its initial magnitude. `random_state` seeds the mini-batch
    shuffle; None means each run reshuffles differently.
    """

    learning_rate: float = 0.0003
    batch_size: int = 100
    epochs: int = 10_000
    tolerance: float = 1e-4
    standardize_features: bool = False
    random_state: int | None = None
    # GLM family. ("mse", "identity") is ordinary least squares by descent;
    # ("gamma", "log") fits a Gamma GLM, whose predictions are exp(eta) and so
    # can never be non-positive. Only gradient descent supports a family -
    # the closed-form OLS path is normal-errors by construction.
    loss_function: str = "mse"
    loss_function_link: str = "identity"


@dataclass(frozen=True)
class TrainingRequest:
    """Everything needed to launch one training run.

    `separator` is the CSV delimiter to use when re-reading the file for
    training. The GUI captures it once (when the file is selected) so that
    training uses the same interpretation the user validated in the preview.
    """

    csv_path: str
    label_column: str
    features: tuple[str, ...]
    method: Method = "gd"
    hyperparameters: Hyperparameters = Hyperparameters()
    separator: str = ","


@dataclass(frozen=True)
class TrainingResult:
    """The outcome of a completed run, in original (unstandardized) space
    except for `weights`/`bias`, which live in whatever space training used.

    `loss_history` is None for OLS runs (there is no iteration to report).
    """

    request: TrainingRequest
    x_train: np.ndarray             # raw (unstandardized) features, shape (n_samples, n_features)
    y_true: np.ndarray
    y_pred: np.ndarray
    weights: np.ndarray
    bias: float
    elapsed_seconds: float
    loss_history: tuple[float, ...] | None = None

    @property
    def epochs_run(self) -> int | None:
        return None if self.loss_history is None else len(self.loss_history)

    @property
    def final_loss(self) -> float | None:
        return None if self.loss_history is None else self.loss_history[-1]


def training_scaling(result: TrainingResult):
    """The (mean, std) a run standardized with, or None if it trained on raw
    features.

    Single source of truth for what space `result.weights` live in: predict(),
    the Predictors tab and the saved report must agree, or they silently
    disagree by a factor of the feature's spread.
    """
    x_raw = np.asarray(result.x_train, dtype=float)
    standardized = (
        result.request.method == "gd"
        and result.request.hyperparameters.standardize_features
        and x_raw.size
    )
    if not standardized:
        return None
    spread = x_raw.std(axis=0)
    # matches the standardize() helper's guard for constant columns
    return x_raw.mean(axis=0), np.where(spread == 0, 1.0, spread)


def coefficients_in_original_space(result: TrainingResult) -> np.ndarray:
    """Feature coefficients expressed per one RAW unit of each predictor.

    A standardized run stores weights per standard deviation; dividing by the
    spread converts them back, so a reported coefficient always answers the
    same question regardless of how the model was trained.
    """
    weights = np.asarray(result.weights, dtype=float)
    scaling = training_scaling(result)
    if scaling is None:
        return weights
    _mean, spread = scaling
    return weights / spread


def intercept_in_original_space(result: TrainingResult) -> float:
    """The intercept of the same original-units equation.

    Undoing the centering moves part of each feature's effect into the
    constant term, so this is not simply `result.bias` for a standardized run.
    """
    weights = np.asarray(result.weights, dtype=float)
    bias = float(result.bias)
    scaling = training_scaling(result)
    if scaling is None:
        return bias
    mean, spread = scaling
    return bias - float(np.sum(weights * mean / spread))
