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
