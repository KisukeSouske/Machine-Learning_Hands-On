"""Domain state for a training session, decoupled from any widget.

The app collects widget values into these frozen dataclasses before training,
and everything downstream (controller, charts, metrics table, saved report)
reads from them - no more untyped tuples or state scattered across tk.Vars.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Hyperparameters:
    """One training configuration.

    `tolerance` is relative: training stops once the gradient norm falls to
    this fraction of its initial magnitude. `random_state` seeds the mini-batch
    shuffle; None means each run reshuffles differently.
    """

    learning_rate: float
    batch_size: int
    epochs: int
    tolerance: float
    standardize_features: bool
    random_state: int | None = None


@dataclass(frozen=True)
class TrainingRequest:
    """Everything needed to launch one training run."""

    csv_path: str
    label_column: str
    features: tuple[str, ...]
    hyperparameters: Hyperparameters


@dataclass(frozen=True)
class TrainingResult:
    """The outcome of a completed run, in original (unstandardized) space
    except for `weights`/`bias`, which live in whatever space training used."""

    request: TrainingRequest
    loss_history: tuple[float, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    weights: np.ndarray
    bias: float
    elapsed_seconds: float

    @property
    def epochs_run(self) -> int:
        return len(self.loss_history)

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1]
