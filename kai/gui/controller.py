"""Training orchestration and result persistence.

`TrainingController` owns the background thread so the UI stays responsive;
it never touches widgets directly - completion is delivered through a
`schedule_on_ui` callable (the app passes `lambda cb: app.after(0, cb)`),
which is the only Tkinter-safe way to cross back from a worker thread.

`build_results_report` / `save_results_report` are pure and unit-tested.
"""
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from kai.gui.helpers import format_elapsed
from kai.gui.state import TrainingRequest, TrainingResult
from kai.model import Model
from kai.visualization import metrics_rows


class TrainingController:
    def __init__(self, schedule_on_ui: Callable[[Callable[[], None]], None]):
        self._schedule_on_ui = schedule_on_ui
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(
        self,
        request: TrainingRequest,
        on_success: Callable[[TrainingResult], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        """Run the request on a daemon thread; exactly one of the callbacks
        fires later, on the UI thread."""
        if self._running:
            raise RuntimeError("A training run is already in progress")
        self._running = True
        threading.Thread(
            target=self._worker, args=(request, on_success, on_error), daemon=True
        ).start()

    def _worker(self, request: TrainingRequest, on_success, on_error) -> None:
        started = time.monotonic()
        hp = request.hyperparameters
        try:
            model = Model(request.csv_path, request.label_column)
            model.start_training(
                list(request.features),
                learning_rate=hp.learning_rate,
                batch_size=hp.batch_size,
                epochs=hp.epochs,
                tolerance=hp.tolerance,
                standardize_features=hp.standardize_features,
                random_state=hp.random_state,
            )
            result = TrainingResult(
                request=request,
                loss_history=tuple(model.loss_history),
                y_true=model.y_train,
                y_pred=model.predict(model.x_train),
                weights=np.atleast_1d(np.asarray(model.weight, dtype=float)),
                bias=float(model.bias),
                elapsed_seconds=time.monotonic() - started,
            )
        except Exception as exc:
            self._running = False
            self._schedule_on_ui(lambda: on_error(exc))
            return
        self._running = False
        self._schedule_on_ui(lambda: on_success(result))


def _format_equation(result: TrainingResult) -> str:
    terms = " + ".join(
        f"{float(w):.6f} * {feature}"
        for w, feature in zip(result.weights, result.request.features)
    )
    sign = "+" if result.bias >= 0 else "-"
    return f"{result.request.label_column} = {terms} {sign} {abs(result.bias):.6f}"


def build_results_report(result: TrainingResult) -> str:
    """Render a training run as a plain-text report (no dataset rows)."""
    request = result.request
    hp = request.hyperparameters
    lines = [
        "=" * 60,
        " kai - Training Report",
        f" Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 60,
        "",
        "Dataset",
        f"  file            : {Path(request.csv_path).name}",
        f"  label column    : {request.label_column}",
        f"  feature columns : {', '.join(request.features)}",
        "",
        "Hyperparameters",
        f"  learning rate   : {hp.learning_rate}",
        f"  batch size      : {hp.batch_size}",
        f"  max epochs      : {hp.epochs}",
        f"  stop tolerance  : {hp.tolerance} (relative: ||grad|| <= tol * ||grad_initial||)",
        f"  standardization : {'enabled (z-score)' if hp.standardize_features else 'disabled'}",
        f"  random state    : {hp.random_state if hp.random_state is not None else 'unseeded (not reproducible)'}",
        "",
        "Training",
        f"  epochs run      : {result.epochs_run}",
        f"  elapsed time    : {format_elapsed(result.elapsed_seconds)} (mm:ss:cc)",
        f"  final MSE       : {result.final_loss:.6f}",
        "",
        "Model",
        f"  {_format_equation(result)}",
    ]
    if hp.standardize_features:
        lines.append("  (weights are in standardized feature space; predict() re-applies the scaling)")
    lines += ["", "Metrics (in-sample)"]
    for name, value in metrics_rows(result.y_true, result.y_pred, len(request.features)):
        lines.append(f"  {name:<18}: {value:.6f}")
    lines += [
        "",
        "  Note: computed on the training data itself - no train/test split - so they",
        "  measure goodness of fit, not generalization to unseen data.",
        "",
    ]
    return "\n".join(lines)


def save_results_report(result: TrainingResult, path) -> Path:
    """Write the report to `path` (UTF-8) and return it."""
    path = Path(path)
    path.write_text(build_results_report(result), encoding="utf-8")
    return path
