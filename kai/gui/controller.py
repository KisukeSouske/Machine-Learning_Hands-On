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
from kai.gui.state import (
    TrainingRequest,
    TrainingResult,
    coefficients_in_original_space,
    intercept_in_original_space,
)
from kai.model import Model, TrainedModel
from kai.visualization import family_label as _family_label
from kai.visualization import family_loss_label as _loss_label
from kai.visualization import metrics_rows


class TrainingController:
    def __init__(self, schedule_on_ui: Callable[[Callable[[], None]], None]):
        self._schedule_on_ui = schedule_on_ui
        self._running = False
        self._cancel_event: threading.Event | None = None

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
        self._cancel_event = threading.Event()
        threading.Thread(
            target=self._worker, args=(request, on_success, on_error), daemon=True
        ).start()

    def stop(self) -> None:
        """Request cancellation of the run in progress, if any. Takes effect
        at the start of the next epoch (gradient descent only - OLS has no
        iterations to interrupt)."""
        if self._cancel_event is not None:
            self._cancel_event.set()

    def _worker(self, request: TrainingRequest, on_success, on_error) -> None:
        started = time.monotonic()
        try:
            trained = _fit_by_method(request, self._cancel_event)
            model = trained.model
            result = TrainingResult(
                request=request,
                x_train=model.x_train,
                y_true=model.y_train,
                y_pred=model.predict(model.x_train),
                weights=np.atleast_1d(np.asarray(model.weight, dtype=float)),
                bias=float(model.bias),
                elapsed_seconds=time.monotonic() - started,
                loss_history=trained.loss_history,
            )
        except Exception as exc:
            self._running = False
            # Python deletes the `exc` name when the except block exits, so the
            # lambda must close over a plain variable instead - otherwise the
            # callback (which only runs later, on the UI thread) raises
            # NameError instead of ever reaching on_error.
            error = exc
            self._schedule_on_ui(lambda: on_error(error))
            return
        self._running = False
        self._schedule_on_ui(lambda: on_success(result))


def _fit_by_method(
    request: TrainingRequest, cancel_event: threading.Event | None = None
) -> TrainedModel:
    """Dispatch to the factory that matches the requested estimation method."""
    if request.method == "gd":
        hp = request.hyperparameters
        return Model.fit_gradient_descent(
            request.csv_path, request.label_column, list(request.features),
            learning_rate=hp.learning_rate,
            batch_size=hp.batch_size,
            epochs=hp.epochs,
            tolerance=hp.tolerance,
            standardize_features=hp.standardize_features,
            random_state=hp.random_state,
            sep=request.separator,
            cancel_event=cancel_event,
            loss_function=hp.loss_function,
            loss_function_link=hp.loss_function_link,
        )
    if request.method == "ols":
        return Model.fit_ols(
            request.csv_path, request.label_column, list(request.features),
            sep=request.separator,
        )
    raise ValueError(f"Unknown estimation method {request.method!r}.")


def _format_equation(result: TrainingResult) -> str:
    # original-units coefficients, so the printed equation can be evaluated on
    # raw feature values and matches the Predictors tab. Printing the stored
    # weights would describe a z-scored input space the reader does not have.
    coefficients = coefficients_in_original_space(result)
    intercept = intercept_in_original_space(result)
    terms = " + ".join(
        f"{float(w):.6f} * {feature}"
        for w, feature in zip(coefficients, result.request.features)
    )
    sign = "+" if intercept >= 0 else "-"
    linear = f"{terms} {sign} {abs(intercept):.6f}"
    label = result.request.label_column
    # under a log link the linear part is log(mu), not the response itself;
    # printing "y = ..." would state a model that was never fitted
    if _family_of(result) == ("gamma", "log"):
        return f"{label} = exp({linear})"
    return f"{label} = {linear}"


def _family_of(result: TrainingResult) -> tuple[str, str]:
    """The GLM family of a run. Closed-form OLS is always normal/identity;
    only gradient descent carries a configurable family."""
    if result.request.method != "gd":
        return ("mse", "identity")
    hp = result.request.hyperparameters
    return (hp.loss_function, hp.loss_function_link)


def build_results_report(result: TrainingResult) -> str:
    """Render a training run as a plain-text report (no dataset rows).

    The layout adapts to the estimation method: OLS has no hyperparameters
    and no epochs/final-loss to report, so those sections are omitted.
    """
    request = result.request
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
        "Method",
        f"  {'ordinary least squares (closed form)' if request.method == 'ols' else 'gradient descent'}",
    ]
    if request.method == "gd":
        hp = request.hyperparameters
        lines += [
            "",
            "Hyperparameters",
            f"  learning rate   : {hp.learning_rate}",
            f"  batch size      : {hp.batch_size}",
            f"  max epochs      : {hp.epochs}",
            f"  stop tolerance  : {hp.tolerance} (relative: ||grad|| <= tol * ||grad_initial||)",
            f"  standardization : {'enabled (z-score)' if hp.standardize_features else 'disabled'}",
            f"  random state    : {hp.random_state if hp.random_state is not None else 'unseeded (not reproducible)'}",
            f"  family          : {_family_label(_family_of(result))}",
            "",
            "Training",
            f"  epochs run      : {result.epochs_run}",
            f"  elapsed time    : {format_elapsed(result.elapsed_seconds)} (mm:ss:cc)",
            f"  final {_loss_label(_family_of(result)):<10}: {result.final_loss:.6f}",
        ]
        if hp.standardize_features:
            model_note = "  (weights are in standardized feature space; predict() re-applies the scaling)"
        else:
            model_note = None
    else:
        lines += [
            "",
            "Timing",
            f"  elapsed time    : {format_elapsed(result.elapsed_seconds)} (mm:ss:cc)",
        ]
        model_note = None

    lines += ["", "Model", f"  {_format_equation(result)}"]
    if model_note:
        lines.append(model_note)
    lines += ["", "Metrics (in-sample)"]
    family = _family_of(result)
    for name, value in metrics_rows(result.y_true, result.y_pred, len(request.features),
                                    loss_function=family[0], loss_function_link=family[1]):
        lines.append(f"  {name:<30}: {value:.6f}")
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
