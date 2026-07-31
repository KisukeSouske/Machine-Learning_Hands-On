"""Regression tests pinning the findings of the 2026-07-31 numerical audit.

Each test corresponds to a finding (F1-F7) so a future refactor cannot silently
undo a fix. Ground truth comes from closed-form OLS (normal equations) and from
the formulas in the reference material (ISLR eq. 3.17/3.23/6.4).
"""
import csv

import numpy as np
import pytest

from kai.metrics import (
    adjusted_r_squared,
    f_statistic,
    loss,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_error_derivation,
    r_squared,
    squared_loss,
    total_sum_of_squares,
)
from kai.model import Model


def _write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def _linear_csv(tmp_path, y_scale=1.0):
    """A well-behaved linear dataset; y_scale lets tests probe scale effects."""
    rng = np.random.default_rng(3)
    x = rng.uniform(10, 90, 80)
    y = (0.47 * x + rng.normal(0, 1.5, 80)) * y_scale
    return _write_csv(tmp_path / f"data_{y_scale}.csv", list(zip(x, y)), ("x", "y"))


def _ols(x_column: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Closed-form least squares via the normal equations: (slope, intercept)."""
    design = np.column_stack([x_column, np.ones(len(y))])
    slope, intercept = np.linalg.solve(design.T @ design, design.T @ y)
    return float(slope), float(intercept)


# --- F1: stopping criterion must not quit while parameters are still moving ---
def test_standardized_run_lands_on_the_closed_form_optimum(tmp_path):
    csv_path = _linear_csv(tmp_path)
    rows = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    slope_star, intercept_star = _ols(rows[:, 0], rows[:, 1])

    model = Model(csv_path, "y")
    model.start_training(["x"], learning_rate=0.5, show_plot=False,
                         standardize_features=True, random_state=0)

    slope = float(model.weight[0]) / model._feature_std[0]
    intercept = float(model.bias) - float(model.weight[0]) * model._feature_mean[0] / model._feature_std[0]
    assert slope == pytest.approx(slope_star, rel=1e-6)
    assert intercept == pytest.approx(intercept_star, rel=1e-6)


# --- F2: the criterion is relative, so it must be invariant to the scale of y ---
def test_stopping_epoch_is_invariant_to_target_scale(tmp_path):
    small = Model(_linear_csv(tmp_path, y_scale=1.0), "y")
    large = Model(_linear_csv(tmp_path, y_scale=1000.0), "y")
    for model in (small, large):
        model.start_training(["x"], learning_rate=0.05, show_plot=False,
                             standardize_features=True, random_state=0)
    assert len(small.loss_history) == len(large.loss_history)


def test_tighter_tolerance_runs_longer_and_lands_closer(tmp_path):
    csv_path = _linear_csv(tmp_path)
    rows = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    slope_star, _ = _ols(rows[:, 0], rows[:, 1])

    errors = {}
    for tolerance in (1e-2, 1e-5):
        model = Model(csv_path, "y")
        model.start_training(["x"], learning_rate=0.05, tolerance=tolerance,
                             show_plot=False, standardize_features=True, random_state=0)
        slope = float(model.weight[0]) / model._feature_std[0]
        errors[tolerance] = (len(model.loss_history), abs(slope - slope_star))

    assert errors[1e-5][0] > errors[1e-2][0]      # more epochs
    assert errors[1e-5][1] < errors[1e-2][1]      # closer to the optimum


# --- F4: degenerate inputs raise instead of returning inf/-inf ---
def test_r_squared_rejects_constant_target():
    with pytest.raises(ValueError, match="constant"):
        r_squared(np.array([3.0, 3.0, 3.0]), np.array([3.0, 3.1, 2.9]))


def test_adjusted_r_squared_rejects_degenerate_sample_size():
    with pytest.raises(ValueError, match="n_samples > n_features"):
        adjusted_r_squared(np.array([1.0, 2.0]), np.array([1.0, 2.1]), n_features=1)


def test_f_statistic_rejects_perfect_fit():
    with pytest.raises(ValueError, match="perfect fit"):
        f_statistic(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), n_features=1)


def test_f_statistic_rejects_degenerate_sample_size():
    with pytest.raises(ValueError, match="n_samples > n_features"):
        f_statistic(np.array([1.0, 2.0]), np.array([1.1, 2.1]), n_features=1)


# --- F6: seeded runs are reproducible, unseeded ones are free to differ ---
def test_random_state_makes_minibatch_training_reproducible(tmp_path):
    csv_path = _linear_csv(tmp_path)
    runs = []
    for _ in range(2):
        model = Model(csv_path, "y")
        model.start_training(["x"], learning_rate=0.05, batch_size=8, show_plot=False,
                             standardize_features=True, random_state=123)
        runs.append((len(model.loss_history), float(model.weight[0]), float(model.bias)))
    assert runs[0] == runs[1]


# --- F7: metrics accept any array-like, not just np.ndarray ---
@pytest.mark.parametrize(
    "metric", [loss, mean_absolute_error, squared_loss, mean_squared_error, r_squared]
)
def test_metrics_accept_python_lists(metric):
    from_lists = metric([1.0, 2.0, 3.0, 4.0], [1.1, 1.9, 3.2, 3.8])
    from_arrays = metric(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.1, 1.9, 3.2, 3.8]))
    assert from_lists == pytest.approx(from_arrays)


def test_total_sum_of_squares_accepts_lists():
    assert total_sum_of_squares([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_gradient_accepts_lists():
    weight_slope, bias_slope = mean_squared_error_derivation(
        [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [[1.0], [2.0], [3.0]]
    )
    assert weight_slope == pytest.approx([2.0])
    assert bias_slope == pytest.approx(1.0)


# --- ISLR formula conformance, kept as executable documentation ---
def test_metrics_match_islr_formulas():
    rng = np.random.default_rng(11)
    y_true = rng.normal(10, 4, 60)
    y_pred = y_true + rng.normal(0, 1.5, 60)
    n, p = 60, 3
    rss = squared_loss(y_true, y_pred)
    tss = total_sum_of_squares(y_true)

    assert r_squared(y_true, y_pred) == pytest.approx(1 - rss / tss)              # ISLR 3.17
    assert adjusted_r_squared(y_true, y_pred, p) == pytest.approx(
        1 - (rss / (n - p - 1)) / (tss / (n - 1))                                  # ISLR 6.4
    )
    assert f_statistic(y_true, y_pred, p) == pytest.approx(
        ((tss - rss) / p) / (rss / (n - p - 1))                                    # ISLR 3.23
    )


def test_gradient_matches_central_difference():
    """The analytic gradient must agree with numerical differentiation."""
    rng = np.random.default_rng(5)
    x = rng.normal(0, 3, (40, 3))
    y = rng.normal(0, 5, 40)
    weight = rng.normal(0, 1, 3)
    bias = 0.7

    analytic_w, analytic_b = mean_squared_error_derivation(y, x @ weight + bias, x)

    eps = 1e-6
    numeric_w = np.empty(3)
    for j in range(3):
        hi, lo = weight.copy(), weight.copy()
        hi[j] += eps
        lo[j] -= eps
        numeric_w[j] = (mean_squared_error(y, x @ hi + bias)
                        - mean_squared_error(y, x @ lo + bias)) / (2 * eps)
    numeric_b = (mean_squared_error(y, x @ weight + bias + eps)
                 - mean_squared_error(y, x @ weight + bias - eps)) / (2 * eps)

    assert analytic_w == pytest.approx(numeric_w, abs=1e-6)
    assert analytic_b == pytest.approx(numeric_b, abs=1e-6)
