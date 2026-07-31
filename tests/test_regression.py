"""Tests for the standalone solvers in kai.regression.

Ground truth for the hand-written linear solver is numpy's own solver; ground
truth for OLS is the normal equations computed independently.
"""
import numpy as np
import pytest

from kai.regression import (
    LinearFit,
    add_intercept,
    fit_gradient_descent,
    fit_ols,
    gradient_norm,
    solve_linear_system,
    variance_inflation_factors,
)


# --- add_intercept ---
def test_add_intercept_prepends_ones_column():
    design = add_intercept(np.array([[2.0, 3.0], [4.0, 5.0]]))
    assert design.shape == (2, 3)
    assert list(design[:, 0]) == [1.0, 1.0]
    assert design[:, 1:].tolist() == [[2.0, 3.0], [4.0, 5.0]]


def test_add_intercept_promotes_1d_input_to_a_column():
    design = add_intercept(np.array([7.0, 8.0, 9.0]))
    assert design.shape == (3, 2)
    assert design[:, 1].tolist() == [7.0, 8.0, 9.0]


# --- solve_linear_system (hand-written Gaussian elimination) ---
def test_solver_matches_numpy_on_random_systems():
    rng = np.random.default_rng(0)
    for size in (2, 5, 12):
        # diagonally dominant => well conditioned, unique solution
        A = rng.normal(0, 1, (size, size)) + size * np.eye(size)
        b = rng.normal(0, 1, size)
        assert solve_linear_system(A, b) == pytest.approx(np.linalg.solve(A, b))


def test_solver_handles_zero_pivot_via_partial_pivoting():
    # A[0, 0] is 0, so naive elimination would divide by zero; pivoting swaps rows
    A = np.array([[0.0, 2.0], [1.0, 3.0]])
    b = np.array([4.0, 5.0])
    assert solve_linear_system(A, b) == pytest.approx([-1.0, 2.0])


def test_solver_leaves_inputs_untouched():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([3.0, 5.0])
    A_before, b_before = A.copy(), b.copy()
    solve_linear_system(A, b)
    assert A.tolist() == A_before.tolist()
    assert b.tolist() == b_before.tolist()


def test_solver_rejects_singular_matrix():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])  # second row is twice the first
    with pytest.raises(ValueError, match="singular"):
        solve_linear_system(A, np.array([1.0, 2.0]))


def test_solver_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="square"):
        solve_linear_system(np.ones((2, 3)), np.ones(2))


# --- fit_ols ---
def test_fit_ols_recovers_exact_coefficients_on_noiseless_data():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 10, (50, 3))
    y = X @ np.array([2.0, -3.0, 0.5]) + 7.0
    fit = fit_ols(X, y)
    assert fit.weights == pytest.approx([2.0, -3.0, 0.5])
    assert fit.bias == pytest.approx(7.0)


def test_fit_ols_matches_independently_computed_normal_equations():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 10, (40, 2))
    y = X @ np.array([1.5, -0.75]) + 3.0 + rng.normal(0, 0.5, 40)

    design = np.column_stack([np.ones(40), X])
    expected = np.linalg.solve(design.T @ design, design.T @ y)

    fit = fit_ols(X, y)
    assert fit.bias == pytest.approx(expected[0])
    assert fit.weights == pytest.approx(expected[1:])


def test_fit_ols_accepts_1d_features():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 3.0 * x + 1.0
    fit = fit_ols(x, y)
    assert fit.weights == pytest.approx([3.0])
    assert fit.bias == pytest.approx(1.0)


def test_fit_ols_has_no_loss_history():
    """Closed form means no iterations, so there is nothing to report."""
    fit = fit_ols(np.array([[1.0], [2.0], [3.0]]), np.array([2.0, 4.0, 6.0]))
    assert fit.loss_history is None


def test_fit_ols_rejects_perfectly_collinear_predictors():
    x = np.linspace(1, 10, 20)
    X = np.column_stack([x, 2 * x])  # exact linear dependence
    with pytest.raises(ValueError, match="singular"):
        fit_ols(X, x + 1)


def test_fit_ols_rejects_more_coefficients_than_samples():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 samples, 4 coefficients
    with pytest.raises(ValueError, match="at least as many samples"):
        fit_ols(X, np.array([1.0, 2.0]))


def test_fit_ols_rejects_mismatched_sample_counts():
    with pytest.raises(ValueError, match="same number of samples"):
        fit_ols(np.array([[1.0], [2.0]]), np.array([1.0, 2.0, 3.0]))


# --- fit_gradient_descent ---
def test_gradient_descent_approaches_the_ols_solution():
    """Both solvers minimise the same objective, so they must agree."""
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (80, 2))  # already standardized-ish => well conditioned
    y = X @ np.array([2.0, -1.0]) + 4.0 + rng.normal(0, 0.2, 80)

    exact = fit_ols(X, y)
    approx = fit_gradient_descent(X, y, learning_rate=0.1, tolerance=1e-8, random_state=0)

    assert approx.weights == pytest.approx(exact.weights, abs=1e-3)
    assert approx.bias == pytest.approx(exact.bias, abs=1e-3)


def test_gradient_descent_reports_a_decreasing_loss_history():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (60, 1))
    y = 3.0 * X[:, 0] + 1.0
    fit = fit_gradient_descent(X, y, learning_rate=0.05, random_state=0)
    assert fit.loss_history is not None
    assert fit.loss_history[-1] < fit.loss_history[0]


def test_gradient_descent_raises_on_divergence():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="diverged"):
        fit_gradient_descent(X, y, learning_rate=10.0, epochs=500)


def test_gradient_descent_is_reproducible_with_a_seed():
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, (50, 2))
    y = X @ np.array([1.0, 2.0]) + 0.5
    runs = [
        fit_gradient_descent(X, y, learning_rate=0.05, batch_size=8, random_state=42)
        for _ in range(2)
    ]
    assert runs[0].weights.tolist() == runs[1].weights.tolist()
    assert runs[0].bias == runs[1].bias


def test_gradient_descent_stops_at_the_epoch_ceiling():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    fit = fit_gradient_descent(X, y, learning_rate=1e-8, epochs=5)
    assert len(fit.loss_history) == 5


# --- LinearFit ---
def test_linear_fit_predicts_from_its_own_coefficients():
    fit = LinearFit(weights=np.array([2.0, 3.0]), bias=1.0)
    assert fit.predict(np.array([[1.0, 1.0], [2.0, 0.0]])) == pytest.approx([6.0, 5.0])


def test_fitted_model_predicts_the_training_targets():
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 5, (30, 2))
    y = X @ np.array([1.0, -2.0]) + 3.0
    fit = fit_ols(X, y)
    assert fit.predict(X) == pytest.approx(y)


def test_variance_inflation_factors_raise_for_perfect_collinearity():
    features = {
        "x1": np.array([1.0, 2.0, 3.0, 4.0]),
        "x2": np.array([2.0, 4.0, 6.0, 8.0]),
    }

    with pytest.raises(ValueError, match="perfectly collinear"):
        variance_inflation_factors(features)


# --- gradient_norm ---
def test_gradient_norm_is_zero_at_the_optimum():
    rng = np.random.default_rng(8)
    X = rng.uniform(0, 10, (40, 2))
    y = X @ np.array([1.0, 2.0]) + 5.0
    fit = fit_ols(X, y)
    assert gradient_norm(y, fit.predict(X), X) == pytest.approx(0.0, abs=1e-9)
