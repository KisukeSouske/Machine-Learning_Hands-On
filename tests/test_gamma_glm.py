"""Gamma/log GLM: gradient correctness, guards, and family propagation.

The engine and every consumer must agree on two things: which objective is
being descended, and which scale a "prediction" is on. Each test below pins
one of those down.
"""
import numpy as np
import pandas as pd
import pytest

from kai.metrics import (
    GAMMA_ETA_CLIP,
    gamma_log_inverse_link,
    gamma_log_nll,
    gamma_log_nll_derivation,
)
from kai.model import Model
from kai.regression import (
    fit_gradient_descent,
    fit_ols,
    predict_with_intervals,
    summarize_inference,
)


def _gamma_dataset(n=400, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    weights, bias = np.array([0.8, -0.5]), 1.2
    y = rng.gamma(shape=30.0, scale=np.exp(bias + x @ weights) / 30.0)
    return x, y, weights, bias


# --------------------------------------------------------------------- #
# The gradient must be the derivative of the loss it is paired with
# --------------------------------------------------------------------- #
def test_gamma_gradient_matches_finite_differences():
    x, y, _w, _b = _gamma_dataset(n=40)
    w0, b0 = np.array([0.2, 0.1]), 0.3

    def objective(w, b):
        mu = gamma_log_inverse_link(b + x @ w)
        return float(np.mean(y / mu + np.log(mu)))

    mu0 = gamma_log_inverse_link(b0 + x @ w0)
    weight_grad, bias_grad = gamma_log_nll_derivation(y, mu0, x)

    eps = 1e-6
    basis = np.eye(2)
    numeric_w = np.array([(objective(w0 + eps * basis[j], b0)
                           - objective(w0 - eps * basis[j], b0)) / (2 * eps)
                          for j in range(2)])
    numeric_b = (objective(w0, b0 + eps) - objective(w0, b0 - eps)) / (2 * eps)

    assert weight_grad == pytest.approx(numeric_w, rel=1e-5)
    assert bias_grad == pytest.approx(numeric_b, rel=1e-5)


def test_gamma_weight_and_bias_gradients_share_the_descent_sign_convention():
    # both must be true gradients, since the update subtracts them; a flipped
    # weight sign ascends in w while descending in b
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    mu = gamma_log_inverse_link(np.array([0.1, 0.2, 0.3]))

    weight_grad, bias_grad = gamma_log_nll_derivation(y, mu, x)
    error = (mu - y) / mu
    assert weight_grad == pytest.approx((x.T @ error) / y.size)
    assert bias_grad == pytest.approx(float(np.sum(error) / y.size))


def test_gamma_loss_is_a_mean_matching_its_gradient_scale():
    y = np.array([1.0, 2.0, 4.0])
    mu = np.array([1.5, 2.5, 3.0])
    assert gamma_log_nll(y, mu) == pytest.approx(float(np.mean(y / mu + np.log(mu))))


def test_gamma_loss_rejects_non_positive_predictions():
    with pytest.raises(ValueError, match="positive"):
        gamma_log_nll(np.array([1.0, 2.0]), np.array([1.0, 0.0]))


# --------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------- #
def test_gamma_gradient_descent_recovers_the_generating_coefficients():
    x, y, weights, bias = _gamma_dataset()

    fit = fit_gradient_descent(x, y, learning_rate=0.05, epochs=1500, batch_size=len(y),
                               tolerance=1e-12, random_state=0,
                               loss_function="gamma", loss_function_link="log")

    assert fit.weights == pytest.approx(weights, abs=0.12)
    assert fit.bias == pytest.approx(bias, abs=0.1)


def test_gamma_training_decreases_its_own_loss():
    x, y, _w, _b = _gamma_dataset()
    fit = fit_gradient_descent(x, y, learning_rate=0.05, epochs=300, batch_size=len(y),
                               random_state=0, loss_function="gamma",
                               loss_function_link="log")
    assert fit.loss_history[-1] < fit.loss_history[0]


def test_gamma_rejects_non_positive_targets():
    x, y, _w, _b = _gamma_dataset(n=20)
    y[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        fit_gradient_descent(x, y, learning_rate=0.01, loss_function="gamma",
                             loss_function_link="log")


def test_gamma_divergence_is_reported_despite_the_clipped_link():
    # the clip bounds the loss, so "not isfinite(loss)" can never fire here;
    # a runaway fit must still be refused rather than returned as a result
    x, y, _w, _b = _gamma_dataset(n=200)
    with pytest.raises(ValueError, match="diverged"):
        fit_gradient_descent(x, y, learning_rate=50.0, epochs=200, batch_size=200,
                             random_state=0, loss_function="gamma",
                             loss_function_link="log")


def test_unknown_family_is_rejected():
    x, y, _w, _b = _gamma_dataset(n=20)
    with pytest.raises(ValueError, match="Unsupported combination"):
        fit_gradient_descent(x, y, learning_rate=0.01, loss_function="gamma",
                             loss_function_link="identity")


# --------------------------------------------------------------------- #
# Propagation: the family must survive every hop to the consumer
# --------------------------------------------------------------------- #
def test_linear_fit_predict_applies_the_inverse_link():
    x, y, _w, _b = _gamma_dataset(n=100)
    fit = fit_gradient_descent(x, y, learning_rate=0.05, epochs=200, batch_size=100,
                               random_state=0, loss_function="gamma",
                               loss_function_link="log")

    assert (fit.loss_function, fit.loss_function_link) == ("gamma", "log")
    assert fit.predict(x) == pytest.approx(np.exp(fit.linear_predictor(x)))
    assert np.all(fit.predict(x) > 0)


def test_ols_fit_keeps_the_identity_link():
    x, y, _w, _b = _gamma_dataset(n=100)
    fit = fit_ols(x, y)

    assert (fit.loss_function, fit.loss_function_link) == ("mse", "identity")
    assert fit.predict(x) == pytest.approx(fit.linear_predictor(x))


def test_normal_theory_helpers_refuse_a_gamma_fit():
    # SE / t / F / intervals assume homoscedastic normal errors on the scale of
    # y; a Gamma-log fit breaks both assumptions, so returning numbers would be
    # worse than refusing
    x, y, _w, _b = _gamma_dataset(n=100)
    fit = fit_gradient_descent(x, y, learning_rate=0.05, epochs=100, batch_size=100,
                               random_state=0, loss_function="gamma",
                               loss_function_link="log")

    with pytest.raises(ValueError, match="normal errors"):
        summarize_inference(x, y, fit)
    with pytest.raises(ValueError, match="normal errors"):
        predict_with_intervals(x, y, fit, x[0])


def _write_csv(tmp_path, x, y):
    path = tmp_path / "gamma.csv"
    pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y}).to_csv(path, index=False)
    return str(path)


def test_model_forwards_the_family_and_predicts_on_the_target_scale(tmp_path):
    x, y, _w, _b = _gamma_dataset()
    path = _write_csv(tmp_path, x, y)

    trained = Model.fit_gradient_descent(
        path, "y", ["x1", "x2"], learning_rate=0.05, epochs=800, batch_size=len(y),
        tolerance=1e-12, random_state=0,
        loss_function="gamma", loss_function_link="log",
    )
    model = trained.model

    assert (model.loss_function, model.loss_function_link) == ("gamma", "log")
    probe = np.array([[0.5, -0.2]])
    assert model.predict(probe) == pytest.approx(np.exp(model.linear_predictor(probe)))
    # the structural promise of a log link: never a non-positive mean
    assert np.all(model.predict(x) > 0)


def test_standardized_gamma_model_still_predicts_on_the_target_scale(tmp_path):
    x, y, _w, _b = _gamma_dataset()
    path = _write_csv(tmp_path, x, y)

    trained = Model.fit_gradient_descent(
        path, "y", ["x1", "x2"], learning_rate=0.05, epochs=800, batch_size=len(y),
        tolerance=1e-12, random_state=0, standardize_features=True,
        loss_function="gamma", loss_function_link="log",
    )
    model = trained.model

    probe = np.array([[0.5, -0.2]])
    assert model.predict(probe) == pytest.approx(np.exp(model.linear_predictor(probe)))
    assert np.all(model.predict(x) > 0)


def test_default_model_path_is_unchanged_by_the_family_plumbing(tmp_path):
    x, y, _w, _b = _gamma_dataset()
    path = _write_csv(tmp_path, x, y)

    trained = Model.fit_gradient_descent(path, "y", ["x1", "x2"],
                                         learning_rate=0.05, epochs=200, random_state=0)
    model = trained.model

    assert (model.loss_function, model.loss_function_link) == ("mse", "identity")
    assert model.predict(x) == pytest.approx(model.linear_predictor(x))


def test_inverse_link_clips_instead_of_overflowing():
    huge = np.array([1e6, -1e6])
    mu = gamma_log_inverse_link(huge)
    assert np.all(np.isfinite(mu))
    assert mu[0] == pytest.approx(np.exp(GAMMA_ETA_CLIP))


# --------------------------------------------------------------------- #
# Family-appropriate reporting
# --------------------------------------------------------------------- #
from kai.gui.controller import build_results_report
from kai.gui.state import Hyperparameters, TrainingRequest, TrainingResult
from kai.metrics import gamma_deviance, gamma_explained_deviance
from kai.visualization import (
    FAMILY_GAMMA_LOG,
    FAMILY_GAUSSIAN,
    family_label,
    family_loss_label,
    metrics_rows,
)


def test_gamma_deviance_is_zero_for_a_perfect_fit():
    y = np.array([1.0, 2.0, 3.0])
    assert gamma_deviance(y, y) == pytest.approx(0.0)


def test_gamma_deviance_matches_its_definition():
    y = np.array([1.0, 2.0, 4.0])
    mu = np.array([1.5, 2.5, 3.0])
    expected = 2.0 * np.sum(-np.log(y / mu) + (y - mu) / mu)
    assert gamma_deviance(y, mu) == pytest.approx(expected)


def test_gamma_deviance_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        gamma_deviance(np.array([1.0, -1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        gamma_deviance(np.array([1.0, 1.0]), np.array([1.0, 0.0]))


def test_explained_deviance_is_one_for_a_perfect_fit_and_zero_for_the_mean():
    y = np.array([1.0, 2.0, 4.0, 8.0])
    assert gamma_explained_deviance(y, y) == pytest.approx(1.0)
    assert gamma_explained_deviance(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_explained_deviance_undefined_for_a_constant_target():
    y = np.array([3.0, 3.0, 3.0])
    with pytest.raises(ValueError, match="constant"):
        gamma_explained_deviance(y, y)


def test_metrics_rows_switch_to_deviance_measures_for_gamma():
    y = np.array([1.0, 2.0, 4.0, 8.0, 5.0])
    mu = np.array([1.2, 2.1, 3.6, 7.0, 5.5])

    gaussian = dict(metrics_rows(y, mu, n_features=1))
    gamma = dict(metrics_rows(y, mu, n_features=1,
                              loss_function="gamma", loss_function_link="log"))

    # R² and the squared-error family describe a least-squares fit only
    assert "R²" in gaussian and "MSE" in gaussian
    assert "R²" not in gamma and "MSE" not in gamma
    assert "Deviance" in gamma
    assert "Explained deviance (pseudo-R²)" in gamma
    assert gamma["Deviance"] == pytest.approx(gamma_deviance(y, mu))
    # distance-on-the-y-scale measures stay meaningful for both
    assert "RMSE" in gamma


def test_metrics_rows_omit_dispersion_without_residual_degrees_of_freedom():
    y = np.array([1.0, 2.0])
    mu = np.array([1.1, 2.1])
    rows = dict(metrics_rows(y, mu, n_features=1,
                             loss_function="gamma", loss_function_link="log"))
    assert "Deviance / df" not in rows


def test_family_labels_name_the_objective_that_was_descended():
    assert family_loss_label(FAMILY_GAUSSIAN) == "MSE"
    assert family_loss_label(FAMILY_GAMMA_LOG) == "Mean Gamma NLL"
    assert "Gamma" in family_label(FAMILY_GAMMA_LOG)


def _gd_result(loss_function, link, standardize, weights, bias, x_train, y_true, y_pred):
    request = TrainingRequest(
        csv_path="d.csv", label_column="y", features=("a", "b"), method="gd",
        hyperparameters=Hyperparameters(standardize_features=standardize,
                                        loss_function=loss_function,
                                        loss_function_link=link),
    )
    return TrainingResult(
        request=request, x_train=np.asarray(x_train, dtype=float),
        y_true=np.asarray(y_true, dtype=float), y_pred=np.asarray(y_pred, dtype=float),
        weights=np.asarray(weights, dtype=float), bias=float(bias),
        elapsed_seconds=0.1, loss_history=(3.0, 2.0),
    )


def test_report_writes_a_log_link_model_as_an_exponential():
    x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    y = np.array([2.0, 4.0, 6.0, 9.0])
    result = _gd_result("gamma", "log", False, [0.1, 0.02], 0.5, x, y, y * 1.01)

    report = build_results_report(result)

    assert "exp(" in report
    assert "Gamma / log" in report
    assert "Deviance" in report
    # the least-squares rows must be gone; "pseudo-R²" legitimately contains
    # the same characters, so match the row labels rather than a substring
    labels = {line.split(":")[0].strip() for line in report.splitlines() if ":" in line}
    assert "R²" not in labels
    assert "MSE" not in labels


def test_report_equation_uses_original_units_for_a_standardized_run():
    # the printed equation must be evaluable on raw feature values, and must
    # agree with what the Predictors tab shows
    x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    y = np.array([2.0, 4.0, 6.0, 9.0])
    spread = x.std(axis=0)
    stored = np.array([0.5, 0.25])          # per standard deviation
    result = _gd_result("mse", "identity", True, stored, 3.0, x, y, y)

    report = build_results_report(result)

    expected = stored / spread
    for coefficient in expected:
        assert f"{coefficient:.6f}" in report


def test_report_keeps_least_squares_metrics_for_the_default_family():
    x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    y = np.array([2.0, 4.0, 6.0, 9.0])
    result = _gd_result("mse", "identity", False, [1.0, 0.1], 0.5, x, y, y * 1.01)

    report = build_results_report(result)

    labels = {line.split(":")[0].strip() for line in report.splitlines() if ":" in line}
    assert "R²" in labels
    assert "MSE" in labels
    assert "Deviance" not in report
    assert "exp(" not in report


# --------------------------------------------------------------------- #
# Deviance residuals
# --------------------------------------------------------------------- #
from kai.metrics import gamma_deviance_residuals
from kai.visualization import CHART_RESIDUALS, build_charts_figure, build_dashboard_figure
import matplotlib.pyplot as plt


def test_deviance_residuals_square_to_the_deviance():
    # the defining property: they are to deviance what raw residuals are to RSS
    y = np.array([1.0, 2.0, 4.0, 8.0, 5.0])
    mu = np.array([1.2, 2.1, 3.6, 7.0, 5.5])

    residuals = gamma_deviance_residuals(y, mu)
    assert float(np.sum(residuals ** 2)) == pytest.approx(gamma_deviance(y, mu))


def test_deviance_residuals_take_the_sign_of_the_raw_error():
    y = np.array([1.0, 5.0, 3.0])
    mu = np.array([2.0, 4.0, 3.0])

    residuals = gamma_deviance_residuals(y, mu)
    assert list(np.sign(residuals)) == list(np.sign(y - mu))


def test_deviance_residual_is_exactly_zero_for_an_exact_point():
    # the unit deviance is non-negative with a minimum of 0 at y == mu; float
    # noise there must not produce a NaN out of the square root
    y = np.array([2.0, 7.5, 0.25])
    assert gamma_deviance_residuals(y, y) == pytest.approx(np.zeros(3))


def test_deviance_residuals_reject_non_positive_inputs():
    with pytest.raises(ValueError):
        gamma_deviance_residuals(np.array([1.0, 1.0]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        gamma_deviance_residuals(np.array([1.0, 0.0]), np.array([1.0, 1.0]))


def test_deviance_residuals_flatten_the_funnel_raw_residuals_show():
    # a Gamma model is heteroscedastic by assumption, so raw residuals fan out
    # with the fitted mean even for a correct fit; deviance residuals should
    # not, which is the whole reason to plot them
    rng = np.random.default_rng(11)
    mu = np.linspace(1.0, 100.0, 400)
    y = rng.gamma(shape=25.0, scale=mu / 25.0)

    raw = y - mu
    deviance = gamma_deviance_residuals(y, mu)
    low, high = mu < np.median(mu), mu >= np.median(mu)

    raw_ratio = raw[high].std() / raw[low].std()
    deviance_ratio = deviance[high].std() / deviance[low].std()
    assert raw_ratio > 2.0                     # the funnel
    assert deviance_ratio == pytest.approx(1.0, abs=0.3)   # flattened


def test_residuals_chart_plots_deviance_residuals_for_a_gamma_family():
    y = np.array([1.0, 2.0, 4.0, 8.0, 5.0])
    mu = np.array([1.2, 2.1, 3.6, 7.0, 5.5])

    fig = plt.figure()
    try:
        build_charts_figure(fig, [], y, mu, charts=[CHART_RESIDUALS],
                            family=FAMILY_GAMMA_LOG)
        ax = fig.axes[0]
        plotted = np.asarray(ax.collections[0].get_offsets())[:, 1]
        assert plotted == pytest.approx(gamma_deviance_residuals(y, mu))
        assert ax.get_ylabel() == "Deviance residual"
        assert "Deviance" in ax.get_title()
    finally:
        plt.close(fig)


def test_residuals_chart_keeps_raw_residuals_for_the_default_family():
    y = np.array([1.0, 2.0, 4.0, 8.0, 5.0])
    mu = np.array([1.2, 2.1, 3.6, 7.0, 5.5])

    fig = plt.figure()
    try:
        build_charts_figure(fig, [], y, mu, charts=[CHART_RESIDUALS])
        ax = fig.axes[0]
        plotted = np.asarray(ax.collections[0].get_offsets())[:, 1]
        assert plotted == pytest.approx(y - mu)
        assert ax.get_title() == "Residuals vs Predicted"
    finally:
        plt.close(fig)


def test_dashboard_also_switches_residual_type_with_the_family():
    # the popup dashboard is a second entry point; it must not keep showing
    # raw residuals for a Gamma fit
    y = np.array([1.0, 2.0, 4.0, 8.0, 5.0])
    mu = np.array([1.2, 2.1, 3.6, 7.0, 5.5])

    fig = plt.figure()
    try:
        build_dashboard_figure(fig, [3.0, 2.0], y, mu, n_features=1,
                               family=FAMILY_GAMMA_LOG)
        residual_axis = fig.axes[1]
        plotted = np.asarray(residual_axis.collections[0].get_offsets())[:, 1]
        assert plotted == pytest.approx(gamma_deviance_residuals(y, mu))
        assert fig.axes[0].get_ylabel() == "Mean Gamma NLL"
    finally:
        plt.close(fig)
