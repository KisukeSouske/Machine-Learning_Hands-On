import matplotlib.pyplot as plt
import numpy as np
import pytest

from kai.visualization import (
    CHART_CALIBRATION,
    CHART_LOSS,
    CHART_PREDICTED_VS_ACTUAL,
    CHART_RESIDUALS,
    build_charts_figure,
    build_dashboard_figure,
    metrics_rows,
)


def test_build_dashboard_figure_has_four_axes():
    loss_history = [10.0, 5.0, 1.0]
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([2.1, 3.9, 6.2])

    fig = plt.figure()
    try:
        build_dashboard_figure(fig, loss_history, y_true, y_pred)
        assert len(fig.axes) == 4
    finally:
        plt.close(fig)


def test_loss_curve_ignores_non_finite_entries():
    loss_history = [float("inf"), 10.0, float("nan"), 5.0, 1.0]
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([2.1, 3.9, 6.2])

    fig = plt.figure()
    try:
        build_dashboard_figure(fig, loss_history, y_true, y_pred)
        ax_loss = fig.axes[0]
        line = ax_loss.lines[0]
        assert np.isfinite(line.get_ydata()).all()
        assert list(line.get_ydata()) == [10.0, 5.0, 1.0]
    finally:
        plt.close(fig)


def test_residuals_plot_shows_actual_minus_predicted():
    loss_history = [1.0]
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([1.0, 4.0, 8.0])

    fig = plt.figure()
    try:
        build_dashboard_figure(fig, loss_history, y_true, y_pred)
        ax_residuals = fig.axes[1]
        offsets = ax_residuals.collections[0].get_offsets()
        # x = predicted, y = actual - predicted
        assert list(offsets[:, 0]) == pytest.approx([1.0, 4.0, 8.0])
        assert list(offsets[:, 1]) == pytest.approx([1.0, 0.0, -2.0])
    finally:
        plt.close(fig)


def test_predicted_vs_actual_plot_scatters_true_against_pred():
    loss_history = [1.0]
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([1.0, 4.0, 8.0])

    fig = plt.figure()
    try:
        build_dashboard_figure(fig, loss_history, y_true, y_pred)
        ax_pred_actual = fig.axes[2]
        offsets = ax_pred_actual.collections[0].get_offsets()
        assert list(offsets[:, 0]) == pytest.approx([2.0, 4.0, 6.0])
        assert list(offsets[:, 1]) == pytest.approx([1.0, 4.0, 8.0])
    finally:
        plt.close(fig)


def test_build_charts_figure_defaults_to_loss_curve_only():
    loss_history = [10.0, 5.0, 1.0]
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([2.1, 3.9, 6.2])

    fig = plt.figure()
    try:
        build_charts_figure(fig, loss_history, y_true, y_pred)
        assert len(fig.axes) == 1
        assert fig.axes[0].get_title() == "Loss curve"
    finally:
        plt.close(fig)


def test_build_charts_figure_draws_only_requested_charts():
    loss_history = [10.0, 5.0, 1.0]
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([2.1, 3.9, 6.2])

    fig = plt.figure()
    try:
        build_charts_figure(fig, loss_history, y_true, y_pred,
                            charts=[CHART_RESIDUALS, CHART_PREDICTED_VS_ACTUAL])
        assert len(fig.axes) == 2
        assert [ax.get_title() for ax in fig.axes] == ["Residuals vs Predicted", "Predicted vs Actual"]
    finally:
        plt.close(fig)


def test_build_charts_figure_draws_all_three_when_requested():
    loss_history = [10.0, 5.0, 1.0]
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([2.1, 3.9, 6.2])

    fig = plt.figure()
    try:
        build_charts_figure(fig, loss_history, y_true, y_pred,
                            charts=[CHART_LOSS, CHART_RESIDUALS, CHART_PREDICTED_VS_ACTUAL])
        assert len(fig.axes) == 3
    finally:
        plt.close(fig)


def test_build_charts_figure_with_no_charts_shows_placeholder():
    loss_history = [10.0, 5.0, 1.0]
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([2.1, 3.9, 6.2])

    fig = plt.figure()
    try:
        build_charts_figure(fig, loss_history, y_true, y_pred, charts=[])
        assert len(fig.axes) == 1
        assert "Select at least one chart" in fig.axes[0].texts[0].get_text()
    finally:
        plt.close(fig)


def test_metrics_rows_returns_all_six_metrics_for_perfect_fit():
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([2.0, 4.0, 6.0])

    rows = metrics_rows(y_true, y_pred, n_features=1)

    assert [name for name, _ in rows] == [
        "Loss (L1)", "Squared Loss (L2)", "MSE", "RMSE", "R²", "R² Adjusted",
    ]
    assert [value for _, value in rows] == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])


def test_metrics_table_shows_expected_values():
    loss_history = [1.0]
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([2.0, 4.0, 6.0])

    # perfect fit: L1/L2/MSE/RMSE are 0, R2/adjusted R2 are 1
    fig = plt.figure()
    try:
        build_dashboard_figure(fig, loss_history, y_true, y_pred)
        ax_table = fig.axes[3]
        table = ax_table.tables[0]
        values = [table.get_celld()[(row, 1)].get_text().get_text() for row in range(1, 7)]
        assert values == ["0.0000", "0.0000", "0.0000", "0.0000", "1.0000", "1.0000"]
    finally:
        plt.close(fig)


def _linear_fixture():
    """A 2-feature dataset with an exactly-known linear model."""
    rng = np.random.default_rng(0)
    x_train = np.column_stack([
        rng.uniform(10.0, 160.0, 60),
        rng.choice([623.0, 648.0, 673.0], 60),
    ])
    weights = np.array([1.5, 2.0])
    bias = 3.0
    y_true = bias + x_train @ weights + rng.normal(0, 5.0, 60)
    predict = lambda m: bias + np.asarray(m, dtype=float) @ weights
    return x_train, y_true, weights, bias, predict


def test_calibration_line_slope_equals_the_model_coefficient():
    x_train, y_true, weights, _bias, predict = _linear_fixture()
    names = ["concentracao_O", "temperatura"]

    for index in range(2):
        fig = plt.figure()
        try:
            build_charts_figure(
                fig, [], y_true, predict(x_train), charts=[CHART_CALIBRATION],
                x_train=x_train, feature_names=names, predict=predict,
                label_name="taxa_oxidacao", calibration_index=index,
            )
            ax = fig.axes[0]
            xs, ys = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
            slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
            assert slope == pytest.approx(weights[index])
        finally:
            plt.close(fig)


def test_calibration_line_passes_through_the_mean_of_the_other_predictors():
    x_train, y_true, _weights, _bias, predict = _linear_fixture()
    names = ["a", "b"]

    fig = plt.figure()
    try:
        build_charts_figure(
            fig, [], y_true, predict(x_train), charts=[CHART_CALIBRATION],
            x_train=x_train, feature_names=names, predict=predict,
            calibration_index=0,
        )
        xs, ys = fig.axes[0].lines[0].get_xdata(), fig.axes[0].lines[0].get_ydata()
        centre = x_train.mean(axis=0)
        assert np.interp(centre[0], xs, ys) == pytest.approx(predict(centre[None, :])[0])
    finally:
        plt.close(fig)


def test_calibration_chart_labels_the_selected_predictor():
    x_train, y_true, _weights, _bias, predict = _linear_fixture()

    fig = plt.figure()
    try:
        build_charts_figure(
            fig, [], y_true, predict(x_train), charts=[CHART_CALIBRATION],
            x_train=x_train, feature_names=["conc", "temp"], predict=predict,
            label_name="taxa", calibration_index=1,
        )
        ax = fig.axes[0]
        assert ax.get_xlabel() == "temp"
        assert ax.get_ylabel() == "taxa"
        assert ax.get_title() == "taxa vs temp"
    finally:
        plt.close(fig)


def test_calibration_chart_is_skipped_without_the_extra_data():
    # older callers pass neither x_train nor predict; the chart must drop out
    # instead of raising
    fig = plt.figure()
    try:
        build_charts_figure(fig, [1.0], np.array([1.0, 2.0]), np.array([1.1, 2.1]),
                            charts=[CHART_CALIBRATION])
        assert "Select at least one chart" in fig.axes[0].texts[0].get_text()
    finally:
        plt.close(fig)


def test_calibration_chart_is_skipped_when_the_index_is_out_of_range():
    x_train, y_true, _weights, _bias, predict = _linear_fixture()

    fig = plt.figure()
    try:
        build_charts_figure(
            fig, [], y_true, predict(x_train),
            charts=[CHART_RESIDUALS, CHART_CALIBRATION],
            x_train=x_train, feature_names=["a", "b"], predict=predict,
            calibration_index=9,
        )
        assert [ax.get_title() for ax in fig.axes] == ["Residuals vs Predicted"]
    finally:
        plt.close(fig)


def test_calibration_chart_composes_with_the_other_charts():
    x_train, y_true, _weights, _bias, predict = _linear_fixture()

    fig = plt.figure()
    try:
        build_charts_figure(
            fig, [3.0, 2.0], y_true, predict(x_train),
            charts=[CHART_LOSS, CHART_RESIDUALS, CHART_PREDICTED_VS_ACTUAL, CHART_CALIBRATION],
            x_train=x_train, feature_names=["a", "b"], predict=predict,
            label_name="y", calibration_index=0,
        )
        assert len(fig.axes) == 4
        assert fig.axes[-1].get_title() == "y vs a"
    finally:
        plt.close(fig)


def test_calibration_handles_a_constant_predictor_without_crashing():
    # a column with no spread has no range to sweep: draw the scatter, skip
    # the line, do not raise
    x_train = np.column_stack([np.full(10, 5.0), np.arange(10.0)])
    y_true = np.arange(10.0)
    predict = lambda m: np.asarray(m, dtype=float) @ np.array([0.0, 1.0])

    fig = plt.figure()
    try:
        build_charts_figure(
            fig, [], y_true, predict(x_train), charts=[CHART_CALIBRATION],
            x_train=x_train, feature_names=["flat", "ramp"], predict=predict,
            calibration_index=0,
        )
        assert len(fig.axes[0].lines) == 0
        assert len(fig.axes[0].collections) == 1
    finally:
        plt.close(fig)


def test_calibration_baseline_shifts_the_line_without_changing_its_slope():
    # pinning the other predictors elsewhere moves the line up or down; the
    # marginal slope of the swept predictor is unaffected
    x_train, y_true, weights, _bias, predict = _linear_fixture()

    lines = {}
    for baseline in (x_train.mean(axis=0), x_train.min(axis=0)):
        fig = plt.figure()
        try:
            build_charts_figure(
                fig, [], y_true, predict(x_train), charts=[CHART_CALIBRATION],
                x_train=x_train, feature_names=["a", "b"], predict=predict,
                calibration_index=0, calibration_baseline=baseline,
            )
            line = fig.axes[0].lines[0]
            lines[tuple(baseline)] = (line.get_xdata(), line.get_ydata())
        finally:
            plt.close(fig)

    (xs_a, ys_a), (xs_b, ys_b) = lines.values()
    slope_a = (ys_a[-1] - ys_a[0]) / (xs_a[-1] - xs_a[0])
    slope_b = (ys_b[-1] - ys_b[0]) / (xs_b[-1] - xs_b[0])
    assert slope_a == pytest.approx(weights[0])
    assert slope_b == pytest.approx(weights[0])
    assert ys_a[0] != pytest.approx(ys_b[0])   # but the intercept differs


def test_calibration_shades_the_region_where_the_fit_goes_negative():
    # a linear model on a strictly positive response can predict below zero
    # once the other predictors are pinned low enough - that has to be visible
    x_train = np.column_stack([np.linspace(5.0, 160.0, 40), np.full(40, 650.0)])
    y_true = np.full(40, 100.0)
    predict = lambda m: -6328.0 + np.asarray(m, dtype=float) @ np.array([2.28, 10.14])

    fig = plt.figure()
    try:
        build_charts_figure(
            fig, [], y_true, predict(x_train), charts=[CHART_CALIBRATION],
            x_train=x_train, feature_names=["conc", "temp"], predict=predict,
            calibration_index=0, calibration_baseline=np.array([80.0, 600.0]),
        )
        ax = fig.axes[0]
        ys = ax.lines[0].get_ydata()
        assert (ys < 0).any()
        labels = [c.get_label() for c in ax.collections]
        assert "predicted < 0 (impossible)" in labels
    finally:
        plt.close(fig)


def test_calibration_does_not_shade_when_the_fit_stays_positive():
    x_train, y_true, _weights, _bias, predict = _linear_fixture()

    fig = plt.figure()
    try:
        build_charts_figure(
            fig, [], y_true, predict(x_train), charts=[CHART_CALIBRATION],
            x_train=x_train, feature_names=["a", "b"], predict=predict,
            calibration_index=0,
        )
        labels = [c.get_label() for c in fig.axes[0].collections]
        assert "predicted < 0 (impossible)" not in labels
    finally:
        plt.close(fig)
