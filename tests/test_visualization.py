import matplotlib.pyplot as plt
import numpy as np
import pytest

from kai.visualization import build_dashboard_figure


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
