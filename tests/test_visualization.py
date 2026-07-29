import matplotlib.pyplot as plt
import numpy as np

from kai.visualization import build_training_figure


def test_build_training_figure_has_four_axes():
    loss_history = [10.0, 5.0, 1.0]
    x_true = np.array([1.0, 2.0, 3.0])
    y_true = np.array([2.0, 4.0, 6.0])

    fig = build_training_figure(loss_history, x_true, y_true, weight=2.0, bias=0.0)
    try:
        # loss curve, feature-vs-label plot, metrics table, and the textbox axes
        assert len(fig.axes) == 4
    finally:
        plt.close(fig)


def test_loss_curve_ignores_non_finite_entries():
    loss_history = [float("inf"), 10.0, float("nan"), 5.0, 1.0]
    x_true = np.array([1.0, 2.0, 3.0])
    y_true = np.array([2.0, 4.0, 6.0])

    fig = build_training_figure(loss_history, x_true, y_true, weight=2.0, bias=0.0)
    try:
        ax_loss = fig.axes[0]
        line = ax_loss.lines[0]
        assert np.isfinite(line.get_ydata()).all()
        assert list(line.get_ydata()) == [10.0, 5.0, 1.0]
    finally:
        plt.close(fig)


def test_metrics_table_shows_expected_values():
    loss_history = [1.0]
    x_true = np.array([1.0, 2.0, 3.0])
    y_true = np.array([2.0, 4.0, 6.0])

    # weight=2, bias=0 => perfect fit: L1/L2/MSE/RMSE are 0, R2/adjusted R2 are 1
    fig = build_training_figure(loss_history, x_true, y_true, weight=2.0, bias=0.0)
    try:
        ax_table = fig.axes[2]
        table = ax_table.tables[0]
        values = [table.get_celld()[(row, 1)].get_text().get_text() for row in range(1, 7)]
        assert values == ["0.0000", "0.0000", "0.0000", "0.0000", "1.0000", "1.0000"]
    finally:
        plt.close(fig)


def test_clicking_the_model_line_shows_prediction():
    loss_history = [10.0, 5.0, 1.0]
    x_true = np.array([1.0, 2.0, 3.0])
    y_true = np.array([2.0, 4.0, 6.0])

    fig = build_training_figure(loss_history, x_true, y_true, weight=2.0, bias=0.0)
    try:
        _, ax_fit, _, _ = fig.axes
        line = ax_fit.lines[0]  # model line, then the (initially empty) prediction marker
        annotation = ax_fit.texts[-1]
        assert annotation.get_visible() is False

        fig.canvas.draw()
        mouse_event = type("MouseEvent", (), {"xdata": 2.0, "ydata": 4.0})()
        pick_event = type("PickEvent", (), {"artist": line, "mouseevent": mouse_event})()
        fig.canvas.callbacks.process("pick_event", pick_event)

        assert annotation.get_visible() is True
        assert "4" in annotation.get_text()
    finally:
        plt.close(fig)


def test_submitting_textbox_value_shows_prediction():
    loss_history = [10.0, 5.0, 1.0]
    x_true = np.array([1.0, 2.0, 3.0])
    y_true = np.array([2.0, 4.0, 6.0])

    fig = build_training_figure(loss_history, x_true, y_true, weight=3.0, bias=1.0)
    try:
        _, ax_fit, _, _ = fig.axes
        annotation = ax_fit.texts[-1]

        fig.kai_textbox.set_val("5")

        assert annotation.get_visible() is True
        assert "16" in annotation.get_text()
    finally:
        plt.close(fig)
