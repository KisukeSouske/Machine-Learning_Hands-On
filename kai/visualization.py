import numpy as np
import matplotlib.pyplot as plt

from kai.metrics import loss, squared_loss, mean_squared_error, r_squared, adjusted_r_squared

PANEL_BACKGROUND = "#E5ECF6"
FIGURE_BACKGROUND = "#FFFFFF"
LOSS_CURVE_COLOR = "#0057E7"
ACCENT_COLOR = "#D62D20"
GRID_COLOR = "#FFFFFF"
FONT_FAMILY = "Roboto"


def _style_panel(ax) -> None:
    ax.set_facecolor(PANEL_BACKGROUND)
    ax.grid(True, color=GRID_COLOR, linewidth=1.2)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_loss_curve(ax, loss_history: list) -> None:
    loss_values = np.asarray(loss_history, dtype=float)
    finite_mask = np.isfinite(loss_values)
    epochs = np.arange(1, len(loss_values) + 1)
    _style_panel(ax)
    ax.plot(epochs[finite_mask], loss_values[finite_mask], color=LOSS_CURVE_COLOR, linewidth=2.5)
    ax.set_title("Loss curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")


def draw_residuals_plot(ax, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    # residuals vs predicted: the classic diagnostic for non-linearity and
    # heteroscedasticity (residuals should scatter randomly around 0, with no
    # funnel shape and no curved pattern)
    residuals = y_true - y_pred
    _style_panel(ax)
    ax.scatter(y_pred, residuals, color="tab:blue", alpha=0.8, zorder=3)
    ax.axhline(0, color=ACCENT_COLOR, linewidth=2, zorder=4)
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")


def draw_predicted_vs_actual_plot(ax, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    # predicted vs actual: how close the model is to a perfect y = x diagonal.
    # unlike a "feature vs label" scatter, this works for any number of
    # features, since it never needs to plot an individual predictor
    _style_panel(ax)
    ax.scatter(y_true, y_pred, color="tab:blue", alpha=0.8, zorder=3)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color=ACCENT_COLOR, linewidth=2, zorder=4, label="y = x")
    ax.set_title("Predicted vs Actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()


def metrics_rows(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1) -> list[tuple[str, float]]:
    """Single source of truth for the reported metrics, shared by the matplotlib
    table and by any GUI widget that renders the same numbers."""
    return [
        ("Loss (L1)", loss(y_true, y_pred)),
        ("Squared Loss (L2)", squared_loss(y_true, y_pred)),
        ("MSE", mean_squared_error(y_true, y_pred)),
        ("RMSE", np.sqrt(mean_squared_error(y_true, y_pred))),
        ("R²", r_squared(y_true, y_pred)),
        ("R² Adjusted", adjusted_r_squared(y_true, y_pred, n_features)),
    ]


def draw_metrics_table(ax, y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1) -> None:
    rows = metrics_rows(y_true, y_pred, n_features)

    ax.axis("off")
    table = ax.table(
        cellText=[[name, f"{value:.4f}"] for name, value in rows],
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(2)
        if row == 0:
            cell.set_facecolor(PANEL_BACKGROUND)
            cell.set_text_props(weight="bold", fontfamily=FONT_FAMILY)
        else:
            cell.set_facecolor(FIGURE_BACKGROUND)
            cell.set_text_props(fontfamily=FONT_FAMILY)


CHART_LOSS = "loss"
CHART_RESIDUALS = "residuals"
CHART_PREDICTED_VS_ACTUAL = "predicted_vs_actual"

CHART_LABELS = {
    CHART_LOSS: "Loss Curve",
    CHART_RESIDUALS: "Residuals Plot",
    CHART_PREDICTED_VS_ACTUAL: "Predicted vs Actual",
}


def build_charts_figure(fig, loss_history: list, y_true: np.ndarray, y_pred: np.ndarray, charts=None):
    """Populate `fig` with the requested charts, side by side. `charts` is a
    sequence of the CHART_* keys; drawing fewer of them keeps each one wide
    enough to read. Used by the GUI, where metrics live in a native table."""
    if charts is None:
        charts = [CHART_LOSS]
    charts = [c for c in charts if c in CHART_LABELS]

    with plt.rc_context({"font.family": FONT_FAMILY}):
        fig.patch.set_facecolor(FIGURE_BACKGROUND)
        if not charts:
            ax = fig.subplots(1, 1)
            ax.axis("off")
            ax.text(0.5, 0.5, "Select at least one chart", ha="center", va="center", color="#888888")
            return fig

        axes = fig.subplots(1, len(charts), squeeze=False)[0]
        for ax, chart in zip(axes, charts):
            if chart == CHART_LOSS:
                draw_loss_curve(ax, loss_history)
            elif chart == CHART_RESIDUALS:
                draw_residuals_plot(ax, y_true, y_pred)
            else:
                draw_predicted_vs_actual_plot(ax, y_true, y_pred)

        fig.tight_layout()
        return fig


def build_dashboard_figure(fig, loss_history: list, y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1):
    """Populate `fig` with the 2x2 training dashboard. `fig` can be a pyplot
    figure (standalone popup) or a bare matplotlib.figure.Figure (embedded in
    a GUI canvas) - this function doesn't care which, it never calls show()."""
    with plt.rc_context({"font.family": FONT_FAMILY}):
        fig.patch.set_facecolor(FIGURE_BACKGROUND)
        (ax_loss, ax_residuals), (ax_pred_actual, ax_table) = fig.subplots(2, 2)

        draw_loss_curve(ax_loss, loss_history)
        draw_residuals_plot(ax_residuals, y_true, y_pred)
        draw_predicted_vs_actual_plot(ax_pred_actual, y_true, y_pred)
        draw_metrics_table(ax_table, y_true, y_pred, n_features)

        fig.tight_layout()
        return fig


def plot_training_results(loss_history: list, y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1) -> None:
    fig = plt.figure(figsize=(11, 8))
    build_dashboard_figure(fig, loss_history, y_true, y_pred, n_features)
    plt.show()
