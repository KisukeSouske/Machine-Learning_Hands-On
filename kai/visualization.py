import numpy as np
import matplotlib.pyplot as plt

from kai.metrics import loss, squared_loss, mean_squared_error, r_squared, adjusted_r_squared
from kai.themes.base import ChartStyle
from kai.themes.default import DEFAULT_THEME

DEFAULT_CHART_STYLE: ChartStyle = DEFAULT_THEME.charts

# kept as module-level aliases for backward compatibility (notebook / old callers)
PANEL_BACKGROUND = DEFAULT_CHART_STYLE.panel_bg
FIGURE_BACKGROUND = DEFAULT_CHART_STYLE.figure_bg
LOSS_CURVE_COLOR = DEFAULT_CHART_STYLE.loss_color
ACCENT_COLOR = DEFAULT_CHART_STYLE.accent_color
GRID_COLOR = DEFAULT_CHART_STYLE.grid_color
FONT_FAMILY = DEFAULT_CHART_STYLE.font_family

CHART_LOSS = "loss"
CHART_RESIDUALS = "residuals"
CHART_PREDICTED_VS_ACTUAL = "predicted_vs_actual"
CHART_CALIBRATION = "calibration"

CHART_LABELS = {
    CHART_LOSS: "Loss Curve",
    CHART_RESIDUALS: "Residuals Plot",
    CHART_PREDICTED_VS_ACTUAL: "Predicted vs Actual",
    CHART_CALIBRATION: "Calibration (pairwise)",
}

# how many points to sample along a predictor when tracing its calibration line
CALIBRATION_LINE_POINTS = 100


def _style_panel(ax, style: ChartStyle) -> None:
    ax.set_facecolor(style.panel_bg)
    ax.grid(True, color=style.grid_color, linewidth=1.2)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_loss_curve(ax, loss_history: list, style: ChartStyle = DEFAULT_CHART_STYLE) -> None:
    loss_values = np.asarray(loss_history, dtype=float)
    finite_mask = np.isfinite(loss_values)
    epochs = np.arange(1, len(loss_values) + 1)
    _style_panel(ax, style)
    ax.plot(epochs[finite_mask], loss_values[finite_mask], color=style.loss_color, linewidth=2.5)
    ax.set_title("Loss curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")


def draw_residuals_plot(ax, y_true: np.ndarray, y_pred: np.ndarray, style: ChartStyle = DEFAULT_CHART_STYLE) -> None:
    # residuals vs predicted: the classic diagnostic for non-linearity and
    # heteroscedasticity (residuals should scatter randomly around 0, with no
    # funnel shape and no curved pattern)
    residuals = y_true - y_pred
    _style_panel(ax, style)
    ax.scatter(y_pred, residuals, color=style.scatter_color, alpha=0.8, zorder=3)
    ax.axhline(0, color=style.accent_color, linewidth=2, zorder=4)
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")


def draw_predicted_vs_actual_plot(ax, y_true: np.ndarray, y_pred: np.ndarray, style: ChartStyle = DEFAULT_CHART_STYLE) -> None:
    # predicted vs actual: how close the model is to a perfect y = x diagonal.
    # unlike a "feature vs label" scatter, this works for any number of
    # features, since it never needs to plot an individual predictor
    _style_panel(ax, style)
    ax.scatter(y_true, y_pred, color=style.scatter_color, alpha=0.8, zorder=3)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color=style.accent_color, linewidth=2, zorder=4, label="y = x")
    ax.set_title("Predicted vs Actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()


def draw_calibration_pair(ax, x_train: np.ndarray, y_true: np.ndarray, feature_index: int,
                          feature_name: str, predict, label_name: str = "y",
                          style: ChartStyle = DEFAULT_CHART_STYLE,
                          baseline: np.ndarray | None = None,
                          baseline_note: str = "others at mean") -> None:
    """Scatter the label against ONE predictor, with the model's calibration
    line drawn on top.

    The line sweeps this predictor across its observed range while every other
    predictor is pinned at `baseline` (the training mean by default), so the
    slope you see is the model's marginal response to this variable alone.
    Without that convention a multi-feature model has no single line to draw
    here: the fitted surface lives in as many dimensions as there are
    predictors.

    Where the pinned values sit matters as much as the slope: a linear model
    can predict an impossible negative response once the other predictors are
    moved far enough, and that only shows up for the baseline that produces
    it. Any stretch of the line below zero is shaded, so an impossible
    prediction is visible rather than something you have to go looking for.
    """
    x_train = np.asarray(x_train, dtype=float)
    column = x_train[:, feature_index]
    _style_panel(ax, style)
    ax.scatter(column, y_true, color=style.scatter_color, alpha=0.8, zorder=3, label="observed")

    low, high = float(np.min(column)), float(np.max(column))
    if not np.isclose(low, high):
        if baseline is None:
            baseline = x_train.mean(axis=0)
        sweep = np.linspace(low, high, CALIBRATION_LINE_POINTS)
        grid = np.tile(np.asarray(baseline, dtype=float), (CALIBRATION_LINE_POINTS, 1))
        grid[:, feature_index] = sweep
        fitted = np.asarray(predict(grid), dtype=float).ravel()
        ax.plot(sweep, fitted, color=style.accent_color, linewidth=2, zorder=4,
                label=f"fitted ({baseline_note})")
        if np.any(fitted < 0):
            # a negative fit is structurally impossible for a positive
            # response; flag it instead of leaving it to be read off the axis
            ax.axhline(0, color="black", linewidth=1, linestyle="--", zorder=5)
            ax.fill_between(sweep, fitted, 0, where=(fitted < 0),
                            color="red", alpha=0.2, zorder=2,
                            label="predicted < 0 (impossible)")
    ax.set_xlabel(feature_name)
    ax.set_ylabel(label_name)
    ax.set_title(f"{label_name} vs {feature_name}")
    ax.legend(fontsize="small")


def metrics_rows(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1) -> list[tuple[str, float]]:
    """Single source of truth for the reported metrics, shared by the matplotlib
    table, the GUI metrics widget and the saved training report."""
    return [
        ("Loss (L1)", loss(y_true, y_pred)),
        ("Squared Loss (L2)", squared_loss(y_true, y_pred)),
        ("MSE", mean_squared_error(y_true, y_pred)),
        ("RMSE", np.sqrt(mean_squared_error(y_true, y_pred))),
        ("R²", r_squared(y_true, y_pred)),
        ("R² Adjusted", adjusted_r_squared(y_true, y_pred, n_features)),
    ]


def draw_metrics_table(ax, y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1,
                       style: ChartStyle = DEFAULT_CHART_STYLE) -> None:
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
        cell.set_edgecolor(style.grid_color)
        cell.set_linewidth(2)
        if row == 0:
            cell.set_facecolor(style.panel_bg)
            cell.set_text_props(weight="bold", fontfamily=style.font_family)
        else:
            cell.set_facecolor(style.figure_bg)
            cell.set_text_props(fontfamily=style.font_family)


def build_charts_figure(fig, loss_history: list, y_true: np.ndarray, y_pred: np.ndarray,
                        charts=None, style: ChartStyle = DEFAULT_CHART_STYLE,
                        x_train: np.ndarray | None = None, feature_names=None,
                        predict=None, label_name: str = "y",
                        calibration_index: int = 0,
                        calibration_baseline: np.ndarray | None = None,
                        calibration_baseline_note: str = "others at mean"):
    """Populate `fig` with the requested charts, side by side. `charts` is a
    sequence of the CHART_* keys; drawing fewer of them keeps each one wide
    enough to read. Used by the GUI, where metrics live in a native table.

    `x_train`, `feature_names` and `predict` are only needed for
    CHART_CALIBRATION, which is skipped when they are missing - that keeps
    every existing caller working unchanged. `calibration_index` picks which
    predictor that chart plots against, and `calibration_baseline` pins the
    remaining predictors (defaulting to their training mean); the GUI exposes
    both as dropdowns.
    """
    if charts is None:
        charts = [CHART_LOSS]
    charts = [c for c in charts if c in CHART_LABELS]

    feature_names = list(feature_names) if feature_names is not None else []
    can_calibrate = (
        x_train is not None and predict is not None
        and 0 <= calibration_index < len(feature_names)
    )
    if not can_calibrate:
        charts = [c for c in charts if c != CHART_CALIBRATION]

    with plt.rc_context({"font.family": style.font_family}):
        fig.patch.set_facecolor(style.figure_bg)
        if not charts:
            ax = fig.subplots(1, 1)
            ax.axis("off")
            ax.text(0.5, 0.5, "Select at least one chart", ha="center", va="center", color="#888888")
            return fig

        axes = fig.subplots(1, len(charts), squeeze=False)[0]
        for ax, chart in zip(axes, charts):
            if chart == CHART_LOSS:
                draw_loss_curve(ax, loss_history, style)
            elif chart == CHART_RESIDUALS:
                draw_residuals_plot(ax, y_true, y_pred, style)
            elif chart == CHART_CALIBRATION:
                draw_calibration_pair(ax, x_train, y_true, calibration_index,
                                      feature_names[calibration_index], predict,
                                      label_name, style, calibration_baseline,
                                      calibration_baseline_note)
            else:
                draw_predicted_vs_actual_plot(ax, y_true, y_pred, style)

        fig.tight_layout()
        return fig


def build_dashboard_figure(fig, loss_history: list, y_true: np.ndarray, y_pred: np.ndarray,
                           n_features: int = 1, style: ChartStyle = DEFAULT_CHART_STYLE):
    """Populate `fig` with the 2x2 training dashboard. `fig` can be a pyplot
    figure (standalone popup) or a bare matplotlib.figure.Figure (embedded in
    a GUI canvas) - this function doesn't care which, it never calls show()."""
    with plt.rc_context({"font.family": style.font_family}):
        fig.patch.set_facecolor(style.figure_bg)
        (ax_loss, ax_residuals), (ax_pred_actual, ax_table) = fig.subplots(2, 2)

        draw_loss_curve(ax_loss, loss_history, style)
        draw_residuals_plot(ax_residuals, y_true, y_pred, style)
        draw_predicted_vs_actual_plot(ax_pred_actual, y_true, y_pred, style)
        draw_metrics_table(ax_table, y_true, y_pred, n_features, style)

        fig.tight_layout()
        return fig


def plot_training_results(loss_history: list, y_true: np.ndarray, y_pred: np.ndarray,
                          n_features: int = 1, style: ChartStyle = DEFAULT_CHART_STYLE) -> None:
    fig = plt.figure(figsize=(11, 8))
    build_dashboard_figure(fig, loss_history, y_true, y_pred, n_features, style)
    plt.show()
