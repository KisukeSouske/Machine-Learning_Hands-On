import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

from kai.metrics import loss, squared_loss, mean_squared_error

PANEL_BACKGROUND = "#E5ECF6"
FIGURE_BACKGROUND = "#FFFFFF"
LOSS_CURVE_COLOR = "#0057E7"
MODEL_LINE_COLOR = "#D62D20"
GRID_COLOR = "#FFFFFF"
FONT_FAMILY = "Roboto"


def _format_equation(weight: float, bias: float, feature_name: str, label_name: str) -> str:
    sign = "+" if bias >= 0 else "-"
    return f"{label_name} = {weight:.4f} * {feature_name} {sign} {abs(bias):.4f}"


def _style_panel(ax) -> None:
    ax.set_facecolor(PANEL_BACKGROUND)
    ax.grid(True, color=GRID_COLOR, linewidth=1.2)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_loss_curve(ax, loss_history: list) -> None:
    loss_values = np.asarray(loss_history, dtype=float)
    finite_mask = np.isfinite(loss_values)
    epochs = np.arange(1, len(loss_values) + 1)
    ax.plot(epochs[finite_mask], loss_values[finite_mask], color=LOSS_CURVE_COLOR, linewidth=2.5)
    ax.set_title("Loss curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")


def _draw_fit(ax, x_true: np.ndarray, y_true: np.ndarray, weight: float, bias: float, feature_name: str, label_name: str):
    ax.scatter(x_true, y_true, color="tab:blue", label="Data", zorder=3)
    x_line = np.linspace(x_true.min(), x_true.max(), 200)
    y_line = bias + weight * x_line
    equation = _format_equation(weight, bias, feature_name, label_name)
    line, = ax.plot(x_line, y_line, color=MODEL_LINE_COLOR, linewidth=2, picker=5, label=equation, zorder=4)
    ax.set_title(f"{label_name} vs {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel(label_name)
    return line


def _draw_metrics_table(ax, x_true: np.ndarray, y_true: np.ndarray, weight: float, bias: float) -> None:
    y_pred = bias + weight * x_true
    rows = [
        ("Loss (L1)", loss(y_true, y_pred)),
        ("Squared Loss (L2)", squared_loss(y_true, y_pred)),
        ("MSE", mean_squared_error(y_true, y_pred)),
        ("RMSE", np.sqrt(mean_squared_error(y_true, y_pred))),
    ]

    ax.axis("off")
    table = ax.table(
        cellText=[[name, f"{value:.4f}"] for name, value in rows],
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(2)
        if row == 0:
            cell.set_facecolor(PANEL_BACKGROUND)
            cell.set_text_props(weight="bold", fontfamily=FONT_FAMILY)
        else:
            cell.set_facecolor(FIGURE_BACKGROUND)
            cell.set_text_props(fontfamily=FONT_FAMILY)


def build_training_figure(
    loss_history: list,
    x_true: np.ndarray,
    y_true: np.ndarray,
    weight: float,
    bias: float,
    feature_name: str = "feature",
    label_name: str = "label",
):
    with plt.rc_context({"font.family": FONT_FAMILY}):
        fig, (ax_loss, ax_fit, ax_table) = plt.subplots(
            1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1, 0.7]}
        )
        fig.patch.set_facecolor(FIGURE_BACKGROUND)
        plt.subplots_adjust(bottom=0.3)

        _style_panel(ax_loss)
        _draw_loss_curve(ax_loss, loss_history)

        _style_panel(ax_fit)
        line = _draw_fit(ax_fit, x_true, y_true, weight, bias, feature_name, label_name)

        _draw_metrics_table(ax_table, x_true, y_true, weight, bias)

        # legend lives outside the plot area (below it) so long column names
        # don't cover the data points
        handles, labels = ax_fit.get_legend_handles_labels()
        fit_bbox = ax_fit.get_position()
        legend_center_x = (fit_bbox.x0 + fit_bbox.x1) / 2
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(legend_center_x, 0.23),
            frameon=True,
            facecolor=PANEL_BACKGROUND,
            edgecolor=GRID_COLOR,
            fontsize=10,
        )

        marker, = ax_fit.plot([], [], "o", color="orange", markersize=10, zorder=5)
        annotation = ax_fit.annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        annotation.set_visible(False)

        def show_prediction(x_value: float) -> None:
            y_value = bias + weight * x_value
            marker.set_data([x_value], [y_value])
            annotation.xy = (x_value, y_value)
            annotation.set_text(f"{feature_name} = {x_value:.3g}\n{label_name} = {y_value:.3g}")
            annotation.set_visible(True)
            fig.canvas.draw_idle()

        def on_pick(event) -> None:
            if event.artist is not line or event.mouseevent.xdata is None:
                return
            show_prediction(event.mouseevent.xdata)

        fig.canvas.mpl_connect("pick_event", on_pick)

        textbox_ax = fig.add_axes([0.3, 0.05, 0.25, 0.05])
        textbox = TextBox(textbox_ax, f"{feature_name} = ")

        def on_submit(text: str) -> None:
            try:
                x_value = float(text)
            except ValueError:
                return
            show_prediction(x_value)

        textbox.on_submit(on_submit)

        # keep a strong reference alive, otherwise the widget can be garbage
        # collected and silently stop responding to input
        fig.kai_textbox = textbox

        return fig


def plot_training_results(
    loss_history: list,
    x_true: np.ndarray,
    y_true: np.ndarray,
    weight: float,
    bias: float,
    feature_name: str = "feature",
    label_name: str = "label",
) -> None:
    build_training_figure(loss_history, x_true, y_true, weight, bias, feature_name, label_name)
    plt.show()
