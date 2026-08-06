"""The main window: layout and event wiring only.

Visual decisions live in `kai.themes`, widget drawing in `kai.gui.widgets`,
training/persistence in `kai.gui.controller`, and typed session state in
`kai.gui.state` - this module just composes them.
"""
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd

from kai.gui.controller import TrainingController, save_results_report
from kai.gui.helpers import (
    count_csv_data_rows,
    detect_csv_separator,
    format_elapsed,
    format_prediction,
    humanize_column,
    list_csv_files,
    read_csv_preview,
)
from kai.gui.state import (
    Hyperparameters,
    TrainingRequest,
    TrainingResult,
    coefficients_in_original_space,
    intercept_in_original_space,
    training_scaling,
)
from kai.gui.widgets import (
    ChartTabBar,
    HelpHint,
    MultiSelectDropdown,
    PanelHeader,
    StatusBar,
    ThemedButton,
)
from kai.metrics import f_statistic
from kai.model import Model
from kai.regression import (
    InferenceSummary,
    LinearFit,
    PredictionInterval,
    TrainingCancelled,
    fit_ols,
    predict_with_intervals,
    summarize_inference,
    variance_inflation_factors,
)
from kai.themes import Theme, get_theme
from kai.visualization import (
    CHART_CALIBRATION,
    CHART_LABELS,
    CHART_LOSS,
    CHART_PREDICTED_VS_ACTUAL,
    CHART_RESIDUALS,
    FAMILY_GAMMA_LOG,
    FAMILY_GAUSSIAN,
    FAMILY_LABELS,
    build_charts_figure,
    family_label,
    metrics_rows,
)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Standardizing changes the geometry of the loss surface: it pulls the Hessian
# condition number toward 1, which raises the stable step size by orders of
# magnitude. A learning rate tuned for raw columns is therefore far too small
# once Z-scoring is on (and vice versa), so each mode carries its own default.
RAW_LEARNING_RATE = 0.0003
STANDARDIZED_LEARNING_RATE = 0.05

# layout geometry (not part of the theme: themes change looks, not structure)
DATA_PANEL_WIDTH = 430
CONFIG_PANEL_WIDTH = 231  # 165 * 1.4
PREVIEW_HEIGHT = 150
TOP_ROW_HEIGHT = 250
STATUS_PANEL_WIDTH = 240
TESTING_PANEL_WIDTH = 260

# Where the calibration chart pins the predictors it is NOT sweeping. The mean
# is the neutral default, but the interesting failure modes of a linear model
# only show up away from it - a positive-only response can be predicted
# negative once the other predictors sit low enough, which is invisible at the
# mean. "Testing inputs" reads the values typed in the Model Testing panel, so
# any combination (including extrapolation beyond the data) can be inspected.
CALIBRATION_AT_MEAN = "Mean"
CALIBRATION_AT_MIN = "Min"
CALIBRATION_AT_MAX = "Max"
CALIBRATION_AT_INPUTS = "Testing inputs"
CALIBRATION_BASELINES = (
    CALIBRATION_AT_MEAN, CALIBRATION_AT_MIN, CALIBRATION_AT_MAX, CALIBRATION_AT_INPUTS,
)

# Explanatory blurbs for the "?" hints. Kept together so wording stays consistent
# and translations (should we ever add them) live in one place.
HELP_LEARNING_RATE = (
    "Step size for gradient descent. Too high => training diverges (loss "
    "explodes). Too low => training crawls. If you standardize the features, "
    "you can safely use a much larger value."
)
HELP_EPOCHS = (
    "Maximum passes over the dataset. Training stops earlier once the "
    "convergence criterion is met; this is a safety ceiling, not a target."
)
HELP_BATCH_SIZE = (
    "Samples used in each gradient step. Values >= number of samples reduce "
    "this to plain full-batch gradient descent."
)
HELP_TOLERANCE = (
    "Relative convergence threshold: training stops when the gradient norm "
    "falls to this fraction of its initial value. Being relative, it is "
    "invariant to the units of y and of the features."
)
HELP_STANDARDIZE = (
    "Z-score each predictor before training (subtract mean, divide by std). "
    "Strongly recommended when features live on different scales; it makes "
    "gradient descent converge in orders of magnitude fewer epochs."
)

HELP_LOSS_L1 = "Sum of absolute residuals: total_L1 = sum(|y_true - y_pred|)."
HELP_SQUARED_LOSS = "Sum of squared residuals (RSS): sum((y_true - y_pred)^2)."
HELP_MSE = "Mean squared error: RSS / n. Same units as y^2."
HELP_RMSE = (
    "Root mean squared error: sqrt(MSE). Same units as y, so directly "
    "comparable to typical values of the target."
)
HELP_R2 = (
    "Proportion of variance in y explained by the model: 1 - RSS/TSS. "
    "1.0 = perfect fit, 0.0 = no better than predicting the mean, "
    "negative = worse than the mean."
)
HELP_R2_ADJ = (
    "R² penalized for the number of predictors, so adding a useless feature "
    "no longer inflates the score (ISLR eq. 6.4)."
)
HELP_COEF = (
    "Estimated regression coefficient (in the same feature space that was "
    "used for training - if standardization is on, the coefficient is in "
    "standardized space)."
)
HELP_SE = (
    "Standard error of the coefficient: how much this estimate would jitter "
    "across different samples from the same population (ISLR eq. 3.8, matrix "
    "form for multiple regression)."
)
HELP_T = (
    "t-statistic = coefficient / SE. Measures how many standard errors the "
    "estimate is away from zero (ISLR eq. 3.14)."
)
HELP_P = (
    "Two-tailed p-value for H0: coefficient = 0. Small p (< 0.05) is evidence "
    "the predictor matters. Look at the F-statistic first to decide whether "
    "the model as a whole is useful, then read the individual p-values."
)
HELP_F = (
    "F-statistic tests whether at least one predictor is useful (H0: all "
    "coefficients are zero). Values much greater than 1 reject H0 (ISLR "
    "eq. 3.23)."
)
HELP_VIF = (
    "Variance Inflation Factor: how much the variance of a coefficient is "
    "inflated by collinearity with the other predictors. VIF = 1 means no "
    "collinearity; > 5 or 10 is a common warning threshold (ISLR p.102)."
)
HELP_PREDICTORS = (
    "One row per predictor. 'Coefficient' is always in ORIGINAL units - how "
    "much the response moves per one unit of that predictor - so it stays "
    "comparable across runs even when training standardized the features. "
    "'Std. coef' is the same effect per one standard deviation, which is what "
    "you compare when predictors are measured on different scales: a small "
    "coefficient on a wide-ranging variable can matter more than a large one "
    "on a narrow variable. 'VIF' flags collinearity with the other predictors."
)
HELP_FAMILY = (
    "The distribution and link the gradient-descent engine fits. "
    "Normal/identity is ordinary least squares: the response can come out "
    "negative, and the errors are assumed to have constant spread. "
    "Gamma/log models a strictly positive response whose spread grows with "
    "its mean - it predicts exp(linear part), so it can never return a "
    "negative value, and its coefficients act multiplicatively. Only "
    "gradient descent supports a family; the closed-form OLS solver is "
    "normal/identity by construction."
)
HELP_INTERVALS = (
    "95% confidence interval: uncertainty about the MEAN response at this "
    "input. 95% prediction interval: uncertainty about a single NEW "
    "observation at this input - wider, since it also includes the "
    "individual error term (ISLR eq. 3.9-3.11). Both require OLS."
)
HELP_METHOD = (
    "How the coefficients are estimated. Gradient descent is iterative and "
    "produces a loss curve you can watch; OLS solves the normal equations "
    "in closed form (no iterations, no hyperparameters), giving the exact "
    "least-squares fit and enabling standard errors / t-tests / F-statistic "
    "on the coefficients."
)

# Only OLS produces statistically valid standard errors / t-tests / F-statistic:
# those formulas assume the exact least-squares minimum, which iterative
# gradient descent only approaches, never touches. So the Inference tab is
# disabled when GD is the active method.
INFERENCE_GD_MESSAGE = (
    "Inference requires the closed-form solution. Switch the estimation "
    "method to OLS to see coefficient standard errors, t-statistics, "
    "p-values and the F-statistic."
)


class TrainingApp(tk.Tk):
    """Main window. Pick the visual theme at construction:

        TrainingApp(theme="default")   # Aero look
        TrainingApp(theme="retro_os")  # classic/retro look
    """

    def __init__(self, csv_dir=None, theme: str | Theme = "default"):
        super().__init__()
        self.theme = get_theme(theme)
        self.title("ML Model Trainer - Supervised Regression")
        self.geometry("1500x900")
        self.minsize(1200, 700)
        self.configure(bg=self.theme.palette.window_bg)

        self.csv_dir = Path(csv_dir) if csv_dir is not None else Path(__file__).resolve().parent.parent.parent
        self.current_csv_path: Path | None = None
        self._current_separator: str = ","
        self._all_columns: list[str] = []
        self._preview_headers: dict[str, tk.Label] = {}
        self._preview_cells: dict[str, list[tk.Label]] = {}
        self._training_start_time = 0.0
        self._syncing_slider = False
        self._last_result: TrainingResult | None = None
        # A fresh Model built alongside every completed run, so the Testing
        # panel can call predict(x) and have the training scaling reapplied.
        self._last_model: Model | None = None

        self.controller = TrainingController(schedule_on_ui=lambda cb: self.after(0, cb))

        self._apply_ttk_styles()
        self._build_layout()

    # ------------------------------------------------------------------ #
    # Chrome
    # ------------------------------------------------------------------ #
    def _apply_ttk_styles(self) -> None:
        palette, fonts = self.theme.palette, self.theme.fonts
        style = ttk.Style(self)
        for candidate in self.theme.ttk_theme_candidates:
            if candidate in style.theme_names():
                style.theme_use(candidate)
                break
        style.configure("TFrame", background=palette.panel_bg)
        style.configure("App.TFrame", background=palette.window_bg)
        style.configure("TLabel", background=palette.panel_bg, foreground=palette.text_fg,
                        font=fonts.body)
        style.configure("Field.TLabel", background=palette.panel_bg, foreground=palette.text_fg,
                        font=fonts.body)
        style.configure("Small.TLabel", background=palette.panel_bg, foreground=palette.text_fg,
                        font=fonts.small)
        style.configure("Stopwatch.TLabel", background=palette.panel_bg,
                        foreground=palette.text_fg, font=fonts.stopwatch)
        # The readout is split in two: the target name in the ordinary UI font
        # (it is prose, not a numeric display) and the value below it. Sized so
        # that neither can outgrow TESTING_PANEL_WIDTH and get clipped.
        style.configure("PredictionName.TLabel", background=palette.panel_bg,
                        foreground=palette.text_fg, font=fonts.small)
        style.configure("Prediction.TLabel", background=palette.panel_bg,
                        foreground=palette.text_fg,
                        font=(fonts.stopwatch[0], 15, "bold"))
        style.configure("TCheckbutton", background=palette.panel_bg, foreground=palette.text_fg,
                        font=fonts.body)
        style.configure("TNotebook", background=palette.panel_bg)
        style.configure("TNotebook.Tab", font=fonts.body, padding=(8, 3))
        style.map("TNotebook.Tab", background=[("selected", palette.cell_bg)])
        style.configure("Treeview", background=palette.cell_bg, fieldbackground=palette.cell_bg,
                        foreground=palette.text_fg, font=fonts.body)
        style.configure("Treeview.Heading", font=fonts.body_bold,
                        background=palette.preview_header_bg, foreground=palette.text_fg)
        style.map("Treeview", background=[("selected", palette.select_bg)],
                  foreground=[("selected", palette.select_fg)])
        style.configure("TEntry", fieldbackground=palette.cell_bg)
        style.configure("TCombobox", fieldbackground=palette.cell_bg)
        style.configure("TSeparator", background=palette.panel_border)

    def _build_layout(self) -> None:
        self.status_bar = StatusBar(self, self.theme)
        self.status_bar.pack(side="bottom", fill="x")
        self.status_bar.set_text("Ready for selection...")

        # fixed layout: data + hyperparameters keep a constant width on the
        # left, the training area takes whatever is left over
        root = ttk.Frame(self, style="App.TFrame")
        root.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(root, width=DATA_PANEL_WIDTH)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        config_column = ttk.Frame(root, width=CONFIG_PANEL_WIDTH)
        config_column.pack(side="left", fill="y", padx=(8, 0))
        config_column.pack_propagate(False)

        center = ttk.Frame(root)
        center.pack(side="left", fill="both", expand=True, padx=(8, 0))

        data_panel = self._panel(left, "Data Selection & Preview")
        data_panel.pack(fill="both", expand=True)

        config_panel = self._panel(config_column, "Hyperparameter Configuration")
        config_panel.pack(fill="both", expand=True)

        # Top row (status + metrics + testing) sits ABOVE Model Training.
        # Model Training then takes all the remaining vertical space.
        top_row = ttk.Frame(center, height=TOP_ROW_HEIGHT)
        top_row.pack(side="top", fill="x")
        top_row.pack_propagate(False)

        status_panel = self._panel(top_row, "Training Status")
        status_panel.pack(side="left", fill="both")
        status_panel.configure(width=STATUS_PANEL_WIDTH)
        status_panel.pack_propagate(False)

        testing_panel = self._panel(top_row, "Model Testing")
        testing_panel.pack(side="left", fill="both", padx=(8, 0))
        testing_panel.configure(width=TESTING_PANEL_WIDTH)
        testing_panel.pack_propagate(False)

        metrics_panel = self._panel(top_row, "Training Metrics")
        metrics_panel.pack(side="left", fill="both", expand=True, padx=(8, 0))

        charts_panel = self._panel(center, "Model Training")
        charts_panel.pack(side="top", fill="both", expand=True, pady=(8, 0))

        self._build_data_panel(data_panel.body)
        self._build_config_panel(config_panel.body)
        self._build_charts_panel(charts_panel.body)
        self._build_status_panel(status_panel.body)
        self._build_metrics_panel(metrics_panel.body)
        self._build_testing_panel(testing_panel.body)

    def _panel(self, parent, title: str) -> tk.Frame:
        """A bordered, titled panel; content goes into `panel.body`."""
        palette = self.theme.palette
        panel = tk.Frame(parent, bg=palette.panel_border, highlightthickness=0, bd=0)
        inner = tk.Frame(panel, bg=palette.panel_bg)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        PanelHeader(inner, title, self.theme).pack(fill="x")
        panel.body = ttk.Frame(inner)
        panel.body.pack(fill="both", expand=True, padx=8, pady=8)
        return panel

    def _label_with_hint(self, parent, text: str, tooltip: str, style: str = "Small.TLabel"):
        """A form label followed by a tiny "?" hint. Returned as a row so the
        caller can `.pack()` it in a form."""
        row = ttk.Frame(parent)
        ttk.Label(row, text=text, style=style).pack(side="left")
        HelpHint(row, self.theme, tooltip).pack(side="left", padx=(4, 0), pady=(2, 0))
        return row

    # ------------------------------------------------------------------ #
    # Data selection & preview
    # ------------------------------------------------------------------ #
    def _build_data_panel(self, parent) -> None:
        palette = self.theme.palette
        ttk.Label(parent, text="Select CSV File", style="Field.TLabel").pack(anchor="w")
        csv_row = ttk.Frame(parent)
        csv_row.pack(fill="x", pady=(2, 0))
        self.csv_var = tk.StringVar()
        self.csv_combo = ttk.Combobox(csv_row, textvariable=self.csv_var, state="readonly")
        self.csv_combo.pack(side="left", fill="x", expand=True)
        self.csv_combo.bind("<Button-1>", lambda _e: self._refresh_csv_list())
        self.csv_combo.bind("<<ComboboxSelected>>", self._on_csv_selected)
        self._refresh_csv_list()
        self.browse_button = ThemedButton(csv_row, "Browse...", self.theme, command=self._on_browse_csv)
        self.browse_button.configure(width=80)
        self.browse_button.pack(side="left", padx=(6, 0))

        # fixed (not expanding) height so the column pickers sit right below it.
        # its children are grid-managed, so grid_propagate is what pins the size
        preview_box = ttk.Frame(parent, height=PREVIEW_HEIGHT)
        preview_box.pack(fill="x")
        preview_box.grid_propagate(False)
        self.preview_canvas = tk.Canvas(
            preview_box, bg=palette.cell_bg, highlightthickness=1,
            highlightbackground=palette.panel_border,
        )
        vbar = ttk.Scrollbar(preview_box, orient="vertical", command=self.preview_canvas.yview)
        hbar = ttk.Scrollbar(preview_box, orient="horizontal", command=self.preview_canvas.xview)
        self.preview_canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        preview_box.rowconfigure(0, weight=1)
        preview_box.columnconfigure(0, weight=1)

        self.preview_frame = tk.Frame(self.preview_canvas, bg=palette.cell_bg)
        self.preview_canvas.create_window((0, 0), window=self.preview_frame, anchor="nw")
        self.preview_frame.bind(
            "<Configure>",
            lambda _e: self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all")),
        )
        self.preview_canvas.bind(
            "<MouseWheel>",
            lambda e: self.preview_canvas.yview_scroll(int(-e.delta / 120), "units"),
        )

        self.row_count_label = tk.Label(
            parent, text="", font=self.theme.fonts.small, fg="#555555", bg=palette.panel_bg,
        )
        self.row_count_label.pack(anchor="w", pady=(2, 0))

        ttk.Label(parent, text="Select Label Column", style="Field.TLabel").pack(anchor="w", pady=(10, 0))
        self.label_var = tk.StringVar()
        self.label_combo = ttk.Combobox(parent, textvariable=self.label_var, state="disabled")
        self.label_combo.pack(fill="x", pady=(2, 10))
        self.label_combo.bind("<<ComboboxSelected>>", self._on_label_selected)

        ttk.Label(parent, text="Select Feature Columns", style="Field.TLabel").pack(anchor="w")
        self.feature_dropdown = MultiSelectDropdown(
            parent, self.theme, placeholder="Select feature columns",
            on_change=self._on_features_changed,
        )
        self.feature_dropdown.pack(fill="x", pady=2)

        self.chips_text = tk.Text(
            parent, height=3, wrap="char", bg=palette.cell_bg, fg=palette.text_fg,
            relief="solid", borderwidth=1, cursor="arrow", state="disabled",
        )
        self.chips_text.pack(fill="x", pady=(4, 0))

    def _refresh_csv_list(self) -> None:
        self.csv_combo["values"] = list_csv_files(self.csv_dir)

    def _on_csv_selected(self, _event=None) -> None:
        filename = self.csv_var.get()
        if not filename:
            return
        self._load_csv(self.csv_dir / filename)

    def _on_browse_csv(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select a CSV file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not chosen:
            return
        path = Path(chosen)
        # Keep the combobox in sync when the browsed file also lives in csv_dir;
        # otherwise leave it blank so the field doesn't lie about where it's from.
        self.csv_var.set(path.name if path.parent == self.csv_dir else "")
        self._load_csv(path)

    def _load_csv(self, path: Path) -> None:
        try:
            separator = detect_csv_separator(path)
            preview_df = read_csv_preview(path, sep=separator)
            row_count = count_csv_data_rows(path)
        except Exception as exc:
            messagebox.showerror("Failed to read CSV", str(exc))
            return

        self.current_csv_path = path
        self._current_separator = separator
        self._all_columns = list(preview_df.columns)
        self._build_preview(preview_df)
        self.row_count_label.configure(text=f"{row_count} example(s) loaded")

        self.label_combo["values"] = self._all_columns
        self.label_var.set("")
        self.label_combo.configure(state="readonly")
        self.feature_dropdown.configure_state(True)
        self.feature_dropdown.set_selected([])
        self._refresh_feature_options()
        self._render_chips()
        self._recolor_preview()
        self.log(f"Loaded {path.name} ({len(self._all_columns)} columns, separator {separator!r})")
        self.status_bar.set_text(f"{path.name} loaded - select the label and feature columns")

    def _build_preview(self, df: pd.DataFrame) -> None:
        palette, fonts = self.theme.palette, self.theme.fonts
        for child in self.preview_frame.winfo_children():
            child.destroy()
        self._preview_headers = {}
        self._preview_cells = {}

        tk.Label(
            self.preview_frame, text="#", font=fonts.body_bold, fg=palette.text_fg,
            bg=palette.preview_header_bg, borderwidth=1, relief="solid", padx=6, pady=4,
        ).grid(row=0, column=0, sticky="nsew")

        for col_idx, col in enumerate(df.columns, start=1):
            header = tk.Label(
                self.preview_frame, text=str(col), font=fonts.body_bold, fg=palette.text_fg,
                bg=palette.preview_header_bg, borderwidth=1, relief="solid", padx=8, pady=4,
            )
            header.grid(row=0, column=col_idx, sticky="nsew")
            self._preview_headers[col] = header
            self._preview_cells[col] = []

        for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
            tk.Label(
                self.preview_frame, text=str(row_idx), bg=palette.preview_header_bg,
                fg=palette.text_fg, borderwidth=1, relief="solid", padx=6, pady=2,
            ).grid(row=row_idx, column=0, sticky="nsew")
            for col_idx, col in enumerate(df.columns, start=1):
                cell = tk.Label(
                    self.preview_frame, text=str(row[col]), bg=palette.cell_bg,
                    fg=palette.text_fg, borderwidth=1, relief="solid", padx=8, pady=2,
                )
                cell.grid(row=row_idx, column=col_idx, sticky="nsew")
                self._preview_cells[col].append(cell)

    def _refresh_feature_options(self) -> None:
        label_col = self.label_var.get()
        self.feature_dropdown.set_options([c for c in self._all_columns if c != label_col])

    def _get_selected_features(self) -> list[str]:
        return self.feature_dropdown.get_selected()

    def _on_features_changed(self) -> None:
        self._render_chips()
        self._recolor_preview()

    def _remove_feature(self, column: str) -> None:
        self.feature_dropdown.set_selected([c for c in self._get_selected_features() if c != column])
        self._on_features_changed()

    def _render_chips(self) -> None:
        palette, fonts = self.theme.palette, self.theme.fonts
        self.chips_text.configure(state="normal")
        self.chips_text.delete("1.0", tk.END)
        for column in self._get_selected_features():
            chip = tk.Frame(self.chips_text, bg=palette.chip_bg, padx=4, pady=1,
                            highlightthickness=1, highlightbackground=palette.panel_border)
            tk.Label(chip, text=column, bg=palette.chip_bg, fg=palette.text_fg,
                     font=fonts.small).pack(side="left")
            tk.Button(
                chip, text="×", bg=palette.chip_bg, fg=palette.text_fg, relief="flat",
                font=fonts.small, cursor="hand2", padx=2, pady=0, borderwidth=0,
                command=lambda c=column: self._remove_feature(c),
            ).pack(side="left")
            self.chips_text.window_create(tk.END, window=chip, padx=2, pady=2)
        self.chips_text.configure(state="disabled")

    def _on_label_selected(self, _event=None) -> None:
        label_col = self.label_var.get()
        # a column can't be both the target and a predictor
        self.feature_dropdown.set_selected([c for c in self._get_selected_features() if c != label_col])
        self._refresh_feature_options()
        self._render_chips()
        self._recolor_preview()

    def _recolor_preview(self) -> None:
        palette = self.theme.palette
        label_col = self.label_var.get()
        selected = set(self._get_selected_features())
        for col, header in self._preview_headers.items():
            if col == label_col:
                header_color = cell_color = palette.label_highlight
            elif col in selected:
                header_color = cell_color = palette.feature_highlight
            else:
                header_color, cell_color = palette.preview_header_bg, palette.cell_bg
            header.configure(bg=header_color)
            for cell in self._preview_cells[col]:
                cell.configure(bg=cell_color)

    # ------------------------------------------------------------------ #
    # Charts / metrics / status / testing
    # ------------------------------------------------------------------ #
    def _build_charts_panel(self, parent) -> None:
        tab_row = ttk.Frame(parent)
        tab_row.pack(fill="x", pady=(0, 6))

        self.chart_tabs = ChartTabBar(
            tab_row,
            self.theme,
            [(key, CHART_LABELS[key]) for key in
             (CHART_LOSS, CHART_RESIDUALS, CHART_PREDICTED_VS_ACTUAL, CHART_CALIBRATION)],
            active_keys=[CHART_LOSS],
            on_change=self._render_charts,
        )
        self.chart_tabs.pack(side="left")

        # Controls for the calibration chart. Packed right-to-left, so the
        # visual order ends up "Calibration vs: [x] | others at: [baseline]".
        # Both are populated after a run, since the features are only known then.
        self.calibration_baseline_var = tk.StringVar(value=CALIBRATION_AT_MEAN)
        self.calibration_baseline_combo = ttk.Combobox(
            tab_row, textvariable=self.calibration_baseline_var, state="disabled",
            width=14, values=CALIBRATION_BASELINES,
        )
        self.calibration_baseline_combo.pack(side="right")
        self.calibration_baseline_combo.bind("<<ComboboxSelected>>",
                                             lambda _event: self._render_charts())
        ttk.Label(tab_row, text="others at:", style="Small.TLabel").pack(
            side="right", padx=(8, 4))

        self.calibration_var = tk.StringVar(value="")
        self.calibration_combo = ttk.Combobox(
            tab_row, textvariable=self.calibration_var, state="disabled",
            width=18, values=(),
        )
        self.calibration_combo.pack(side="right")
        self.calibration_combo.bind("<<ComboboxSelected>>",
                                    lambda _event: self._render_charts())
        ttk.Label(tab_row, text="Calibration vs:", style="Small.TLabel").pack(
            side="right", padx=(8, 4))

        self.figure = Figure(figsize=(9, 4))
        self.figure.patch.set_facecolor(self.theme.charts.figure_bg)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)

        # matplotlib's own toolbar: pan, rectangle zoom, back/forward, save
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side="bottom", fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x")
        self._restyle_toolbar()

        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

    def _restyle_toolbar(self) -> None:
        background = self.theme.palette.panel_bg
        try:
            self.toolbar.configure(background=background)
        except tk.TclError:
            pass
        for child in self.toolbar.winfo_children():
            try:
                child.configure(background=background)
            except tk.TclError:
                pass

    def _selected_charts(self) -> list[str]:
        return self.chart_tabs.get_active()

    @staticmethod
    def _result_family(result: TrainingResult) -> tuple[str, str]:
        """The family a finished run actually used. Read from the result, not
        from the dropdown: the user may have changed the dropdown since, and
        the displayed metrics must describe the model on screen."""
        if result.request.method != "gd":
            return FAMILY_GAUSSIAN
        hp = result.request.hyperparameters
        return (hp.loss_function, hp.loss_function_link)

    def _rebuild_calibration_choices(self, feature_names: tuple[str, ...]) -> None:
        """Point the calibration dropdown at the features of the last run,
        keeping the current pick when it survived the new selection."""
        features = list(feature_names)
        self.calibration_combo.configure(values=features)
        if not features:
            self.calibration_var.set("")
            self.calibration_combo.configure(state="disabled")
            self.calibration_baseline_combo.configure(state="disabled")
            return
        if self.calibration_var.get() not in features:
            self.calibration_var.set(features[0])
        self.calibration_combo.configure(state="readonly")
        self.calibration_baseline_combo.configure(state="readonly")

    def _calibration_baseline(self, result: TrainingResult) -> tuple[np.ndarray, str]:
        """The point at which the calibration chart pins the other predictors,
        plus a short note naming it for the legend.

        Falls back to the mean whenever the requested baseline cannot be built
        - notably when "Testing inputs" is selected but a field is blank or
        not a number, which would otherwise raise mid-render.
        """
        choice = self.calibration_baseline_var.get()
        x_train = np.asarray(result.x_train, dtype=float)
        if choice == CALIBRATION_AT_MIN:
            return x_train.min(axis=0), "others at min"
        if choice == CALIBRATION_AT_MAX:
            return x_train.max(axis=0), "others at max"
        if choice == CALIBRATION_AT_INPUTS:
            try:
                values = [float(self._testing_inputs[name].get())
                          for name in result.request.features]
            except (KeyError, ValueError):
                return x_train.mean(axis=0), "others at mean (inputs invalid)"
            return np.array(values, dtype=float), "others at testing inputs"
        return x_train.mean(axis=0), "others at mean"

    def _render_charts(self) -> None:
        if self._last_result is None:
            return
        result = self._last_result
        # OLS has no iterations, so silently drop the Loss chart from the
        # selection if it happens to be active. list() copy so we do not mutate
        # the tab bar's own state.
        selected = list(self._selected_charts())
        if result.loss_history is None and CHART_LOSS in selected:
            selected.remove(CHART_LOSS)
        loss_history = list(result.loss_history) if result.loss_history is not None else []
        features = list(result.request.features)
        # the dropdown holds a feature name; fall back to the first predictor
        # if it is empty or stale (e.g. right after a run with new features)
        try:
            calibration_index = features.index(self.calibration_var.get())
        except ValueError:
            calibration_index = 0
        baseline, baseline_note = self._calibration_baseline(result)
        self.figure.clear()
        build_charts_figure(
            self.figure, loss_history, result.y_true, result.y_pred,
            charts=selected, style=self.theme.charts,
            x_train=result.x_train, feature_names=features,
            predict=self._last_model.predict if self._last_model else None,
            label_name=result.request.label_column,
            calibration_index=calibration_index,
            calibration_baseline=baseline,
            calibration_baseline_note=baseline_note,
            family=self._result_family(result),
        )
        self.canvas.draw()
        # reset the zoom/pan history so the toolbar's "home" matches the new axes
        self.toolbar.update()
        self._restyle_toolbar()

    def _build_metrics_panel(self, parent) -> None:
        # honest framing: there is no train/test split here, so these numbers
        # describe fit on the training data, not generalization (ISLR ch. 2)
        ttk.Label(
            parent,
            text="In-sample metrics: computed on the training data, so they measure fit, "
                 "not generalization to unseen data.",
            style="Small.TLabel", wraplength=520, justify="left",
        ).pack(side="bottom", fill="x", pady=(6, 0))

        notebook = ttk.Notebook(parent)
        notebook.pack(side="top", fill="both", expand=True)

        # --- Fit tab: the old two-column table (metric, value) ---
        fit_tab = ttk.Frame(notebook)
        notebook.add(fit_tab, text="Fit")
        self.metrics_tree = ttk.Treeview(fit_tab, columns=("metric", "value"),
                                          show="headings", height=6)
        self.metrics_tree.heading("metric", text="Metric")
        self.metrics_tree.heading("value", text="Value")
        self.metrics_tree.column("metric", anchor="w", width=180)
        self.metrics_tree.column("value", anchor="e", width=120)
        fit_scrollbar = ttk.Scrollbar(fit_tab, orient="vertical", command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=fit_scrollbar.set)
        self.metrics_tree.pack(side="left", fill="both", expand=True)
        fit_scrollbar.pack(side="right", fill="y")

        # --- Inference tab: coefficient / SE / t / p, one row per parameter ---
        inference_tab = ttk.Frame(notebook)
        notebook.add(inference_tab, text="Inference")

        self.f_stat_var = tk.StringVar(value="F-statistic: run a model to see the value.")
        f_row = ttk.Frame(inference_tab)
        f_row.pack(fill="x", pady=(0, 4))
        ttk.Label(f_row, textvariable=self.f_stat_var, style="Small.TLabel").pack(side="left")
        HelpHint(f_row, self.theme, HELP_F).pack(side="left", padx=(4, 0))

        cols = ("name", "coef", "se", "t", "p")
        self.inference_tree = ttk.Treeview(inference_tab, columns=cols, show="headings", height=6)
        headings = [("name", "Parameter", "w", 110),
                    ("coef", "Coefficient", "e", 110),
                    ("se", "Std. error", "e", 100),
                    ("t", "t", "e", 70),
                    ("p", "p-value", "e", 100)]
        for key, label, anchor, width in headings:
            self.inference_tree.heading(key, text=label)
            self.inference_tree.column(key, anchor=anchor, width=width)
        inf_scrollbar = ttk.Scrollbar(inference_tab, orient="vertical",
                                       command=self.inference_tree.yview)
        self.inference_tree.configure(yscrollcommand=inf_scrollbar.set)
        self.inference_tree.pack(side="left", fill="both", expand=True)
        inf_scrollbar.pack(side="right", fill="y")

        # --- Predictors tab: the fitted coefficient of each predictor, with
        # its VIF alongside so collinearity is read next to the number it
        # actually undermines (a large coefficient on a highly collinear
        # predictor is not the stable effect it looks like).
        predictors_tab = ttk.Frame(notebook)
        notebook.add(predictors_tab, text="Predictors")
        predictors_row = ttk.Frame(predictors_tab)
        predictors_row.pack(fill="x", pady=(0, 4))
        self.predictors_note_var = tk.StringVar(value="Coefficients in original units.")
        ttk.Label(predictors_row, textvariable=self.predictors_note_var,
                  style="Small.TLabel").pack(side="left")
        HelpHint(predictors_row, self.theme, HELP_PREDICTORS).pack(side="left", padx=(4, 0))
        HelpHint(predictors_row, self.theme, HELP_VIF).pack(side="left", padx=(2, 0))

        predictor_cols = ("feature", "coef", "std_coef", "vif")
        self.predictors_tree = ttk.Treeview(predictors_tab, columns=predictor_cols,
                                            show="headings", height=6)
        predictor_headings = [("feature", "Predictor", "w", 140),
                              ("coef", "Coefficient", "e", 110),
                              ("std_coef", "Std. coef", "e", 100),
                              ("vif", "VIF", "e", 80)]
        for key, label, anchor, width in predictor_headings:
            self.predictors_tree.heading(key, text=label)
            self.predictors_tree.column(key, anchor=anchor, width=width)
        predictors_scrollbar = ttk.Scrollbar(predictors_tab, orient="vertical",
                                             command=self.predictors_tree.yview)
        self.predictors_tree.configure(yscrollcommand=predictors_scrollbar.set)
        self.predictors_tree.pack(side="left", fill="both", expand=True)
        predictors_scrollbar.pack(side="right", fill="y")

    def _build_status_panel(self, parent) -> None:
        self.stopwatch_var = tk.StringVar(value=format_elapsed(0))
        ttk.Label(parent, textvariable=self.stopwatch_var, style="Stopwatch.TLabel",
                  anchor="center").pack(fill="x", pady=(4, 2))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.status_var, anchor="center").pack(fill="x")
        self.start_button = ThemedButton(parent, "START TRAINING", self.theme,
                                          command=self._start_training, height=32)
        self.start_button.pack(fill="x", pady=(8, 4))
        self.stop_button = ThemedButton(parent, "STOP", self.theme,
                                         command=self._on_stop_training, height=28)
        self.stop_button.configure(state="disabled")
        self.stop_button.pack(fill="x", pady=(0, 4))
        self.save_button = ThemedButton(parent, "SAVE RESULTS", self.theme,
                                         command=self._save_results, height=28)
        self.save_button.configure(state="disabled")
        self.save_button.pack(fill="x")

    def _build_testing_panel(self, parent) -> None:
        """Live prediction for a single input row, populated after a training run."""
        # Packed first, anchored to the bottom: this claims its slot in the
        # panel's fixed height BEFORE anything else, so it can never be
        # crowded out or clipped by the (variable-length) content above it.
        self.predict_button = ThemedButton(parent, "PREDICT", self.theme,
                                            command=self._on_predict, height=28)
        self.predict_button.configure(state="disabled")
        self.predict_button.pack(fill="x", side="bottom")

        # The whole readout is bottom-anchored and packed BEFORE the input
        # form, for the same reason as the button: pack hands out space in
        # packing order, so anything declared after the form (which expands)
        # gets squeezed to nothing inside this panel's fixed height. That is
        # what used to swallow the predicted value.
        interval_row = ttk.Frame(parent)
        interval_row.pack(side="bottom", fill="x", pady=(0, 2))
        HelpHint(interval_row, self.theme, HELP_INTERVALS).pack(side="left", anchor="n", padx=(0, 4))
        self.interval_var = tk.StringVar(value="")
        ttk.Label(interval_row, textvariable=self.interval_var, style="Small.TLabel",
                  justify="left", wraplength=TESTING_PANEL_WIDTH - 40).pack(side="left", fill="x", expand=True)

        # Name and value on separate lines: together on one line they overflow
        # the panel width for any realistic column name, and the number (drawn
        # last) is the part that falls off the edge.
        self.prediction_var = tk.StringVar(value="—")
        ttk.Label(parent, textvariable=self.prediction_var, style="Prediction.TLabel",
                  anchor="center").pack(side="bottom", fill="x", pady=(0, 2))
        self.prediction_name_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.prediction_name_var, style="PredictionName.TLabel",
                  anchor="center", wraplength=TESTING_PANEL_WIDTH - 24,
                  justify="center").pack(side="bottom", fill="x", pady=(4, 0))

        ttk.Label(parent, text="Predict a single input in the ORIGINAL feature space.",
                  style="Small.TLabel", wraplength=TESTING_PANEL_WIDTH - 24,
                  justify="left").pack(anchor="w", pady=(0, 4))

        # Container refreshed after each training run - the input rows change
        # because the feature list changes.
        self._testing_form = ttk.Frame(parent)
        self._testing_form.pack(fill="both", expand=True)
        self._testing_inputs: dict[str, tk.StringVar] = {}
        self._testing_empty_hint = ttk.Label(
            self._testing_form, text="Train a model first.",
            style="Small.TLabel", anchor="center",
        )
        self._testing_empty_hint.pack(pady=8)

    def _rebuild_testing_form(self, feature_names: tuple[str, ...]) -> None:
        """Rebuild the input rows to match the features of the last training run."""
        for child in self._testing_form.winfo_children():
            child.destroy()
        self._testing_inputs = {}
        self.interval_var.set("")
        # the old readout belongs to the previous model - clear it so a stale
        # number is never shown next to the new run's inputs
        self.prediction_name_var.set("")
        self.prediction_var.set("—")
        for feature in feature_names:
            row = ttk.Frame(self._testing_form)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=feature, style="Small.TLabel", width=14,
                      anchor="w").pack(side="left")
            var = tk.StringVar(value="0")
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            self._testing_inputs[feature] = var
        self.predict_button.configure(state="normal")

    def _on_predict(self) -> None:
        if self._last_model is None or not self._testing_inputs:
            return
        try:
            values = [float(var.get()) for var in self._testing_inputs.values()]
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Every field must be a number to compute a prediction.")
            return
        row = np.array([values])
        prediction = float(self._last_model.predict(row).ravel()[0])
        label = self._last_result.request.label_column if self._last_result else "y"
        self.prediction_name_var.set(f"Predicted {humanize_column(label)}")
        self.prediction_var.set(format_prediction(prediction))
        self._update_prediction_intervals(np.array(values, dtype=float))
        input_summary = ", ".join(f"{name}={var.get()}" for name, var in self._testing_inputs.items())
        self.log(f"Prediction: {input_summary} -> {label}={prediction:.6f}")
        # the calibration chart is pinned to these very inputs in that mode, so
        # it would otherwise keep showing the previous row's line
        if self.calibration_baseline_var.get() == CALIBRATION_AT_INPUTS:
            self._render_charts()

    def _update_prediction_intervals(self, x0: np.ndarray) -> None:
        """Show the 95% confidence and prediction intervals at x0, or explain
        why they are unavailable (same OLS-only restriction as the Inference
        tab: the formulas assume the exact least-squares solution)."""
        result = self._last_result
        if result is None or result.request.method != "ols":
            self.interval_var.set("Requires the OLS method.")
            return
        fit = LinearFit(weights=np.asarray(result.weights, dtype=float), bias=float(result.bias))
        try:
            interval: PredictionInterval = predict_with_intervals(
                result.x_train, result.y_true, fit, x0,
            )
        except ValueError as exc:
            self.interval_var.set(f"Unavailable: {exc}")
            return
        self.interval_var.set(
            f"CI (mean): [{interval.confidence_lower:.4f}, {interval.confidence_upper:.4f}]\n"
            f"PI (new obs): [{interval.prediction_lower:.4f}, {interval.prediction_upper:.4f}]"
        )

    # ------------------------------------------------------------------ #
    # Hyperparameters + logs
    # ------------------------------------------------------------------ #
    def _build_config_panel(self, parent) -> None:
        palette, fonts = self.theme.palette, self.theme.fonts
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        config_tab = ttk.Frame(notebook)
        logs_tab = ttk.Frame(notebook)
        notebook.add(config_tab, text="Config")
        notebook.add(logs_tab, text="Logs")

        body = ttk.Frame(config_tab)
        body.pack(fill="both", expand=True, padx=6, pady=6)

        # --- Method selector (top of the panel) ---
        self._label_with_hint(body, "Estimation method", HELP_METHOD).pack(anchor="w")
        self.method_var = tk.StringVar(value="gd")
        method_row = ttk.Frame(body)
        method_row.pack(anchor="w", fill="x", pady=(2, 8))
        ttk.Radiobutton(method_row, text="Gradient descent", variable=self.method_var,
                        value="gd", command=self._on_method_changed).pack(side="left")
        ttk.Radiobutton(method_row, text="OLS", variable=self.method_var,
                        value="ols", command=self._on_method_changed).pack(side="left",
                                                                             padx=(10, 0))

        ttk.Separator(body, orient="horizontal").pack(fill="x", pady=(0, 4))

        # --- GD hyperparameters (disabled when OLS is active) ---
        # Each _*_field returns (label_row_widget, active_widgets_list); we
        # collect them so _on_method_changed can enable/disable in one loop.
        self._gd_only_widgets: list[tk.Widget] = []

        # --- GLM family. Only gradient descent can fit one: the closed-form
        # OLS path solves the normal equations, which is normal/identity by
        # construction, so this is disabled alongside the other GD controls.
        self._label_with_hint(body, "GLM family", HELP_FAMILY).pack(anchor="w")
        self.family_var = tk.StringVar(value=FAMILY_LABELS[FAMILY_GAUSSIAN])
        self.family_combo = ttk.Combobox(
            body, textvariable=self.family_var, state="readonly",
            values=[FAMILY_LABELS[FAMILY_GAUSSIAN], FAMILY_LABELS[FAMILY_GAMMA_LOG]],
        )
        self.family_combo.pack(anchor="w", fill="x", pady=(2, 2))
        self.family_combo.bind("<<ComboboxSelected>>",
                               lambda _event: self._on_family_changed())
        self._gd_only_widgets.append(self.family_combo)
        self.family_note_var = tk.StringVar(value="")
        ttk.Label(body, textvariable=self.family_note_var, style="Small.TLabel",
                  justify="left", wraplength=CONFIG_PANEL_WIDTH - 24).pack(anchor="w", pady=(0, 6))

        self.learning_rate_var = tk.StringVar(value=str(RAW_LEARNING_RATE))
        self.epochs_var = tk.StringVar(value="10000")
        self.batch_size_var = tk.StringVar(value="100")
        self.tolerance_var = tk.StringVar(value="1e-4")
        self.standardize_var = tk.BooleanVar(value=False)

        self._gd_only_widgets += self._slider_field(
            body, "Learning Rate", self.learning_rate_var, 0.00001, 0.5,
            tooltip=HELP_LEARNING_RATE)
        self._gd_only_widgets += self._slider_field(
            body, "Epochs", self.epochs_var, 10, 50000, is_int=True,
            tooltip=HELP_EPOCHS)
        self._gd_only_widgets += self._slider_field(
            body, "Batch Size", self.batch_size_var, 1, 500, is_int=True,
            tooltip=HELP_BATCH_SIZE)
        self._gd_only_widgets += self._entry_field(
            body, "Stop Tolerance", self.tolerance_var, tooltip=HELP_TOLERANCE)

        ttk.Separator(body, orient="horizontal").pack(fill="x", pady=10)
        standardize_row = ttk.Frame(body)
        standardize_row.pack(anchor="w", fill="x")
        self._standardize_checkbutton = ttk.Checkbutton(
            standardize_row, text="Standardize features (Z-score)",
            variable=self.standardize_var, command=self._on_standardize_toggled,
        )
        self._standardize_checkbutton.pack(side="left")
        HelpHint(standardize_row, self.theme, HELP_STANDARDIZE).pack(side="left", padx=(4, 0))
        self._gd_only_widgets.append(self._standardize_checkbutton)
        ttk.Label(
            body,
            text=("Z-scoring rescales the problem,\nso the learning rate is reset\n"
                  "to a value suited to it."),
            style="Small.TLabel", justify="left",
        ).pack(anchor="w", pady=(4, 0))

        self.logs_text = tk.Text(logs_tab, wrap="word", height=10, state="disabled",
                                  bg=palette.cell_bg, fg=palette.text_fg, font=fonts.mono)
        logs_scroll = ttk.Scrollbar(logs_tab, orient="vertical", command=self.logs_text.yview)
        self.logs_text.configure(yscrollcommand=logs_scroll.set)
        self.logs_text.pack(side="left", fill="both", expand=True)
        logs_scroll.pack(side="right", fill="y")

    def _slider_field(self, parent, text: str, var: tk.StringVar, lo: float, hi: float,
                      is_int: bool = False, tooltip: str = "") -> list[tk.Widget]:
        """Returns the slider + entry widgets, so the caller can toggle their
        enabled state (used when switching to OLS, which has no hyperparams)."""
        if tooltip:
            self._label_with_hint(parent, text, tooltip).pack(anchor="w", pady=(8, 1))
        else:
            ttk.Label(parent, text=text, style="Small.TLabel").pack(anchor="w", pady=(8, 1))
        slider = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal")
        slider.pack(fill="x")
        entry = ttk.Entry(parent, textvariable=var)
        entry.pack(fill="x", pady=(2, 0))

        def on_slider(value) -> None:
            if self._syncing_slider:
                return
            self._syncing_slider = True
            numeric = float(value)
            var.set(str(int(numeric)) if is_int else f"{numeric:.5f}")
            self._syncing_slider = False

        def on_entry(_event=None) -> None:
            if self._syncing_slider:
                return
            try:
                numeric = float(var.get())
            except ValueError:
                return
            self._syncing_slider = True
            slider.set(min(max(numeric, lo), hi))
            self._syncing_slider = False

        slider.configure(command=on_slider)
        entry.bind("<Return>", on_entry)
        entry.bind("<FocusOut>", on_entry)
        on_entry()
        return [slider, entry]

    def _on_standardize_toggled(self) -> None:
        """Reset the learning rate to the default that suits the new scale."""
        enabled = bool(self.standardize_var.get())
        recommended = STANDARDIZED_LEARNING_RATE if enabled else RAW_LEARNING_RATE
        self.learning_rate_var.set(str(recommended))
        self.log(
            f"Standardization {'enabled' if enabled else 'disabled'}; "
            f"learning rate reset to {recommended}"
        )

    def _entry_field(self, parent, text: str, var: tk.StringVar, tooltip: str = "") -> list[tk.Widget]:
        if tooltip:
            self._label_with_hint(parent, text, tooltip).pack(anchor="w", pady=(8, 1))
        else:
            ttk.Label(parent, text=text, style="Small.TLabel").pack(anchor="w", pady=(8, 1))
        entry = ttk.Entry(parent, textvariable=var)
        entry.pack(fill="x")
        return [entry]

    def _on_method_changed(self) -> None:
        """Enable/disable the GD-only widgets to reflect the current method."""
        is_gd = self.method_var.get() == "gd"
        for widget in self._gd_only_widgets:
            try:
                widget.configure(state="normal" if is_gd else "disabled")
            except tk.TclError:
                pass
        # readonly, not normal: this is a fixed list of families, not free text
        if is_gd:
            self.family_combo.configure(state="readonly")
        self._on_family_changed(log=False)
        self.log(f"Estimation method: {'gradient descent' if is_gd else 'ordinary least squares'}")

    def _on_family_changed(self, log: bool = True) -> None:
        """Explain what the current family changes, and keep the note honest
        about OLS ignoring it."""
        family = self._selected_family()
        if self.method_var.get() != "gd":
            self.family_note_var.set("OLS always fits normal / identity.")
            return
        if family == FAMILY_GAMMA_LOG:
            self.family_note_var.set(
                "Requires a strictly positive target. Predictions are exp(...), "
                "so they are always positive; metrics switch to deviance-based ones."
            )
        else:
            self.family_note_var.set("Least squares: predictions may be negative.")
        if log:
            self.log(f"GLM family: {family_label(family)}")

    def log(self, message: str) -> None:
        self.logs_text.configure(state="normal")
        self.logs_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.logs_text.see(tk.END)
        self.logs_text.configure(state="disabled")

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _selected_family(self) -> tuple[str, str]:
        """The family picked in the dropdown, as a (loss_function, link) key."""
        for family, label in FAMILY_LABELS.items():
            if self.family_var.get() == label:
                return family
        return FAMILY_GAUSSIAN

    def _parse_hyperparameters(self) -> Hyperparameters | None:
        family = self._selected_family()
        try:
            return Hyperparameters(
                learning_rate=float(self.learning_rate_var.get()),
                batch_size=int(float(self.batch_size_var.get())),
                epochs=int(float(self.epochs_var.get())),
                tolerance=float(self.tolerance_var.get()),
                standardize_features=bool(self.standardize_var.get()),
                loss_function=family[0],
                loss_function_link=family[1],
            )
        except ValueError:
            messagebox.showerror("Invalid hyperparameters", "Check the values in the Config tab.")
            return None

    def _build_request(self) -> TrainingRequest | None:
        """Validate the current selections into a TrainingRequest, or explain
        what is missing and return None."""
        if self.current_csv_path is None:
            messagebox.showwarning("No CSV selected", "Choose a CSV file before training.")
            return None
        label_col = self.label_var.get()
        if not label_col:
            messagebox.showwarning("No label column", "Choose the target (label) column.")
            return None
        features = self._get_selected_features()
        if not features:
            messagebox.showwarning("No feature columns", "Choose at least one feature column.")
            return None
        hyperparameters = self._parse_hyperparameters()
        if hyperparameters is None:
            return None
        return TrainingRequest(
            csv_path=str(self.current_csv_path),
            label_column=label_col,
            features=tuple(features),
            method=self.method_var.get(),
            hyperparameters=hyperparameters,
            separator=self._current_separator,
        )

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self.start_button.configure(state="normal" if enabled else "disabled")
        self.stop_button.configure(state="disabled" if enabled else "normal")
        self.csv_combo.configure(state="readonly" if enabled else "disabled")
        self.browse_button.configure(state="normal" if enabled else "disabled")
        self.label_combo.configure(state="readonly" if enabled else "disabled")
        self.feature_dropdown.configure_state(enabled)

    def _start_training(self) -> None:
        request = self._build_request()
        if request is None:
            return

        self._set_inputs_enabled(False)
        self.save_button.configure(state="disabled")
        self.predict_button.configure(state="disabled")
        self.status_var.set("Training...")
        method_label = "gradient descent" if request.method == "gd" else "OLS (closed form)"
        self.log(f"Training ({method_label}): label={request.label_column}, "
                 f"features={list(request.features)}")
        if request.method == "gd":
            hp = request.hyperparameters
            self.log(
                f"lr={hp.learning_rate}, batch={hp.batch_size}, epochs={hp.epochs}, "
                f"tol={hp.tolerance}, standardize={hp.standardize_features}"
            )
        self.status_bar.set_text(
            f"Training ({method_label}): {request.label_column} ~ "
            f"{' + '.join(request.features)} ..."
        )
        self._training_start_time = time.monotonic()
        self._tick_stopwatch()

        self.controller.start(request, self._on_training_finished, self._on_training_failed)

    def _on_stop_training(self) -> None:
        if not self.controller.is_running:
            return
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopping...")
        self.status_bar.set_text("Stopping training...")
        self.log("Stop requested by user.")
        self.controller.stop()

    def _tick_stopwatch(self) -> None:
        self.stopwatch_var.set(format_elapsed(time.monotonic() - self._training_start_time))
        if self.controller.is_running:
            self.after(50, self._tick_stopwatch)

    def _on_training_failed(self, error: Exception) -> None:
        self._set_inputs_enabled(True)
        if isinstance(error, TrainingCancelled):
            self.status_var.set("Stopped")
            self.log("Training stopped by user.")
            self.status_bar.set_text("Training stopped - ready for a new run")
            return
        self.status_var.set("Failed")
        self.log(f"ERROR: {error}")
        self.status_bar.set_text("Training failed - see the Logs tab")
        messagebox.showerror("Training error", str(error))

    def _on_training_finished(self, result: TrainingResult) -> None:
        self._set_inputs_enabled(True)
        self._last_result = result
        self._last_model = self._rebuild_predict_only_model(result)
        self.save_button.configure(state="normal")

        family = self._result_family(result)
        rows = metrics_rows(result.y_true, result.y_pred,
                            n_features=len(result.request.features),
                            loss_function=family[0], loss_function_link=family[1])
        self.metrics_tree.delete(*self.metrics_tree.get_children())
        for name, value in rows:
            self.metrics_tree.insert("", tk.END, values=(name, f"{value:.5f}"))

        self._populate_inference_and_collinearity(result)
        self._rebuild_testing_form(result.request.features)
        self._rebuild_calibration_choices(result.request.features)
        self._render_charts()

        # Adapt status text to what the method actually produced: GD reports
        # epochs and a final loss; OLS just landed on the exact minimum.
        if result.request.method == "gd" and result.epochs_run is not None:
            self.status_var.set(f"Done - {result.epochs_run} epochs")
            self.status_bar.set_text(
                f"Training complete - {result.epochs_run} epochs, "
                f"final MSE {result.final_loss:.5f}"
            )
            self.log(f"Finished in {result.epochs_run} epochs, "
                     f"final MSE={result.final_loss:.6f}")
        else:
            self.status_var.set("Done - closed form")
            self.status_bar.set_text("Training complete - OLS closed-form solution")
            self.log(f"OLS finished in {result.elapsed_seconds*1000:.1f} ms")
        self.log(f"weights={result.weights}, bias={result.bias:.6f}")

    def _rebuild_predict_only_model(self, result: TrainingResult) -> Model:
        """Rebuild a Model whose predict() reproduces this run's fit.

        The scaling comes from the training run: GD may have Z-scored, OLS
        never does. We recover it from `result.x_train` (raw features) so
        predict() reapplies the same transform.
        """
        scaling = training_scaling(result)
        feature_mean, feature_std = scaling if scaling is not None else (None, None)
        return Model(
            csv_file=result.request.csv_path,
            label_column=result.request.label_column,
            features=list(result.request.features),
            x_train=result.x_train,
            y_train=result.y_true,
            weights=np.asarray(result.weights, dtype=float),
            bias=float(result.bias),
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

    def _populate_inference_and_collinearity(self, result: TrainingResult) -> None:
        """Fill the Inference and Predictors tabs.

        Inference (SE / t / p / F) is only defined for the exact least-squares
        solution, so it is skipped for GD runs - the tab just displays a
        message pointing the user to OLS. Coefficients and VIF are defined for
        both methods, so the Predictors tab is always populated.
        """
        self.inference_tree.delete(*self.inference_tree.get_children())
        self.predictors_tree.delete(*self.predictors_tree.get_children())

        x_raw = np.asarray(result.x_train, dtype=float)
        y = np.asarray(result.y_true, dtype=float)
        feature_names = list(result.request.features)

        if result.request.method != "ols":
            self.f_stat_var.set(INFERENCE_GD_MESSAGE)
        else:
            try:
                ols_fit = fit_ols(x_raw, y)
                summary = summarize_inference(x_raw, y, ols_fit,
                                              feature_names=feature_names)
            except ValueError as exc:
                self.f_stat_var.set(f"Inference unavailable: {exc}")
                self.log(f"Inference: {exc}")
            else:
                self._populate_inference_table(summary)

        try:
            columns = {name: x_raw[:, i] for i, name in enumerate(feature_names)}
            vifs = variance_inflation_factors(columns) if len(feature_names) > 1 else None
        except ValueError as exc:
            self.log(f"VIF: {exc}")
            vifs = None
        self._populate_predictors_table(result, vifs)

    def _populate_inference_table(self, summary: InferenceSummary) -> None:
        for name, coef, se, t_val, p_val in zip(
            summary.names, summary.coefficients, summary.standard_errors,
            summary.t_statistics, summary.p_values,
        ):
            self.inference_tree.insert(
                "", tk.END,
                values=(name, f"{coef:.5f}", f"{se:.5f}", f"{t_val:.4f}",
                        _format_p_value(p_val)),
            )
        # F-statistic uses the same OLS predictions the summary is built on,
        # reconstructed from summary.coefficients (intercept + weights).
        result = self._last_result
        y_pred_ols = _predict_with_ols_summary(summary, result.x_train)
        try:
            f = f_statistic(result.y_true, y_pred_ols,
                            n_features=len(result.request.features))
            self.f_stat_var.set(
                f"F-statistic = {f:.2f}  |  df = {summary.degrees_of_freedom}"
            )
        except ValueError as exc:
            self.f_stat_var.set(f"F-statistic unavailable: {exc}")

    def _populate_predictors_table(self, result: TrainingResult, vifs) -> None:
        """One row per predictor: coefficient, standardized coefficient, VIF.

        VIF is undefined for a single predictor (there is nothing to be
        collinear with) and may be missing if its computation failed; both
        show as "-" rather than dropping the row, so the coefficient is still
        visible.
        """
        feature_names = list(result.request.features)
        x_raw = np.asarray(result.x_train, dtype=float)
        coefficients = coefficients_in_original_space(result)
        spreads = x_raw.std(axis=0) if x_raw.size else np.ones(len(feature_names))

        standardized = (result.request.method == "gd"
                        and result.request.hyperparameters.standardize_features)
        units_note = ("converted back to original units" if standardized
                      else "in original units")

        # Under a log link the coefficients are additive on log(y), which means
        # multiplicative on y. Reporting a one-SD change as a FACTOR keeps the
        # column both scale-comparable and readable in the way the model works.
        is_log_link = self._result_family(result) == FAMILY_GAMMA_LOG
        if is_log_link:
            self.predictors_tree.heading("coef", text="Coef (log)")
            self.predictors_tree.heading("std_coef", text="x per SD")
            self.predictors_note_var.set(
                f"Log link: coefficients {units_note}, additive on log({result.request.label_column})."
            )
        else:
            self.predictors_tree.heading("coef", text="Coefficient")
            self.predictors_tree.heading("std_coef", text="Std. coef")
            self.predictors_note_var.set(f"Coefficients {units_note}.")

        for index, name in enumerate(feature_names):
            coefficient = float(coefficients[index])
            per_spread = coefficient * float(spreads[index])
            vif_value = vifs.get(name) if vifs else None
            self.predictors_tree.insert(
                "", tk.END,
                values=(name, f"{coefficient:.5f}",
                        f"{np.exp(per_spread):.5f}" if is_log_link else f"{per_spread:.5f}",
                        "-" if vif_value is None else f"{vif_value:.3f}"),
            )
        # the intercept completes the fitted equation but is not a predictor:
        # it has no spread to scale by and nothing to be collinear with
        self.predictors_tree.insert(
            "", tk.END,
            values=("(intercept)", f"{intercept_in_original_space(result):.5f}", "-", "-"),
        )

    # ------------------------------------------------------------------ #
    # Saving results
    # ------------------------------------------------------------------ #
    def _save_results(self) -> None:
        if self._last_result is None:
            return
        default_name = f"training_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        path = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            title="Save training report",
        )
        if not path:
            return
        saved = save_results_report(self._last_result, path)
        self.log(f"Report saved to {saved}")
        self.status_bar.set_text(f"Report saved to {saved}")


def _format_p_value(p: float) -> str:
    """Render a p-value the way ISLR tables do: tiny values become '< 1e-4'."""
    if p < 1e-4:
        return "< 0.0001"
    return f"{p:.4f}"


def _predict_with_ols_summary(summary: InferenceSummary, X: np.ndarray) -> np.ndarray:
    """Reconstruct y_pred from an InferenceSummary. The first coefficient is
    the intercept, the rest are the feature weights."""
    intercept = summary.coefficients[0]
    weights = summary.coefficients[1:]
    return intercept + np.asarray(X, dtype=float) @ weights


if __name__ == "__main__":
    TrainingApp().mainloop()
