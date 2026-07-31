"""The main window: layout and event wiring only.

Visual decisions live in `kai.themes`, widget drawing in `kai.gui.widgets`,
training/persistence in `kai.gui.controller`, and typed session state in
`kai.gui.state` - this module just composes them.
"""
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from kai.gui.controller import TrainingController, save_results_report
from kai.gui.helpers import format_elapsed, list_csv_files, read_csv_preview
from kai.gui.state import Hyperparameters, TrainingRequest, TrainingResult
from kai.gui.widgets import ChartTabBar, MultiSelectDropdown, PanelHeader, StatusBar, ThemedButton
from kai.themes import Theme, get_theme
from kai.visualization import (
    CHART_LABELS,
    CHART_LOSS,
    CHART_PREDICTED_VS_ACTUAL,
    CHART_RESIDUALS,
    build_charts_figure,
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
BOTTOM_ROW_HEIGHT = 210
STATUS_PANEL_WIDTH = 240


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
        self._all_columns: list[str] = []
        self._preview_headers: dict[str, tk.Label] = {}
        self._preview_cells: dict[str, list[tk.Label]] = {}
        self._training_start_time = 0.0
        self._syncing_slider = False
        self._last_result: TrainingResult | None = None

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

        bottom_row = ttk.Frame(center, height=BOTTOM_ROW_HEIGHT)
        bottom_row.pack(side="bottom", fill="x", pady=(8, 0))
        bottom_row.pack_propagate(False)

        charts_panel = self._panel(center, "Model Training")
        charts_panel.pack(fill="both", expand=True)

        status_panel = self._panel(bottom_row, "Training Status")
        status_panel.pack(side="left", fill="both")
        status_panel.configure(width=STATUS_PANEL_WIDTH)
        status_panel.pack_propagate(False)
        metrics_panel = self._panel(bottom_row, "Training Metrics")
        metrics_panel.pack(side="right", fill="both", expand=True, padx=(8, 0))

        self._build_data_panel(data_panel.body)
        self._build_config_panel(config_panel.body)
        self._build_charts_panel(charts_panel.body)
        self._build_status_panel(status_panel.body)
        self._build_metrics_panel(metrics_panel.body)

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

    # ------------------------------------------------------------------ #
    # Data selection & preview
    # ------------------------------------------------------------------ #
    def _build_data_panel(self, parent) -> None:
        palette = self.theme.palette
        ttk.Label(parent, text="Select CSV File", style="Field.TLabel").pack(anchor="w")
        self.csv_var = tk.StringVar()
        self.csv_combo = ttk.Combobox(parent, textvariable=self.csv_var, state="readonly")
        self.csv_combo.pack(fill="x", pady=(2, 10))
        self.csv_combo.bind("<Button-1>", lambda _e: self._refresh_csv_list())
        self.csv_combo.bind("<<ComboboxSelected>>", self._on_csv_selected)
        self._refresh_csv_list()

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
        path = self.csv_dir / filename
        try:
            preview_df = read_csv_preview(path)
        except Exception as exc:
            messagebox.showerror("Failed to read CSV", str(exc))
            return

        self.current_csv_path = path
        self._all_columns = list(preview_df.columns)
        self._build_preview(preview_df)

        self.label_combo["values"] = self._all_columns
        self.label_var.set("")
        self.label_combo.configure(state="readonly")
        self.feature_dropdown.configure_state(True)
        self.feature_dropdown.set_selected([])
        self._refresh_feature_options()
        self._render_chips()
        self._recolor_preview()
        self.log(f"Loaded {filename} ({len(self._all_columns)} columns)")
        self.status_bar.set_text(f"{filename} loaded - select the label and feature columns")

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
    # Charts / metrics / status
    # ------------------------------------------------------------------ #
    def _build_charts_panel(self, parent) -> None:
        self.chart_tabs = ChartTabBar(
            parent,
            self.theme,
            [(key, CHART_LABELS[key]) for key in (CHART_LOSS, CHART_RESIDUALS, CHART_PREDICTED_VS_ACTUAL)],
            active_keys=[CHART_LOSS],
            on_change=self._render_charts,
        )
        self.chart_tabs.pack(fill="x", pady=(0, 6))

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

    def _render_charts(self) -> None:
        if self._last_result is None:
            return
        result = self._last_result
        self.figure.clear()
        build_charts_figure(
            self.figure, list(result.loss_history), result.y_true, result.y_pred,
            charts=self._selected_charts(), style=self.theme.charts,
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

        tree_row = ttk.Frame(parent)
        tree_row.pack(side="top", fill="both", expand=True)
        self.metrics_tree = ttk.Treeview(tree_row, columns=("metric", "value"), show="headings", height=6)
        self.metrics_tree.heading("metric", text="Metric")
        self.metrics_tree.heading("value", text="Value")
        self.metrics_tree.column("metric", anchor="w", width=180)
        self.metrics_tree.column("value", anchor="e", width=120)
        scrollbar = ttk.Scrollbar(tree_row, orient="vertical", command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=scrollbar.set)
        self.metrics_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_status_panel(self, parent) -> None:
        self.stopwatch_var = tk.StringVar(value=format_elapsed(0))
        ttk.Label(parent, textvariable=self.stopwatch_var, style="Stopwatch.TLabel",
                  anchor="center").pack(fill="x", pady=(4, 2))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.status_var, anchor="center").pack(fill="x")
        self.start_button = ThemedButton(parent, "START TRAINING", self.theme,
                                          command=self._start_training, height=32)
        self.start_button.pack(fill="x", pady=(8, 4))
        self.save_button = ThemedButton(parent, "SAVE RESULTS", self.theme,
                                         command=self._save_results, height=28)
        self.save_button.configure(state="disabled")
        self.save_button.pack(fill="x")

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

        self.learning_rate_var = tk.StringVar(value=str(RAW_LEARNING_RATE))
        self.epochs_var = tk.StringVar(value="10000")
        self.batch_size_var = tk.StringVar(value="100")
        self.tolerance_var = tk.StringVar(value="1e-4")
        self.standardize_var = tk.BooleanVar(value=False)

        self._slider_field(body, "Learning Rate", self.learning_rate_var, 0.00001, 0.5)
        self._slider_field(body, "Epochs", self.epochs_var, 10, 50000, is_int=True)
        self._slider_field(body, "Batch Size", self.batch_size_var, 1, 500, is_int=True)
        self._entry_field(body, "Stop Tolerance", self.tolerance_var)

        ttk.Separator(body, orient="horizontal").pack(fill="x", pady=10)
        ttk.Checkbutton(
            body, text="Standardize\nfeatures (Z-score)", variable=self.standardize_var,
            command=self._on_standardize_toggled,
        ).pack(anchor="w")
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
                      is_int: bool = False) -> None:
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

    def _on_standardize_toggled(self) -> None:
        """Reset the learning rate to the default that suits the new scale."""
        enabled = bool(self.standardize_var.get())
        recommended = STANDARDIZED_LEARNING_RATE if enabled else RAW_LEARNING_RATE
        self.learning_rate_var.set(str(recommended))
        self.log(
            f"Standardization {'enabled' if enabled else 'disabled'}; "
            f"learning rate reset to {recommended}"
        )

    def _entry_field(self, parent, text: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=text, style="Small.TLabel").pack(anchor="w", pady=(8, 1))
        ttk.Entry(parent, textvariable=var).pack(fill="x")

    def log(self, message: str) -> None:
        self.logs_text.configure(state="normal")
        self.logs_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.logs_text.see(tk.END)
        self.logs_text.configure(state="disabled")

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _parse_hyperparameters(self) -> Hyperparameters | None:
        try:
            return Hyperparameters(
                learning_rate=float(self.learning_rate_var.get()),
                batch_size=int(float(self.batch_size_var.get())),
                epochs=int(float(self.epochs_var.get())),
                tolerance=float(self.tolerance_var.get()),
                standardize_features=bool(self.standardize_var.get()),
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
            hyperparameters=hyperparameters,
        )

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self.start_button.configure(state="normal" if enabled else "disabled")
        self.csv_combo.configure(state="readonly" if enabled else "disabled")
        self.label_combo.configure(state="readonly" if enabled else "disabled")
        self.feature_dropdown.configure_state(enabled)

    def _start_training(self) -> None:
        request = self._build_request()
        if request is None:
            return
        hp = request.hyperparameters

        self._set_inputs_enabled(False)
        self.save_button.configure(state="disabled")
        self.status_var.set("Training...")
        self.log(f"Training started: label={request.label_column}, features={list(request.features)}")
        self.log(
            f"lr={hp.learning_rate}, batch={hp.batch_size}, epochs={hp.epochs}, "
            f"tol={hp.tolerance}, standardize={hp.standardize_features}"
        )
        self.status_bar.set_text(
            f"Training {request.label_column} ~ {' + '.join(request.features)} ..."
        )
        self._training_start_time = time.monotonic()
        self._tick_stopwatch()

        self.controller.start(request, self._on_training_finished, self._on_training_failed)

    def _tick_stopwatch(self) -> None:
        self.stopwatch_var.set(format_elapsed(time.monotonic() - self._training_start_time))
        if self.controller.is_running:
            self.after(50, self._tick_stopwatch)

    def _on_training_failed(self, error: Exception) -> None:
        self._set_inputs_enabled(True)
        self.status_var.set("Failed")
        self.log(f"ERROR: {error}")
        self.status_bar.set_text("Training failed - see the Logs tab")
        messagebox.showerror("Training error", str(error))

    def _on_training_finished(self, result: TrainingResult) -> None:
        self._set_inputs_enabled(True)
        self._last_result = result
        self.save_button.configure(state="normal")

        rows = metrics_rows(result.y_true, result.y_pred, n_features=len(result.request.features))
        self.metrics_tree.delete(*self.metrics_tree.get_children())
        for name, value in rows:
            self.metrics_tree.insert("", tk.END, values=(name, f"{value:.5f}"))

        self._render_charts()

        self.status_var.set(f"Done - {result.epochs_run} epochs")
        self.status_bar.set_text(
            f"Training complete - {result.epochs_run} epochs, final MSE {result.final_loss:.5f}"
        )
        self.log(f"Finished in {result.epochs_run} epochs, final MSE={result.final_loss:.6f}")
        self.log(f"weights={result.weights}, bias={result.bias:.6f}")

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


if __name__ == "__main__":
    TrainingApp().mainloop()
