import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd

from kai import aero
from kai.aero import AeroButton, AeroTab, GradientHeader, StatusBar
from kai.model import Model
from kai.visualization import (
    CHART_LABELS,
    CHART_LOSS,
    CHART_PREDICTED_VS_ACTUAL,
    CHART_RESIDUALS,
    FIGURE_BACKGROUND,
    build_charts_figure,
    metrics_rows,
)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

APP_BACKGROUND = aero.WINDOW_BG
PANEL_HEADER_BG = aero.HEADER_BOTTOM
LABEL_HIGHLIGHT = "#FFE0B2"
FEATURE_HIGHLIGHT = "#E1D5F7"
DEFAULT_CELL_BG = aero.PANEL_BG
CHIP_BG = "#E1D5F7"
TAB_ACTIVE_BG = aero.TAB_ACTIVE_TOP
TAB_INACTIVE_BG = aero.TAB_INACTIVE_TOP
PREVIEW_ROWS = 10

DATA_PANEL_WIDTH = 430
CONFIG_PANEL_WIDTH = 231  # 165 * 1.4
PREVIEW_HEIGHT = 150
BOTTOM_ROW_HEIGHT = 210
STATUS_PANEL_WIDTH = 240


def list_csv_files(directory) -> list[str]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.glob("*.csv"))


def read_csv_preview(csv_path, n_rows: int = PREVIEW_ROWS) -> pd.DataFrame:
    return pd.read_csv(csv_path, nrows=n_rows)


def format_elapsed(seconds: float) -> str:
    minutes, remainder = divmod(max(seconds, 0.0), 60)
    whole_seconds = int(remainder)
    centiseconds = int(round((remainder - whole_seconds) * 100))
    if centiseconds == 100:
        whole_seconds += 1
        centiseconds = 0
    return f"{int(minutes):02d}:{whole_seconds:02d}:{centiseconds:02d}"


class MultiSelectDropdown(ttk.Frame):
    """A dropdown holding a multi-select Listbox: rows are toggled by clicking
    them directly (no checkboxes), and the popup stays open until an outside
    click, Escape, or Close - unlike ttk.Combobox, which collapses on every pick."""

    def __init__(self, parent, placeholder: str = "Select...", on_change=None):
        super().__init__(parent)
        self._option_names: list[str] = []
        self._selected: list[str] = []
        self._on_change = on_change
        self._placeholder = placeholder
        self._popup = None
        self._listbox = None
        self._enabled = False

        self._button = ttk.Button(self, text=f"{placeholder}  ▾", command=self.toggle)
        self._button.pack(fill="x")

    def configure_state(self, enabled: bool) -> None:
        self._enabled = enabled
        self._button.configure(state="normal" if enabled else "disabled")
        if not enabled:
            self.close()

    def set_options(self, options: list[str]) -> None:
        self._option_names = list(options)
        self._selected = [c for c in self._selected if c in self._option_names]
        self._refresh_button_text()
        if self.is_open():
            self.close()
            self.open()

    def get_selected(self) -> list[str]:
        return list(self._selected)

    def set_selected(self, values: list[str]) -> None:
        self._selected = [v for v in values if v in self._option_names]
        self._refresh_button_text()
        self._sync_listbox_selection()

    def is_open(self) -> bool:
        return self._popup is not None and self._popup.winfo_exists()

    def toggle(self) -> None:
        self.close() if self.is_open() else self.open()

    def open(self) -> None:
        if not self._enabled or self.is_open() or not self._option_names:
            return
        popup = tk.Toplevel(self)
        popup.overrideredirect(True)
        popup.configure(bg="#999999")
        x = self._button.winfo_rootx()
        y = self._button.winfo_rooty() + self._button.winfo_height()
        popup.geometry(f"{max(self._button.winfo_width(), 200)}x200+{x}+{y}")

        container = tk.Frame(popup, bg=DEFAULT_CELL_BG)
        container.pack(fill="both", expand=True, padx=1, pady=1)

        listbox = tk.Listbox(
            container, selectmode="multiple", exportselection=False,
            activestyle="none", highlightthickness=0, borderwidth=0,
            bg=DEFAULT_CELL_BG, selectbackground=FEATURE_HIGHLIGHT, selectforeground="#000000",
        )
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for option in self._option_names:
            listbox.insert(tk.END, option)
        listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self._listbox = listbox
        self._sync_listbox_selection()

        tk.Button(container, text="Close", relief="flat", bg=PANEL_HEADER_BG,
                  command=self.close).pack(side="bottom", fill="x")

        # grab_set routes every click to the popup; clicks landing on the
        # listbox are handled by it, anything else reaches this binding
        # and means "the user clicked outside", so we close.
        popup.bind("<Button-1>", self._maybe_close_on_outside_click)
        popup.bind("<Escape>", lambda _e: self.close())
        popup.grab_set()
        self._popup = popup

    def _sync_listbox_selection(self) -> None:
        if self._listbox is None or not self._listbox.winfo_exists():
            return
        self._listbox.selection_clear(0, tk.END)
        for index, option in enumerate(self._option_names):
            if option in self._selected:
                self._listbox.selection_set(index)

    def _on_listbox_select(self, _event=None) -> None:
        if self._listbox is None:
            return
        self._selected = [self._option_names[i] for i in self._listbox.curselection()]
        self._refresh_button_text()
        if self._on_change is not None:
            self._on_change()

    def _maybe_close_on_outside_click(self, event) -> None:
        popup = self._popup
        if popup is None:
            return
        inside = (
            popup.winfo_rootx() <= event.x_root <= popup.winfo_rootx() + popup.winfo_width()
            and popup.winfo_rooty() <= event.y_root <= popup.winfo_rooty() + popup.winfo_height()
        )
        if not inside:
            self.close()

    def _refresh_button_text(self) -> None:
        count = len(self._selected)
        text = self._placeholder if count == 0 else f"{count} column{'s' if count > 1 else ''} selected"
        self._button.configure(text=f"{text}  ▾")

    def close(self) -> None:
        if self.is_open():
            self._popup.grab_release()
            self._popup.destroy()
        self._popup = None
        self._listbox = None


class ChartTabBar(tk.Frame):
    """Tab-looking toggles: each tab turns blue when active, and any number of
    them can be active at once (unlike a real Notebook, which is exclusive)."""

    def __init__(self, parent, keys_and_labels, active_keys, on_change):
        super().__init__(parent, bg=DEFAULT_CELL_BG)
        self._on_change = on_change
        self._active = set(active_keys)
        self._tabs = {}
        for key, label in keys_and_labels:
            tab = AeroTab(self, label, on_click=lambda k=key: self._toggle(k))
            tab.pack(side="left", padx=(0, 2))
            self._tabs[key] = tab
        self._repaint()

    def _toggle(self, key: str) -> None:
        if key in self._active:
            self._active.remove(key)
        else:
            self._active.add(key)
        self._repaint()
        self._on_change()

    def _repaint(self) -> None:
        for key, tab in self._tabs.items():
            tab.set_active(key in self._active)

    def get_active(self) -> list[str]:
        return [key for key in self._tabs if key in self._active]


class TrainingApp(tk.Tk):
    def __init__(self, csv_dir=None):
        super().__init__()
        self.title("ML Model Trainer - Supervised Regression")
        self.geometry("1500x900")
        self.minsize(1200, 700)
        self.configure(bg=APP_BACKGROUND)

        self.csv_dir = Path(csv_dir) if csv_dir is not None else Path(__file__).resolve().parent.parent
        self.current_csv_path = None
        self._all_columns: list[str] = []
        self._preview_headers: dict[str, tk.Label] = {}
        self._preview_cells: dict[str, list[tk.Label]] = {}
        self._training_thread_alive = False
        self._training_start_time = 0.0
        self._syncing_slider = False
        self._last_result = None

        self._configure_styles()

        # fixed layout: data + hyperparameters keep a constant width on the
        # left, the training area takes whatever is left over
        self.status_bar = StatusBar(self)
        self.status_bar.pack(side="bottom", fill="x")
        self.status_bar.set_text("Ready for selection...")

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

        # mirrored vs. the previous layout: status on the left, metrics on the right
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

    # ------------------------------------------------------------------ #
    # Chrome
    # ------------------------------------------------------------------ #
    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        # the native Windows theme already draws Aero-style comboboxes,
        # entries, scrollbars and treeviews; the glossy pieces are hand-drawn
        for theme in ("vista", "xpnative", "winnative", "clam"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        style.configure("TFrame", background=DEFAULT_CELL_BG)
        style.configure("App.TFrame", background=APP_BACKGROUND)
        style.configure("TLabel", background=DEFAULT_CELL_BG, foreground=aero.HEADER_FG)
        style.configure("Field.TLabel", background=DEFAULT_CELL_BG, font=("Segoe UI", 9),
                        foreground=aero.HEADER_FG)
        style.configure("Small.TLabel", background=DEFAULT_CELL_BG, font=("Segoe UI", 8),
                        foreground=aero.HEADER_FG)
        style.configure("Stopwatch.TLabel", background=DEFAULT_CELL_BG, font=("Consolas", 26, "bold"),
                        foreground=aero.HEADER_FG)
        style.configure("TCheckbutton", background=DEFAULT_CELL_BG, foreground=aero.HEADER_FG)
        style.configure("Treeview", background="#FFFFFF", fieldbackground="#FFFFFF",
                        font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), foreground=aero.HEADER_FG)
        style.map("Treeview", background=[("selected", aero.SELECT_BG)],
                  foreground=[("selected", "#000000")])

    def _panel(self, parent, title: str) -> tk.Frame:
        panel = tk.Frame(parent, bg=aero.PANEL_BORDER, highlightthickness=0, bd=0)
        inner = tk.Frame(panel, bg=DEFAULT_CELL_BG)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        GradientHeader(inner, title).pack(fill="x")
        panel.body = ttk.Frame(inner)
        panel.body.pack(fill="both", expand=True, padx=8, pady=8)
        return panel

    # ------------------------------------------------------------------ #
    # Data selection & preview
    # ------------------------------------------------------------------ #
    def _build_data_panel(self, parent) -> None:
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
            preview_box, bg=DEFAULT_CELL_BG, highlightthickness=1, highlightbackground="#CCCCCC"
        )
        vbar = ttk.Scrollbar(preview_box, orient="vertical", command=self.preview_canvas.yview)
        hbar = ttk.Scrollbar(preview_box, orient="horizontal", command=self.preview_canvas.xview)
        self.preview_canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        preview_box.rowconfigure(0, weight=1)
        preview_box.columnconfigure(0, weight=1)

        self.preview_frame = tk.Frame(self.preview_canvas, bg=DEFAULT_CELL_BG)
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
            parent, placeholder="Select feature columns", on_change=self._on_features_changed
        )
        self.feature_dropdown.pack(fill="x", pady=2)

        self.chips_text = tk.Text(
            parent, height=3, wrap="char", bg=DEFAULT_CELL_BG, relief="solid",
            borderwidth=1, cursor="arrow", state="disabled",
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
        for child in self.preview_frame.winfo_children():
            child.destroy()
        self._preview_headers = {}
        self._preview_cells = {}

        tk.Label(
            self.preview_frame, text="#", font=("Segoe UI", 9, "bold"),
            bg=PANEL_HEADER_BG, borderwidth=1, relief="solid", padx=6, pady=4,
        ).grid(row=0, column=0, sticky="nsew")

        for col_idx, col in enumerate(df.columns, start=1):
            header = tk.Label(
                self.preview_frame, text=str(col), font=("Segoe UI", 9, "bold"),
                bg=PANEL_HEADER_BG, borderwidth=1, relief="solid", padx=8, pady=4,
            )
            header.grid(row=0, column=col_idx, sticky="nsew")
            self._preview_headers[col] = header
            self._preview_cells[col] = []

        for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
            tk.Label(
                self.preview_frame, text=str(row_idx), bg=PANEL_HEADER_BG,
                borderwidth=1, relief="solid", padx=6, pady=2,
            ).grid(row=row_idx, column=0, sticky="nsew")
            for col_idx, col in enumerate(df.columns, start=1):
                cell = tk.Label(
                    self.preview_frame, text=str(row[col]), bg=DEFAULT_CELL_BG,
                    borderwidth=1, relief="solid", padx=8, pady=2,
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
        self.chips_text.configure(state="normal")
        self.chips_text.delete("1.0", tk.END)
        for column in self._get_selected_features():
            chip = tk.Frame(self.chips_text, bg=CHIP_BG, padx=4, pady=1)
            tk.Label(chip, text=column, bg=CHIP_BG, font=("Segoe UI", 8)).pack(side="left")
            tk.Button(
                chip, text="×", bg=CHIP_BG, relief="flat", font=("Segoe UI", 8, "bold"),
                cursor="hand2", padx=2, pady=0, borderwidth=0,
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
        label_col = self.label_var.get()
        selected = set(self._get_selected_features())
        for col, header in self._preview_headers.items():
            if col == label_col:
                header_color = cell_color = LABEL_HIGHLIGHT
            elif col in selected:
                header_color = cell_color = FEATURE_HIGHLIGHT
            else:
                header_color, cell_color = PANEL_HEADER_BG, DEFAULT_CELL_BG
            header.configure(bg=header_color)
            for cell in self._preview_cells[col]:
                cell.configure(bg=cell_color)

    # ------------------------------------------------------------------ #
    # Charts / metrics / status
    # ------------------------------------------------------------------ #
    def _build_charts_panel(self, parent) -> None:
        self.chart_tabs = ChartTabBar(
            parent,
            [(key, CHART_LABELS[key]) for key in (CHART_LOSS, CHART_RESIDUALS, CHART_PREDICTED_VS_ACTUAL)],
            active_keys=[CHART_LOSS],
            on_change=self._render_charts,
        )
        self.chart_tabs.pack(fill="x", pady=(0, 6))

        self.figure = Figure(figsize=(9, 4))
        self.figure.patch.set_facecolor(FIGURE_BACKGROUND)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)

        # matplotlib's own toolbar: pan, rectangle zoom, back/forward, save
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side="bottom", fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x")

        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

    def _selected_charts(self) -> list[str]:
        return self.chart_tabs.get_active()

    def _render_charts(self) -> None:
        if self._last_result is None:
            return
        loss_history, y_true, y_pred = self._last_result
        self.figure.clear()
        build_charts_figure(self.figure, loss_history, y_true, y_pred, charts=self._selected_charts())
        self.canvas.draw()
        # reset the zoom/pan history so the toolbar's "home" matches the new axes
        self.toolbar.update()

    def _build_metrics_panel(self, parent) -> None:
        self.metrics_tree = ttk.Treeview(parent, columns=("metric", "value"), show="headings", height=6)
        self.metrics_tree.heading("metric", text="Metric")
        self.metrics_tree.heading("value", text="Value")
        self.metrics_tree.column("metric", anchor="w", width=180)
        self.metrics_tree.column("value", anchor="e", width=120)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=scrollbar.set)
        self.metrics_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_status_panel(self, parent) -> None:
        self.stopwatch_var = tk.StringVar(value=format_elapsed(0))
        ttk.Label(parent, textvariable=self.stopwatch_var, style="Stopwatch.TLabel", anchor="center").pack(
            fill="x", pady=(4, 2)
        )
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.status_var, anchor="center").pack(fill="x")
        self.start_button = AeroButton(parent, "START TRAINING", command=self._start_training, height=32)
        self.start_button.pack(fill="x", pady=8)

    # ------------------------------------------------------------------ #
    # Hyperparameters + logs
    # ------------------------------------------------------------------ #
    def _build_config_panel(self, parent) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        config_tab = ttk.Frame(notebook)
        logs_tab = ttk.Frame(notebook)
        notebook.add(config_tab, text="Config")
        notebook.add(logs_tab, text="Logs")

        body = ttk.Frame(config_tab)
        body.pack(fill="both", expand=True, padx=6, pady=6)

        self.learning_rate_var = tk.StringVar(value="0.0003")
        self.epochs_var = tk.StringVar(value="10000")
        self.batch_size_var = tk.StringVar(value="100")
        self.tolerance_var = tk.StringVar(value="1e-6")
        self.standardize_var = tk.BooleanVar(value=False)

        self._slider_field(body, "Learning Rate", self.learning_rate_var, 0.00001, 0.1)
        self._slider_field(body, "Epochs", self.epochs_var, 10, 50000, is_int=True)
        self._slider_field(body, "Batch Size", self.batch_size_var, 1, 500, is_int=True)
        self._entry_field(body, "Stop Tolerance", self.tolerance_var)

        ttk.Separator(body, orient="horizontal").pack(fill="x", pady=10)
        ttk.Checkbutton(
            body, text="Standardize\nfeatures (Z-score)", variable=self.standardize_var
        ).pack(anchor="w")

        self.logs_text = tk.Text(logs_tab, wrap="word", height=10, state="disabled",
                                  bg=DEFAULT_CELL_BG, font=("Consolas", 8))
        logs_scroll = ttk.Scrollbar(logs_tab, orient="vertical", command=self.logs_text.yview)
        self.logs_text.configure(yscrollcommand=logs_scroll.set)
        self.logs_text.pack(side="left", fill="both", expand=True)
        logs_scroll.pack(side="right", fill="y")

    def _slider_field(self, parent, text: str, var: tk.StringVar, lo: float, hi: float, is_int: bool = False) -> None:
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
    def _parse_hyperparameters(self):
        try:
            return {
                "learning_rate": float(self.learning_rate_var.get()),
                "batch_size": int(float(self.batch_size_var.get())),
                "epochs": int(float(self.epochs_var.get())),
                "tolerance": float(self.tolerance_var.get()),
                "standardize_features": bool(self.standardize_var.get()),
            }
        except ValueError:
            messagebox.showerror("Invalid hyperparameters", "Check the values in the Config tab.")
            return None

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self.start_button.configure(state="normal" if enabled else "disabled")
        self.csv_combo.configure(state="readonly" if enabled else "disabled")
        self.label_combo.configure(state="readonly" if enabled else "disabled")
        self.feature_dropdown.configure_state(enabled)

    def _start_training(self) -> None:
        if self.current_csv_path is None:
            messagebox.showwarning("No CSV selected", "Choose a CSV file before training.")
            return
        label_col = self.label_var.get()
        if not label_col:
            messagebox.showwarning("No label column", "Choose the target (label) column.")
            return
        features = self._get_selected_features()
        if not features:
            messagebox.showwarning("No feature columns", "Choose at least one feature column.")
            return
        hyperparams = self._parse_hyperparameters()
        if hyperparams is None:
            return

        self._set_inputs_enabled(False)
        self.status_var.set("Training...")
        self.log(f"Training started: label={label_col}, features={features}")
        self.log(
            f"lr={hyperparams['learning_rate']}, batch={hyperparams['batch_size']}, "
            f"epochs={hyperparams['epochs']}, tol={hyperparams['tolerance']}, "
            f"standardize={hyperparams['standardize_features']}"
        )
        self.status_bar.set_text(f"Training {label_col} ~ {' + '.join(features)} ...")
        self._training_start_time = time.monotonic()
        self._training_thread_alive = True
        self._tick_stopwatch()

        thread = threading.Thread(
            target=self._train_worker,
            args=(str(self.current_csv_path), label_col, features, hyperparams),
            daemon=True,
        )
        thread.start()

    def _train_worker(self, csv_path: str, label_col: str, features: list[str], hyperparams: dict) -> None:
        model = Model(csv_path, label_col)
        error = None
        try:
            model.start_training(
                features,
                learning_rate=hyperparams["learning_rate"],
                batch_size=hyperparams["batch_size"],
                epochs=hyperparams["epochs"],
                tolerance=hyperparams["tolerance"],
                standardize_features=hyperparams["standardize_features"],
                show_plot=False,
            )
        except Exception as exc:  # surfaced to the user via messagebox, not swallowed
            error = exc
        self._training_thread_alive = False
        self.after(0, self._on_training_finished, model, features, error)

    def _tick_stopwatch(self) -> None:
        self.stopwatch_var.set(format_elapsed(time.monotonic() - self._training_start_time))
        if self._training_thread_alive:
            self.after(50, self._tick_stopwatch)

    def _on_training_finished(self, model: Model, features: list[str], error: Exception | None) -> None:
        self._set_inputs_enabled(True)
        if error is not None:
            self.status_var.set("Failed")
            self.log(f"ERROR: {error}")
            self.status_bar.set_text("Training failed - see the Logs tab")
            messagebox.showerror("Training error", str(error))
            return

        y_pred = model.predict(model.x_train)
        rows = metrics_rows(model.y_train, y_pred, n_features=len(features))

        self.metrics_tree.delete(*self.metrics_tree.get_children())
        for name, value in rows:
            self.metrics_tree.insert("", tk.END, values=(name, f"{value:.5f}"))

        self._last_result = (model.loss_history, model.y_train, y_pred)
        self._render_charts()

        self.status_var.set(f"Done - {len(model.loss_history)} epochs")
        self.status_bar.set_text(
            f"Training complete - {len(model.loss_history)} epochs, "
            f"final MSE {model.loss_history[-1]:.5f}"
        )
        self.log(f"Finished in {len(model.loss_history)} epochs, final MSE={model.loss_history[-1]:.6f}")
        self.log(f"weights={model.weight}, bias={model.bias:.6f}")


if __name__ == "__main__":
    TrainingApp().mainloop()
