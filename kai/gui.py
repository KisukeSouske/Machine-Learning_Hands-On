import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd

from kai.model import Model
from kai.visualization import FIGURE_BACKGROUND, PANEL_BACKGROUND, build_dashboard_figure

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

LABEL_HIGHLIGHT = "#FFE0B2"
FEATURE_HIGHLIGHT = "#E1D5F7"
DEFAULT_CELL_BG = "#FFFFFF"
PREVIEW_ROWS = 10


def list_csv_files(directory) -> list[str]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.glob("*.csv"))


def read_csv_preview(csv_path, n_rows: int = PREVIEW_ROWS) -> pd.DataFrame:
    return pd.read_csv(csv_path, nrows=n_rows)


class TrainingApp(tk.Tk):
    def __init__(self, csv_dir=None):
        super().__init__()
        self.title("kai - Treinamento de Regressão Linear")
        self.geometry("1150x950")
        self.configure(bg=FIGURE_BACKGROUND)

        self.csv_dir = Path(csv_dir) if csv_dir is not None else Path(__file__).resolve().parent.parent
        self.current_csv_path = None
        self._all_columns: list[str] = []
        self._preview_headers: dict[str, tk.Label] = {}
        self._preview_cells: dict[str, list[tk.Label]] = {}
        self._training_thread_alive = False
        self._training_start_time = 0.0

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.training_tab = ttk.Frame(notebook)
        self.hyperparams_tab = ttk.Frame(notebook)
        notebook.add(self.training_tab, text="Treino")
        notebook.add(self.hyperparams_tab, text="Hiperparâmetros")

        self._build_training_tab(self.training_tab)
        self._build_hyperparams_tab(self.hyperparams_tab)

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_training_tab(self, parent) -> None:
        csv_row = ttk.Frame(parent)
        csv_row.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Label(csv_row, text="CSV:").pack(side="left")
        self.csv_var = tk.StringVar()
        self.csv_combo = ttk.Combobox(csv_row, textvariable=self.csv_var, state="readonly", width=45)
        self.csv_combo.pack(side="left", padx=(6, 0))
        self.csv_combo.bind("<Button-1>", lambda _e: self._refresh_csv_list())
        self.csv_combo.bind("<<ComboboxSelected>>", self._on_csv_selected)
        self._refresh_csv_list()

        preview_container = ttk.Frame(parent)
        preview_container.pack(fill="x", padx=10, pady=5)
        ttk.Label(preview_container, text=f"Prévia (primeiras {PREVIEW_ROWS} linhas):").pack(anchor="w")
        preview_scroll = tk.Canvas(preview_container, height=220, bg=DEFAULT_CELL_BG, highlightthickness=1,
                                    highlightbackground="#CCCCCC")
        preview_scroll.pack(fill="x", pady=(4, 0))
        hbar = ttk.Scrollbar(preview_container, orient="horizontal", command=preview_scroll.xview)
        hbar.pack(fill="x")
        preview_scroll.configure(xscrollcommand=hbar.set)
        self.preview_frame = tk.Frame(preview_scroll, bg=DEFAULT_CELL_BG)
        preview_scroll.create_window((0, 0), window=self.preview_frame, anchor="nw")
        self.preview_frame.bind(
            "<Configure>", lambda _e: preview_scroll.configure(scrollregion=preview_scroll.bbox("all"))
        )

        label_row = ttk.Frame(parent)
        label_row.pack(fill="x", padx=10, pady=5)
        ttk.Label(label_row, text="Coluna de label:").pack(side="left")
        self.label_var = tk.StringVar()
        self.label_combo = ttk.Combobox(label_row, textvariable=self.label_var, state="disabled", width=30)
        self.label_combo.pack(side="left", padx=(6, 0))
        self.label_combo.bind("<<ComboboxSelected>>", self._on_label_selected)

        features_row = ttk.Frame(parent)
        features_row.pack(fill="x", padx=10, pady=5)
        ttk.Label(features_row, text="Colunas de features:").pack(side="left", anchor="n")
        self.features_listbox = tk.Listbox(
            features_row, selectmode="multiple", height=5, exportselection=False, state="disabled"
        )
        self.features_listbox.pack(side="left", padx=(6, 0), fill="x", expand=True)
        self.features_listbox.bind("<<ListboxSelect>>", self._on_features_selected)

        controls_row = ttk.Frame(parent)
        controls_row.pack(fill="x", padx=10, pady=10)
        self.start_button = ttk.Button(controls_row, text="Start", command=self._start_training)
        self.start_button.pack(side="left")
        ttk.Label(controls_row, text="Tempo:").pack(side="left", padx=(20, 4))
        self.stopwatch_var = tk.StringVar(value="00:00.0")
        ttk.Label(controls_row, textvariable=self.stopwatch_var, font=("Segoe UI", 12, "bold")).pack(side="left")

        dashboard_container = ttk.Frame(parent)
        dashboard_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.figure = Figure(figsize=(10, 7))
        self.figure.patch.set_facecolor(FIGURE_BACKGROUND)
        self.canvas = FigureCanvasTkAgg(self.figure, master=dashboard_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

    def _build_hyperparams_tab(self, parent) -> None:
        self.learning_rate_var = tk.StringVar(value="0.001")
        self.batch_size_var = tk.StringVar(value="100")
        self.epochs_var = tk.StringVar(value="10000")
        self.tolerance_var = tk.StringVar(value="1e-6")

        fields = [
            ("Taxa de aprendizagem (learning_rate):", self.learning_rate_var),
            ("Tamanho do lote (batch_size):", self.batch_size_var),
            ("Épocas máximas (epochs):", self.epochs_var),
            ("Tolerância de parada (tolerance):", self.tolerance_var),
        ]
        for row_idx, (text, var) in enumerate(fields):
            row = ttk.Frame(parent)
            row.pack(fill="x", padx=20, pady=10)
            ttk.Label(row, text=text, width=32, anchor="w").pack(side="left")
            ttk.Entry(row, textvariable=var, width=15).pack(side="left")

    # ------------------------------------------------------------------ #
    # CSV / preview / column selection
    # ------------------------------------------------------------------ #
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
            messagebox.showerror("Erro ao ler CSV", str(exc))
            return

        self.current_csv_path = path
        self._all_columns = list(preview_df.columns)
        self._build_preview(preview_df)

        self.label_combo["values"] = self._all_columns
        self.label_var.set("")
        self.label_combo.configure(state="readonly")

        self.features_listbox.configure(state="normal")
        self._refresh_features_listbox()
        self._recolor_preview()

    def _build_preview(self, df: pd.DataFrame) -> None:
        for child in self.preview_frame.winfo_children():
            child.destroy()
        self._preview_headers = {}
        self._preview_cells = {}

        for col_idx, col in enumerate(df.columns):
            header = tk.Label(
                self.preview_frame, text=str(col), font=("Segoe UI", 9, "bold"),
                bg=PANEL_BACKGROUND, borderwidth=1, relief="solid", padx=8, pady=4,
            )
            header.grid(row=0, column=col_idx, sticky="nsew")
            self._preview_headers[col] = header
            self._preview_cells[col] = []

        for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
            for col_idx, col in enumerate(df.columns):
                cell = tk.Label(
                    self.preview_frame, text=str(row[col]), bg=DEFAULT_CELL_BG,
                    borderwidth=1, relief="solid", padx=8, pady=2,
                )
                cell.grid(row=row_idx, column=col_idx, sticky="nsew")
                self._preview_cells[col].append(cell)

    def _refresh_features_listbox(self) -> None:
        label_col = self.label_var.get()
        previously_selected = set(self._get_selected_features())
        self.features_listbox.delete(0, tk.END)
        for col in self._all_columns:
            if col == label_col:
                continue
            self.features_listbox.insert(tk.END, col)
        for i in range(self.features_listbox.size()):
            if self.features_listbox.get(i) in previously_selected:
                self.features_listbox.selection_set(i)

    def _get_selected_features(self) -> list[str]:
        return [self.features_listbox.get(i) for i in self.features_listbox.curselection()]

    def _on_label_selected(self, _event=None) -> None:
        self._refresh_features_listbox()
        self._recolor_preview()

    def _on_features_selected(self, _event=None) -> None:
        self._recolor_preview()

    def _recolor_preview(self) -> None:
        label_col = self.label_var.get()
        selected_features = set(self._get_selected_features())
        for col, header in self._preview_headers.items():
            if col == label_col:
                header_color = cell_color = LABEL_HIGHLIGHT
            elif col in selected_features:
                header_color = cell_color = FEATURE_HIGHLIGHT
            else:
                header_color, cell_color = PANEL_BACKGROUND, DEFAULT_CELL_BG
            header.configure(bg=header_color)
            for cell in self._preview_cells[col]:
                cell.configure(bg=cell_color)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _parse_hyperparameters(self):
        try:
            return {
                "learning_rate": float(self.learning_rate_var.get()),
                "batch_size": int(self.batch_size_var.get()),
                "epochs": int(self.epochs_var.get()),
                "tolerance": float(self.tolerance_var.get()),
            }
        except ValueError:
            messagebox.showerror(
                "Hiperparâmetros inválidos", "Verifique os valores na aba Hiperparâmetros."
            )
            return None

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self.start_button.configure(state="normal" if enabled else "disabled")
        self.csv_combo.configure(state="readonly" if enabled else "disabled")
        self.label_combo.configure(state="readonly" if enabled else "disabled")
        self.features_listbox.configure(state="normal" if enabled else "disabled")

    def _start_training(self) -> None:
        if self.current_csv_path is None:
            messagebox.showwarning("Selecione um CSV", "Escolha um arquivo CSV antes de iniciar.")
            return
        label_col = self.label_var.get()
        if not label_col:
            messagebox.showwarning("Selecione a coluna de label", "Escolha a coluna alvo (label).")
            return
        features = self._get_selected_features()
        if not features:
            messagebox.showwarning("Selecione as features", "Escolha ao menos uma coluna de feature.")
            return
        hyperparams = self._parse_hyperparameters()
        if hyperparams is None:
            return

        self._set_inputs_enabled(False)
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
                show_plot=False,
            )
        except Exception as exc:  # surfaced to the user via messagebox, not crashed silently
            error = exc
        self._training_thread_alive = False
        self.after(0, self._on_training_finished, model, features, error)

    def _tick_stopwatch(self) -> None:
        elapsed = time.monotonic() - self._training_start_time
        minutes, seconds = divmod(elapsed, 60)
        self.stopwatch_var.set(f"{int(minutes):02d}:{seconds:04.1f}")
        if self._training_thread_alive:
            self.after(100, self._tick_stopwatch)

    def _on_training_finished(self, model: Model, features: list[str], error: Exception | None) -> None:
        self._set_inputs_enabled(True)
        if error is not None:
            messagebox.showerror("Erro no treinamento", str(error))
            return

        y_pred = model.predict(model.x_train)
        self.figure.clear()
        build_dashboard_figure(self.figure, model.loss_history, model.y_train, y_pred, n_features=len(features))
        self.canvas.draw()


if __name__ == "__main__":
    TrainingApp().mainloop()
