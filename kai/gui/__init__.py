"""kai.gui - the training application.

Public surface:
    TrainingApp(theme="default" | "retro_os" | Theme instance)
plus the pure helpers and the report builder, re-exported for tests and
programmatic use.
"""
from kai.gui.app import (
    BOTTOM_ROW_HEIGHT,
    CONFIG_PANEL_WIDTH,
    DATA_PANEL_WIDTH,
    PREVIEW_HEIGHT,
    STATUS_PANEL_WIDTH,
    TrainingApp,
)
from kai.gui.controller import TrainingController, build_results_report, save_results_report
from kai.gui.helpers import PREVIEW_ROWS, format_elapsed, list_csv_files, read_csv_preview
from kai.gui.state import Hyperparameters, TrainingRequest, TrainingResult

__all__ = [
    "TrainingApp",
    "TrainingController",
    "build_results_report",
    "save_results_report",
    "Hyperparameters",
    "TrainingRequest",
    "TrainingResult",
    "format_elapsed",
    "list_csv_files",
    "read_csv_preview",
    "PREVIEW_ROWS",
    "DATA_PANEL_WIDTH",
    "CONFIG_PANEL_WIDTH",
    "PREVIEW_HEIGHT",
    "BOTTOM_ROW_HEIGHT",
    "STATUS_PANEL_WIDTH",
]
