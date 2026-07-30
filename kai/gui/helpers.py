"""Pure helpers used by the GUI: no Tkinter, fully unit-testable."""
from pathlib import Path

import pandas as pd

PREVIEW_ROWS = 10


def list_csv_files(directory) -> list[str]:
    """List the .csv file names directly inside `directory`, sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.glob("*.csv"))


def read_csv_preview(csv_path, n_rows: int = PREVIEW_ROWS) -> pd.DataFrame:
    """Read only the first `n_rows` of a CSV, for the preview grid."""
    return pd.read_csv(csv_path, nrows=n_rows)


def format_elapsed(seconds: float) -> str:
    """Format a duration as MM:SS:CC (centiseconds), clamping negatives to 0."""
    minutes, remainder = divmod(max(seconds, 0.0), 60)
    whole_seconds = int(remainder)
    centiseconds = int(round((remainder - whole_seconds) * 100))
    if centiseconds == 100:
        whole_seconds += 1
        centiseconds = 0
    return f"{int(minutes):02d}:{whole_seconds:02d}:{centiseconds:02d}"
