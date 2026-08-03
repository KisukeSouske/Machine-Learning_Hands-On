"""Pure helpers used by the GUI: no Tkinter, fully unit-testable."""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

PREVIEW_ROWS = 10


def list_csv_files(directory) -> list[str]:
    """List the .csv file names directly inside `directory`, sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.glob("*.csv"))


def detect_csv_separator(csv_path) -> str:
    """Guess the delimiter of a CSV by reading a short sample and asking
    csv.Sniffer. Falls back to ',' if the file is too short or the sniffer
    cannot decide.

    Restricted to `,`, `;` and tab: these are the three separators actually
    seen in the wild for the kind of files kai reads. Broadening the set
    would just increase the chance of misdetecting something like `.` inside
    a numeric column.
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
    if not sample:
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def read_csv_preview(csv_path, n_rows: int = PREVIEW_ROWS, sep: str | None = None) -> pd.DataFrame:
    """Read only the first `n_rows` of a CSV, for the preview grid.

    `sep=None` asks pandas to sniff the delimiter itself (uses the python
    engine). Pass an explicit separator to skip the sniffing.
    """
    engine = "python" if sep is None else None
    return pd.read_csv(csv_path, nrows=n_rows, sep=sep, engine=engine)


def count_csv_data_rows(csv_path) -> int:
    """Return the number of data rows in a CSV (i.e. excluding the header),
    counted without loading the file into memory.
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        total_lines = sum(1 for _ in f)
    return max(total_lines - 1, 0)   # subtract the header; guard the empty case


def humanize_column(name: str) -> str:
    """Turn a raw CSV column identifier into ordinary prose for display.

    `taxa_oxidacao` reads as a code variable, not as a label; the readout in
    the testing panel shows the humanized form while every lookup elsewhere
    keeps using the original column name.
    """
    words = str(name).replace("_", " ").replace("-", " ").split()
    if not words:
        return str(name)
    text = " ".join(words)
    # only lift the first character: str.capitalize() would lowercase the rest
    # and turn a meaningful "concentracao_O" into "Concentracao o"
    return text[0].upper() + text[1:]


def format_prediction(value: float, max_chars: int = 13) -> str:
    """Format a predicted value so it always fits the fixed-width readout.

    The panel has a hard pixel budget, and a plain `:.4f` silently overflows
    it for very large or very small magnitudes - which clips the number out
    of view entirely. Falling back to scientific notation keeps the string
    bounded no matter the scale.
    """
    if not np.isfinite(value):
        return str(value)
    # a nonzero magnitude below the 4th decimal would render as a flat
    # "0.0000", which reads as an exact zero rather than a small number
    if value != 0.0 and abs(value) < 1e-4:
        return f"{value:.4e}"
    fixed = f"{value:,.4f}"
    if len(fixed) <= max_chars:
        return fixed
    return f"{value:.4e}"


def format_elapsed(seconds: float) -> str:
    """Format a duration as MM:SS:CC (centiseconds), clamping negatives to 0."""
    minutes, remainder = divmod(max(seconds, 0.0), 60)
    whole_seconds = int(remainder)
    centiseconds = int(round((remainder - whole_seconds) * 100))
    if centiseconds == 100:
        whole_seconds += 1
        centiseconds = 0
    return f"{int(minutes):02d}:{whole_seconds:02d}:{centiseconds:02d}"
