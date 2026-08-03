import csv

from kai.gui import (
    count_csv_data_rows,
    detect_csv_separator,
    format_elapsed,
    format_prediction,
    humanize_column,
    list_csv_files,
    read_csv_preview,
)


def _write_csv(path, rows, header=("x", "y")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def test_list_csv_files_finds_only_csv_files(tmp_path):
    _write_csv(tmp_path / "b_data.csv", [(1, 2)])
    _write_csv(tmp_path / "a_data.csv", [(1, 2)])
    (tmp_path / "notes.txt").write_text("not a csv")

    assert list_csv_files(tmp_path) == ["a_data.csv", "b_data.csv"]


def test_list_csv_files_empty_directory_returns_empty_list(tmp_path):
    assert list_csv_files(tmp_path) == []


def test_list_csv_files_nonexistent_directory_returns_empty_list(tmp_path):
    assert list_csv_files(tmp_path / "does_not_exist") == []


def test_read_csv_preview_limits_to_n_rows(tmp_path):
    rows = [(i, i * 2) for i in range(30)]
    path = _write_csv(tmp_path / "data.csv", rows)

    preview = read_csv_preview(path, n_rows=10)

    assert len(preview) == 10
    assert list(preview.columns) == ["x", "y"]
    assert preview["x"].tolist() == list(range(10))


def test_read_csv_preview_handles_fewer_rows_than_requested(tmp_path):
    rows = [(1, 2), (3, 4)]
    path = _write_csv(tmp_path / "data.csv", rows)

    preview = read_csv_preview(path, n_rows=10)

    assert len(preview) == 2


# Test cases for detect_csv_separator
def test_detect_csv_separator_comma(tmp_path):
    path = _write_csv(tmp_path / "data.csv", [(1, 2), (3, 4)])
    assert detect_csv_separator(path) == ","


def test_detect_csv_separator_semicolon(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x;y\n1;2\n3;4\n", encoding="utf-8")
    assert detect_csv_separator(path) == ";"


def test_detect_csv_separator_tab(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x\ty\n1\t2\n3\t4\n", encoding="utf-8")
    assert detect_csv_separator(path) == "\t"


def test_detect_csv_separator_empty_file_falls_back_to_comma(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("", encoding="utf-8")
    assert detect_csv_separator(path) == ","


def test_detect_csv_separator_single_column_falls_back_to_comma(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n2\n3\n", encoding="utf-8")
    assert detect_csv_separator(path) == ","


# Test cases for count_csv_data_rows
def test_count_csv_data_rows_excludes_header(tmp_path):
    rows = [(i, i * 2) for i in range(5)]
    path = _write_csv(tmp_path / "data.csv", rows)
    assert count_csv_data_rows(path) == 5


def test_count_csv_data_rows_header_only(tmp_path):
    path = _write_csv(tmp_path / "data.csv", [])
    assert count_csv_data_rows(path) == 0


def test_count_csv_data_rows_empty_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("", encoding="utf-8")
    assert count_csv_data_rows(path) == 0


# Test cases for format_elapsed
def test_format_elapsed_zero():
    assert format_elapsed(0) == "00:00:00"


def test_format_elapsed_sub_minute():
    assert format_elapsed(2.45) == "00:02:45"


def test_format_elapsed_over_a_minute():
    assert format_elapsed(65.5) == "01:05:50"


def test_format_elapsed_rounds_centiseconds_without_overflowing():
    # 3.999s must not render as "00:03:100"
    assert format_elapsed(3.999) == "00:04:00"


def test_format_elapsed_negative_clamps_to_zero():
    assert format_elapsed(-1.0) == "00:00:00"


# Test cases for humanize_column
def test_humanize_column_replaces_separators_and_lifts_first_letter():
    assert humanize_column("taxa_oxidacao") == "Taxa oxidacao"
    assert humanize_column("learning-rate") == "Learning rate"


def test_humanize_column_preserves_inner_capitalisation():
    # "O" is oxygen here, not a stray capital: str.capitalize() would eat it
    assert humanize_column("concentracao_O") == "Concentracao O"


def test_humanize_column_survives_degenerate_names():
    assert humanize_column("") == ""
    assert humanize_column("___") == "___"


# Test cases for format_prediction
def test_format_prediction_uses_fixed_notation_when_it_fits():
    assert format_prediction(218.0) == "218.0000"
    assert format_prediction(-12345.6789) == "-12,345.6789"


def test_format_prediction_falls_back_to_scientific_when_too_wide():
    # a plain :.4f here would be far wider than the readout and get clipped
    assert format_prediction(1.23e12) == "1.2300e+12"


def test_format_prediction_does_not_flatten_small_values_to_zero():
    # 4.5e-9 must not read as an exact "0.0000"
    assert format_prediction(4.5e-9) == "4.5000e-09"
    assert format_prediction(0.0) == "0.0000"


def test_format_prediction_stays_within_the_readout_budget():
    values = [218.0, -12345.6789, 1.23e12, 4.5e-9, -9.87e-15, 0.0]
    assert all(len(format_prediction(v)) <= 13 for v in values)
