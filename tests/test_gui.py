import csv

from kai.gui import format_elapsed, list_csv_files, read_csv_preview


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
