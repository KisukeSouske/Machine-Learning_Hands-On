import csv

import numpy as np
import pytest

from kai.model import Model


def _write_csv(path, rows, header=("x", "y")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


# Test cases for get_data
def test_get_data_reads_rows(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv", [(1, 10), (2, 20), (3, 30)])
    model = Model(csv_path, "y")
    data = model.get_data("y", "x")
    assert data == [("1", "10"), ("2", "20"), ("3", "30")]


def test_get_data_missing_column_raises(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv", [(1, 10)])
    model = Model(csv_path, "y")
    with pytest.raises(KeyError):
        model.get_data("y", "does_not_exist")


# weight/bias/loss_history should be read-only from outside the class
def test_weight_bias_loss_history_have_no_public_setter():
    model = Model("unused.csv", "y")
    with pytest.raises(AttributeError):
        model.weight = 2.0
    with pytest.raises(AttributeError):
        model.bias = 5.0
    with pytest.raises(AttributeError):
        model.loss_history = [1.0]


def test_loss_history_property_returns_a_copy():
    model = Model("unused.csv", "y")
    model._loss_history = [1.0, 2.0]
    returned = model.loss_history
    returned.append(999.0)
    assert model.loss_history == [1.0, 2.0]


# Test cases for predict
def test_predict_scalar():
    model = Model("unused.csv", "y")
    model._weight = 2.0
    model._bias = 5.0
    assert model.predict(10) == 25.0


def test_predict_array():
    model = Model("unused.csv", "y")
    model._weight = 2.0
    model._bias = 5.0
    result = model.predict(np.array([0, 1, 2]))
    assert list(result) == [5.0, 7.0, 9.0]


# Test cases for start_training
def test_start_training_converges_on_linear_data(tmp_path):
    rows = [(x, 2 * x + 5) for x in range(20)]
    csv_path = _write_csv(tmp_path / "data.csv", rows)
    model = Model(csv_path, "y")
    model.start_training("x", learning_rate=0.001)
    assert model.weight == pytest.approx(2.0, abs=0.05)
    assert model.bias == pytest.approx(5.0, abs=0.5)
    assert model.loss_history[-1] < model.loss_history[0]


def test_start_training_resets_loss_history_between_runs(tmp_path):
    rows = [(x, 2 * x + 5) for x in range(20)]
    csv_path = _write_csv(tmp_path / "data.csv", rows)
    model = Model(csv_path, "y")
    model.start_training("x", learning_rate=0.001)
    first_run_length = len(model.loss_history)
    model.start_training("x", learning_rate=0.001)
    assert len(model.loss_history) == first_run_length


def test_start_training_stops_at_epochs(tmp_path):
    rows = [(x, 2 * x + 5) for x in range(20)]
    csv_path = _write_csv(tmp_path / "data.csv", rows)
    model = Model(csv_path, "y")
    model.start_training("x", learning_rate=0.001, epochs=3)
    assert len(model.loss_history) == 3
