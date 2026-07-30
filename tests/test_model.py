import csv

import numpy as np
import pandas as pd
import pytest

from kai.model import Model


def _write_csv(path, rows, header=("x", "y")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


# Test cases for get_data
def test_get_data_reads_single_feature(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv", [(1, 10), (2, 20), (3, 30)])
    model = Model(csv_path, "y")
    X, y = model.get_data("y", ["x"])
    pd.testing.assert_frame_equal(X, pd.DataFrame({"x": [1, 2, 3]}))
    pd.testing.assert_series_equal(y, pd.Series([10, 20, 30], name="y"))


def test_get_data_reads_multiple_features(tmp_path):
    csv_path = _write_csv(
        tmp_path / "data.csv",
        [(1, 10, 100), (2, 20, 200)],
        header=("x1", "x2", "y"),
    )
    model = Model(csv_path, "y")
    X, y = model.get_data("y", ["x1", "x2"])
    pd.testing.assert_frame_equal(X, pd.DataFrame({"x1": [1, 2], "x2": [10, 20]}))
    pd.testing.assert_series_equal(y, pd.Series([100, 200], name="y"))


def test_get_data_missing_column_raises(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv", [(1, 10)])
    model = Model(csv_path, "y")
    with pytest.raises(KeyError):
        model.get_data("y", ["does_not_exist"])


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
def test_predict_single_feature_batch():
    model = Model("unused.csv", "y")
    model._weight = np.array([2.0])
    model._bias = 5.0
    result = model.predict(np.array([[0.0], [1.0], [2.0]]))
    assert list(result) == [5.0, 7.0, 9.0]


def test_predict_single_feature_one_sample():
    model = Model("unused.csv", "y")
    model._weight = np.array([2.0])
    model._bias = 5.0
    assert model.predict(np.array([10.0])) == 25.0


def test_predict_multiple_features_batch():
    model = Model("unused.csv", "y")
    model._weight = np.array([2.0, 3.0])
    model._bias = 1.0
    result = model.predict(np.array([[1.0, 1.0], [2.0, 1.0]]))
    # [1 + 2*1 + 3*1, 1 + 2*2 + 3*1] = [6, 8]
    assert list(result) == [6.0, 8.0]


def test_predict_multiple_features_one_sample():
    model = Model("unused.csv", "y")
    model._weight = np.array([2.0, 3.0])
    model._bias = 1.0
    assert model.predict(np.array([1.0, 1.0])) == 6.0


# Test cases for start_training
def test_start_training_converges_on_single_feature_linear_data(tmp_path):
    rows = [(x, 2 * x + 5) for x in range(20)]
    csv_path = _write_csv(tmp_path / "data.csv", rows)
    model = Model(csv_path, "y")
    model.start_training(["x"], learning_rate=0.001, show_plot=False)
    assert model.weight == pytest.approx([2.0], abs=0.05)
    assert model.bias == pytest.approx(5.0, abs=0.5)
    assert model.loss_history[-1] < model.loss_history[0]


def test_start_training_converges_on_multiple_features(tmp_path):
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 20, 60)
    x2 = rng.uniform(0, 20, 60)
    y = 2 * x1 + 3 * x2 + 5
    rows = list(zip(x1, x2, y))
    csv_path = _write_csv(tmp_path / "data.csv", rows, header=("x1", "x2", "y"))
    model = Model(csv_path, "y")
    # batch_size (default) >= n_samples => full-batch gradient descent, which is
    # deterministic (the epoch shuffle doesn't change a full-batch sum), unlike
    # mini-batch SGD whose unseeded shuffling makes the trajectory non-reproducible
    model.start_training(["x1", "x2"], learning_rate=0.0015, epochs=50_000, show_plot=False)
    assert model.weight == pytest.approx([2.0, 3.0], abs=0.05)
    assert model.bias == pytest.approx(5.0, abs=0.5)


def test_start_training_with_standardization_converges(tmp_path):
    # features on wildly different scales: raw gradient descent struggles here,
    # standardization is what makes a sane learning rate work
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 1000, 80)
    x2 = rng.uniform(0, 1, 80)
    y = 0.5 * x1 + 20 * x2 + 3
    rows = list(zip(x1, x2, y))
    csv_path = _write_csv(tmp_path / "data.csv", rows, header=("x1", "x2", "y"))
    model = Model(csv_path, "y")
    model.start_training(["x1", "x2"], learning_rate=0.1, epochs=5000, show_plot=False,
                         standardize_features=True)
    # predictions must be accurate in the ORIGINAL feature space
    predictions = model.predict(model.x_train)
    assert predictions == pytest.approx(model.y_train, abs=0.5)


def test_predict_applies_training_standardization_to_new_data(tmp_path):
    rows = [(x, 2 * x + 5) for x in range(20)]
    csv_path = _write_csv(tmp_path / "data.csv", rows)
    model = Model(csv_path, "y")
    model.start_training(["x"], learning_rate=0.1, epochs=5000, show_plot=False,
                         standardize_features=True)
    # a raw (unstandardized) input must still map to the right prediction
    assert model.predict(np.array([[10.0]])) == pytest.approx([25.0], abs=0.2)


def test_start_training_without_standardization_clears_previous_scaling(tmp_path):
    rows = [(x, 2 * x + 5) for x in range(20)]
    csv_path = _write_csv(tmp_path / "data.csv", rows)
    model = Model(csv_path, "y")
    model.start_training(["x"], learning_rate=0.1, epochs=2000, show_plot=False,
                         standardize_features=True)
    assert model._feature_mean is not None

    model.start_training(["x"], learning_rate=0.001, epochs=2000, show_plot=False,
                         standardize_features=False)
    assert model._feature_mean is None
    assert model.predict(np.array([[10.0]])) == pytest.approx([25.0], abs=0.5)


def test_start_training_resets_loss_history_between_runs(tmp_path):
    rows = [(x, 2 * x + 5) for x in range(20)]
    csv_path = _write_csv(tmp_path / "data.csv", rows)
    model = Model(csv_path, "y")
    model.start_training(["x"], learning_rate=0.001, show_plot=False)
    first_run_length = len(model.loss_history)
    model.start_training(["x"], learning_rate=0.001, show_plot=False)
    assert len(model.loss_history) == first_run_length


def test_start_training_stops_at_epochs(tmp_path):
    rows = [(x, 2 * x + 5) for x in range(20)]
    csv_path = _write_csv(tmp_path / "data.csv", rows)
    model = Model(csv_path, "y")
    model.start_training(["x"], learning_rate=0.001, epochs=3, show_plot=False)
    assert len(model.loss_history) == 3
