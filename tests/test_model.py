"""Tests for the new Model API: prediction object built by two factories,
one per estimation method.
"""
import csv

import numpy as np
import pytest

from kai.model import Model, TrainedModel


def _write_csv(path, rows, header=("x", "y")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def _multi_feature_csv(tmp_path):
    """y = 2*x1 + 3*x2 + 5, no noise, both predictors on the same scale."""
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 20, 60)
    x2 = rng.uniform(0, 20, 60)
    y = 2 * x1 + 3 * x2 + 5
    return _write_csv(tmp_path / "data.csv", list(zip(x1, x2, y)), header=("x1", "x2", "y"))


# ---------------- data loading ----------------
def test_load_columns_reads_features_as_2d_array(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv", [(1, 10), (2, 20), (3, 30)])
    X, y = Model.load_columns(csv_path, "y", ["x"])
    assert X.shape == (3, 1)
    assert X[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert y.tolist() == [10.0, 20.0, 30.0]


def test_load_columns_missing_column_raises(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv", [(1, 10)])
    with pytest.raises(KeyError):
        Model.load_columns(csv_path, "y", ["does_not_exist"])


# ---------------- predict (state-only tests, no factory) ----------------
def _bare_model(weights, bias, mean=None, std=None) -> Model:
    """Build a Model directly, bypassing the factories, for predict-only tests."""
    return Model(
        csv_file="unused.csv", label_column="y", features=["x"] * len(weights),
        x_train=np.zeros((1, len(weights))), y_train=np.zeros(1),
        weights=np.asarray(weights, dtype=float), bias=bias,
        feature_mean=mean, feature_std=std,
    )


def test_predict_single_feature_batch():
    model = _bare_model([2.0], 5.0)
    assert model.predict(np.array([[0.0], [1.0], [2.0]])).tolist() == [5.0, 7.0, 9.0]


def test_predict_multiple_features_batch():
    model = _bare_model([2.0, 3.0], 1.0)
    result = model.predict(np.array([[1.0, 1.0], [2.0, 1.0]]))
    # [1 + 2*1 + 3*1, 1 + 2*2 + 3*1] = [6, 8]
    assert result.tolist() == [6.0, 8.0]


def test_predict_reapplies_training_standardization():
    # a model whose weights live in standardized space still predicts correctly
    # from raw inputs, because predict() reapplies the training mean/std
    model = _bare_model([1.0], 0.0, mean=np.array([5.0]), std=np.array([2.0]))
    assert model.predict(np.array([[9.0]])) == pytest.approx([2.0])  # (9-5)/2 = 2, * 1 + 0


# ---------------- read-only properties ----------------
def test_weight_bias_have_no_public_setter():
    model = _bare_model([2.0], 5.0)
    with pytest.raises(AttributeError):
        model.weight = np.array([1.0])
    with pytest.raises(AttributeError):
        model.bias = 0.0


def test_train_arrays_are_returned_as_copies():
    model = _bare_model([1.0], 0.0)
    returned = model.x_train
    returned.fill(999.0)
    assert model.x_train.tolist() == [[0.0]]


# ---------------- fit_gradient_descent factory ----------------
def test_fit_gradient_descent_returns_trained_model_with_loss_history(tmp_path):
    csv_path = _multi_feature_csv(tmp_path)
    trained = Model.fit_gradient_descent(csv_path, "y", ["x1", "x2"],
                                          learning_rate=0.1, epochs=1000,
                                          standardize_features=True, random_state=0)
    assert isinstance(trained, TrainedModel)
    assert trained.method == "gd"
    assert trained.loss_history is not None
    assert trained.loss_history[-1] < trained.loss_history[0]


def test_fit_gradient_descent_recovers_coefficients_on_noiseless_data(tmp_path):
    csv_path = _multi_feature_csv(tmp_path)
    trained = Model.fit_gradient_descent(csv_path, "y", ["x1", "x2"],
                                          learning_rate=0.1, epochs=50_000,
                                          standardize_features=True, random_state=0)
    model = trained.model
    # translate the standardized-space coefficients back to raw space
    weights_raw = model.weight / model.feature_std
    bias_raw = model.bias - np.sum(model.weight * model.feature_mean / model.feature_std)
    assert weights_raw == pytest.approx([2.0, 3.0], abs=0.01)
    assert bias_raw == pytest.approx(5.0, abs=0.05)
    assert len(trained.loss_history) < 500


def test_fit_gradient_descent_without_standardization_leaves_scaling_none(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv", [(x, 2 * x + 5) for x in range(20)])
    trained = Model.fit_gradient_descent(csv_path, "y", ["x"],
                                          learning_rate=0.001, epochs=2000)
    assert trained.model.feature_mean is None
    assert trained.model.feature_std is None


def test_fit_gradient_descent_predict_matches_raw_targets(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv", [(x, 2 * x + 5) for x in range(20)])
    trained = Model.fit_gradient_descent(csv_path, "y", ["x"],
                                          learning_rate=0.1, epochs=5000,
                                          standardize_features=True, random_state=0)
    assert trained.model.predict(np.array([[10.0]])) == pytest.approx([25.0], abs=0.2)


# ---------------- fit_ols factory ----------------
def test_fit_ols_returns_trained_model_without_loss_history(tmp_path):
    csv_path = _multi_feature_csv(tmp_path)
    trained = Model.fit_ols(csv_path, "y", ["x1", "x2"])
    assert isinstance(trained, TrainedModel)
    assert trained.method == "ols"
    assert trained.loss_history is None


def test_fit_ols_recovers_exact_coefficients_on_noiseless_data(tmp_path):
    csv_path = _multi_feature_csv(tmp_path)
    trained = Model.fit_ols(csv_path, "y", ["x1", "x2"])
    assert trained.model.weight == pytest.approx([2.0, 3.0], abs=1e-10)
    assert trained.model.bias == pytest.approx(5.0, abs=1e-10)


def test_fit_ols_does_not_standardize_features(tmp_path):
    """OLS is affine-invariant, so it never touches the training scaling."""
    csv_path = _multi_feature_csv(tmp_path)
    trained = Model.fit_ols(csv_path, "y", ["x1", "x2"])
    assert trained.model.feature_mean is None
    assert trained.model.feature_std is None


def test_fit_ols_predicts_from_raw_features(tmp_path):
    csv_path = _multi_feature_csv(tmp_path)
    trained = Model.fit_ols(csv_path, "y", ["x1", "x2"])
    # y = 2*x1 + 3*x2 + 5; try x1=10, x2=4 -> 20 + 12 + 5 = 37
    assert trained.model.predict(np.array([[10.0, 4.0]])) == pytest.approx([37.0], abs=1e-9)


# ---------------- both methods agree on well-conditioned problems ----------------
def test_gd_and_ols_agree_on_predictions_up_to_convergence_tolerance(tmp_path):
    csv_path = _multi_feature_csv(tmp_path)
    ols = Model.fit_ols(csv_path, "y", ["x1", "x2"])
    gd = Model.fit_gradient_descent(csv_path, "y", ["x1", "x2"],
                                     learning_rate=0.1, epochs=50_000,
                                     standardize_features=True,
                                     tolerance=1e-8, random_state=0)
    same_rows = np.array([[7.0, 3.0], [1.5, 10.0]])
    assert gd.model.predict(same_rows) == pytest.approx(ols.model.predict(same_rows), abs=1e-3)
