import numpy as np
import pytest

from kai.preprocessing import standardize


def test_standardize_returns_zero_mean_unit_std():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    X_scaled, mean, std = standardize(X)
    assert X_scaled.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-12)
    assert X_scaled.std(axis=0) == pytest.approx([1.0, 1.0])


def test_standardize_returns_correct_mean_and_std():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    _, mean, std = standardize(X)
    assert mean == pytest.approx([2.5, 25.0])
    assert std == pytest.approx([1.1180339887, 11.180339887])


def test_standardize_known_values():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    X_scaled, _, _ = standardize(X)
    expected = np.array(
        [
            [-1.34164079, -1.34164079],
            [-0.4472136, -0.4472136],
            [0.4472136, 0.4472136],
            [1.34164079, 1.34164079],
        ]
    )
    assert X_scaled == pytest.approx(expected)


def test_standardize_single_feature_1d():
    X = np.array([1.0, 2.0, 3.0, 4.0])
    X_scaled, mean, std = standardize(X)
    assert mean == pytest.approx(2.5)
    assert X_scaled.mean() == pytest.approx(0.0, abs=1e-12)
    assert X_scaled.std() == pytest.approx(1.0)


def test_standardize_constant_column_does_not_divide_by_zero():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    X_scaled, _, std = standardize(X)
    assert np.isfinite(X_scaled).all()
    assert list(X_scaled[:, 1]) == pytest.approx([0.0, 0.0, 0.0])
    assert std[1] == 1.0


def test_standardize_new_data_reuses_train_mean_and_std():
    X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    _, mean, std = standardize(X_train)
    X_new = np.array([2.5, 25.0])
    # the training mean maps to 0 in standardized space
    assert (X_new - mean) / std == pytest.approx([0.0, 0.0], abs=1e-12)
