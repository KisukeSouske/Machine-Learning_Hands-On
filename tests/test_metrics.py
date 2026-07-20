import numpy as np
import pytest

from kai.metrics import calculate_loss


def test_calculate_loss_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert calculate_loss(y_true, y_pred) == 0


def test_calculate_loss_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 0.0, 5.0])
    # |1-2| + |2-0| + |3-5| = 1 + 2 + 2 = 5
    assert calculate_loss(y_true, y_pred) == 5


def test_calculate_loss_negative_values():
    y_true = np.array([-1.0, -2.0, -3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    # |-1-1| + |-2-2| + |-3-3| = 2 + 4 + 6 = 12
    assert calculate_loss(y_true, y_pred) == 12


def test_calculate_loss_single_value():
    y_true = np.array([10.0])
    y_pred = np.array([7.0])
    assert calculate_loss(y_true, y_pred) == 3


def test_calculate_loss_empty_arrays():
    y_true = np.array([])
    y_pred = np.array([])
    assert calculate_loss(y_true, y_pred) == 0


def test_calculate_loss_shape_mismatch_raises():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        calculate_loss(y_true, y_pred)
