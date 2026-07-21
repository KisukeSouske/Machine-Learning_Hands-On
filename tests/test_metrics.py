import numpy as np
import pytest

from kai.metrics import loss
from kai.metrics import mean_absolute_error
from kai.metrics import squared_loss
from kai.metrics import mean_squared_error

# Test cases for loss function
def test_loss_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert loss(y_true, y_pred) == 0


def test_loss_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 0.0, 5.0])
    # |1-2| + |2-0| + |3-5| = 1 + 2 + 2 = 5
    assert loss(y_true, y_pred) == 5


def test_loss_negative_values():
    y_true = np.array([-1.0, -2.0, -3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    # |-1-1| + |-2-2| + |-3-3| = 2 + 4 + 6 = 12
    assert loss(y_true, y_pred) == 12


def test_loss_single_value():
    y_true = np.array([10.0])
    y_pred = np.array([7.0])
    assert loss(y_true, y_pred) == 3


def test_loss_empty_arrays():
    y_true = np.array([])
    y_pred = np.array([])
    assert loss(y_true, y_pred) == 0


def test_loss_shape_mismatch_raises():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        loss(y_true, y_pred)

# Mean Absolute Error Tests
def test_mean_absolute_error_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert mean_absolute_error(y_true, y_pred) == 0

def test_mean_absolute_error_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 0.0, 5.0])
    # (|1-2| + |2-0| + |3-5|) / 3 = (1 + 2 + 2) / 3 = 5 / 3
    assert mean_absolute_error(y_true, y_pred) == 5/3

def test_mean_absolute_error_negative_values():
    y_true = np.array([-1.0, -2.0, -3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    # (|-1-1| + |-2-2| + |-3-3|) / 3 = (2 + 4 + 6) / 3 = 12 / 3 = 4
    assert mean_absolute_error(y_true, y_pred) == 4

def test_mean_absolute_error_shape_mismatch_raises():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        mean_absolute_error(y_true, y_pred)

# Test cases for squared_loss function
def test_squared_loss_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert squared_loss(y_true, y_pred) == 0

def test_squared_loss_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 0.0, 5.0])
    # (1-2)^2 + (2-0)^2 + (3-5)^2 = 1 + 4 + 4 = 9
    assert squared_loss(y_true, y_pred) == 9

def test_squared_loss_negative_values():
    y_true = np.array([-1.0, -2.0, -3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    # (-1-1)^2 + (-2-2)^2 + (-3-3)^2 = 4 + 16 + 36 = 56
    assert squared_loss(y_true, y_pred) == 56

def test_squared_loss_empty_arrays():
    y_true = np.array([])
    y_pred = np.array([])
    assert squared_loss(y_true, y_pred) == 0

def test_squared_loss_shape_mismatch_raises():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        squared_loss(y_true, y_pred)

# Test cases for mean_squared_error function
def test_mean_squared_error_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert mean_squared_error(y_true, y_pred) == 0

def test_mean_squared_error_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 0.0, 5.0])
    # (1 + 4 + 4) / 3 = 9 / 3 = 3
    assert mean_squared_error(y_true, y_pred) == 3

def test_mean_squared_error_negative_values():
    y_true = np.array([-1.0, -2.0, -3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    # (4 + 16 + 36) / 3 = 56 / 3
    assert mean_squared_error(y_true, y_pred) == 56 / 3

def test_mean_squared_error_shape_mismatch_raises():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        mean_squared_error(y_true, y_pred)