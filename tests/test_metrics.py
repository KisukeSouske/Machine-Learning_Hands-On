import numpy as np
import pytest

from kai.metrics import loss
from kai.metrics import mean_absolute_error
from kai.metrics import squared_loss
from kai.metrics import mean_squared_error
from kai.metrics import mean_squared_error_derivation
from kai.metrics import r_squared
from kai.metrics import adjusted_r_squared

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

# Test cases for mean_squared_error_derivation function
def test_mean_squared_error_derivation_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    x_true = np.array([2.0, 1.0, 3.0])
    weight_derivation, bias_derivation = mean_squared_error_derivation(y_true, y_pred, x_true)
    assert weight_derivation == 0
    assert bias_derivation == 0

def test_mean_squared_error_derivation_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 0.0, 5.0])
    x_true = np.array([2.0, 1.0, 3.0])
    # (y_pred - y_true) = [1, -2, 2]
    # weight = sum([1, -2, 2] * 2 * [2, 1, 3]) / 3 = sum([4, -4, 12]) / 3 = 4
    # bias = sum([1, -2, 2] * 2) / 3 = sum([2, -4, 4]) / 3 = 2/3
    weight_derivation, bias_derivation = mean_squared_error_derivation(y_true, y_pred, x_true)
    assert weight_derivation == pytest.approx(4)
    assert bias_derivation == pytest.approx(2 / 3)

def test_mean_squared_error_derivation_y_shape_mismatch_raises():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    x_true = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        mean_squared_error_derivation(y_true, y_pred, x_true)

def test_mean_squared_error_derivation_x_shape_mismatch_raises():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    x_true = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        mean_squared_error_derivation(y_true, y_pred, x_true)

# Test cases for r_squared function
def test_r_squared_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y_true, y_pred) == pytest.approx(1.0)

def test_r_squared_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    # rss = 0.01 + 0.01 + 0.04 + 0.04 = 0.10
    # tss = sum((y_true - 2.5)^2) = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
    # r2 = 1 - 0.10/5.0 = 0.98
    assert r_squared(y_true, y_pred) == pytest.approx(0.98)

def test_r_squared_no_better_than_mean_is_zero():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert r_squared(y_true, y_pred) == pytest.approx(0.0)

def test_r_squared_worse_than_mean_is_negative():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([10.0, 10.0, 10.0, 10.0])
    assert r_squared(y_true, y_pred) < 0

def test_r_squared_shape_mismatch_raises():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        r_squared(y_true, y_pred)

# Test cases for adjusted_r_squared function
def test_adjusted_r_squared_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert adjusted_r_squared(y_true, y_pred, n_features=1) == pytest.approx(1.0)

def test_adjusted_r_squared_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    # r2 = 0.98, n = 4, n_features = 1
    # adj = 1 - (1 - 0.98) * (4 - 1) / (4 - 1 - 1) = 0.97
    assert adjusted_r_squared(y_true, y_pred, n_features=1) == pytest.approx(0.97)

def test_adjusted_r_squared_penalizes_extra_features():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    # adj(n_features=2) = 1 - 0.02 * 3 / 1 = 0.94
    assert adjusted_r_squared(y_true, y_pred, n_features=2) == pytest.approx(0.94)
    assert adjusted_r_squared(y_true, y_pred, n_features=2) < adjusted_r_squared(y_true, y_pred, n_features=1)

def test_adjusted_r_squared_shape_mismatch_raises():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        adjusted_r_squared(y_true, y_pred, n_features=1)