"""Tests for the t-test / standard-error machinery in kai.regression.

Ground truth is deliberately NOT self-authored. Three external sources:

1. UFMG-EST-027 Cap.11 "Testes de Hipoteses na Regressao Linear Simples",
   Exemplo 11.1/11.2 (oxygen purity data): published beta1_hat, Sxx, sigma2_hat,
   t0=11.41, P=1.13e-9 - numbers straight off the course slides, computed by
   the book's author with a calculator, not by this codebase.
2. ISLR (James, Witten, Hastie, Tibshirani) Table 3.1: published coefficient
   and standard error for the Advertising/TV regression, t=17.67.
3. statsmodels.OLS - an independent, widely used implementation - as the
   ground truth for end-to-end runs where the book only gives summary
   statistics rather than the raw dataset.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from kai.regression import (
    coefficient_standard_errors,
    fit_ols,
    p_values,
    residual_standard_error,
    summarize_inference,
    t_statistics,
)

PUBLICIDADE_CSV = Path(__file__).resolve().parent.parent / "publicidade.csv"


# --- 1. UFMG-EST-027 Cap.11, Exemplo 11.2 (oxygen purity data) ---
# beta1_hat=14.97, n=20, Sxx=0.68, sigma2_hat=1.17 => t0=11.41 (book), P~=1.13e-9 (book)
CAP11_BETA1 = 14.97
CAP11_SXX = 0.68
CAP11_SIGMA2 = 1.17
CAP11_N = 20
CAP11_SE = np.sqrt(CAP11_SIGMA2 / CAP11_SXX)


def test_t_statistics_matches_cap11_worked_example():
    t0 = t_statistics(np.array([CAP11_BETA1]), np.array([CAP11_SE]))
    assert t0[0] == pytest.approx(11.41, abs=0.005)


def test_p_values_matches_cap11_worked_example():
    t0 = CAP11_BETA1 / CAP11_SE
    df = CAP11_N - 2
    p = p_values(np.array([t0]), df)
    assert p[0] == pytest.approx(1.13e-9, rel=0.01)


# --- 2. ISLR Table 3.1 (Advertising ~ TV, simple regression) ---
# published: coef(TV)=0.0475, SE(TV)=0.0027, t=17.67, n=200
def test_t_statistics_matches_islr_table_3_1():
    # the book's own coefficient/SE are already rounded to 4 decimals before
    # printing, so recomputing the ratio only approximates the book's t
    # (0.0475/0.0027 = 17.59 vs the book's reported 17.67); the gap is the
    # book's rounding, not this implementation
    t = t_statistics(np.array([0.0475]), np.array([0.0027]))
    assert t[0] == pytest.approx(17.67, rel=0.01)


def test_p_values_matches_islr_table_3_1_reported_as_tiny():
    # the book reports "< 0.0001" rather than an exact figure
    t = 0.0475 / 0.0027
    p = p_values(np.array([t]), degrees_of_freedom=198)
    assert p[0] < 0.0001


# --- 3. coefficient_standard_errors reduces to the textbook's own simple-
#    regression formula (ISLR eq. 3.8) computed independently, not via kai ---
def test_standard_errors_reduce_to_islr_simple_regression_formula():
    rng = np.random.default_rng(20)
    x = rng.uniform(0, 10, 40)
    y = 3.0 * x + 2.0 + rng.normal(0, 1.5, 40)

    fit = fit_ols(x, y)
    se = coefficient_standard_errors(x, y, fit.predict(x))

    # ISLR eq. 3.8, computed from scratch (not calling any kai function)
    x_bar = x.mean()
    sxx = np.sum((x - x_bar) ** 2)
    rss = np.sum((y - fit.predict(x)) ** 2)
    sigma2 = rss / (len(x) - 2)
    se_beta0_expected = np.sqrt(sigma2 * (1 / len(x) + x_bar ** 2 / sxx))
    se_beta1_expected = np.sqrt(sigma2 / sxx)

    assert se[0] == pytest.approx(se_beta0_expected)
    assert se[1] == pytest.approx(se_beta1_expected)


# --- 4. End-to-end, cross-validated against statsmodels.OLS ---
@pytest.fixture(scope="module")
def advertising_data():
    df = pd.read_csv(PUBLICIDADE_CSV)
    return df


def test_summarize_inference_matches_statsmodels_three_predictors(advertising_data):
    X = advertising_data[["TV", "radio", "jornal"]].to_numpy(float)
    y = advertising_data["vendas"].to_numpy(float)

    fit = fit_ols(X, y)
    summary = summarize_inference(X, y, fit, feature_names=["TV", "radio", "jornal"])

    sm_fit = sm.OLS(y, sm.add_constant(X)).fit()

    assert summary.coefficients == pytest.approx(np.asarray(sm_fit.params), abs=1e-8)
    assert summary.standard_errors == pytest.approx(np.asarray(sm_fit.bse), abs=1e-8)
    assert summary.t_statistics == pytest.approx(np.asarray(sm_fit.tvalues), abs=1e-6)
    assert summary.p_values == pytest.approx(np.asarray(sm_fit.pvalues), abs=1e-8)
    assert summary.residual_standard_error == pytest.approx(np.sqrt(sm_fit.mse_resid))
    assert summary.degrees_of_freedom == int(sm_fit.df_resid)
    assert summary.names == ("Intercept", "TV", "radio", "jornal")


def test_summarize_inference_matches_statsmodels_single_predictor(advertising_data):
    X = advertising_data[["TV"]].to_numpy(float)
    y = advertising_data["vendas"].to_numpy(float)

    fit = fit_ols(X, y)
    summary = summarize_inference(X, y, fit, feature_names=["TV"])
    sm_fit = sm.OLS(y, sm.add_constant(X)).fit()

    assert summary.t_statistics == pytest.approx(np.asarray(sm_fit.tvalues), abs=1e-6)
    assert summary.p_values == pytest.approx(np.asarray(sm_fit.pvalues), abs=1e-8)


def test_coefficient_standard_errors_matches_statsmodels_on_synthetic_data():
    """A case where a predictor is genuinely non-significant, cross-checked
    against statsmodels rather than a hand-derived expectation."""
    rng = np.random.default_rng(99)
    n = 60
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    noise_feature = rng.normal(0, 1, n)  # unrelated to y
    y = 4.0 * x1 - 2.0 * x2 + 5.0 + rng.normal(0, 0.5, n)

    X = np.column_stack([x1, x2, noise_feature])
    fit = fit_ols(X, y)
    summary = summarize_inference(X, y, fit)

    sm_fit = sm.OLS(y, sm.add_constant(X)).fit()
    assert summary.p_values == pytest.approx(np.asarray(sm_fit.pvalues), abs=1e-6)
    # the noise feature should NOT look significant
    assert summary.p_values[3] > 0.05


# --- guards ---
def test_residual_standard_error_rejects_degenerate_degrees_of_freedom():
    with pytest.raises(ValueError, match="n_samples > n_features"):
        residual_standard_error(np.array([1.0, 2.0]), np.array([1.1, 2.1]), n_features=1)


def test_coefficient_standard_errors_rejects_degenerate_degrees_of_freedom():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 samples, 2 features -> df=2-3=-1
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="n_samples > n_features"):
        coefficient_standard_errors(X, y, y)


def test_t_statistics_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        t_statistics(np.array([1.0, 2.0]), np.array([1.0]))


def test_p_values_rejects_non_positive_degrees_of_freedom():
    with pytest.raises(ValueError, match="degrees_of_freedom must be positive"):
        p_values(np.array([2.0]), degrees_of_freedom=0)


def test_summarize_inference_rejects_wrong_number_of_feature_names():
    X = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 8.0], [4.0, 1.0]])
    y = np.array([2.0, 3.0, 5.0, 4.0])
    fit = fit_ols(X, y)
    with pytest.raises(ValueError, match="feature_names has"):
        summarize_inference(X, y, fit, feature_names=["only_one"])


def test_summarize_inference_default_feature_names():
    X = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 8.0], [4.0, 1.0]])
    y = np.array([2.0, 3.0, 5.0, 4.0])
    fit = fit_ols(X, y)
    summary = summarize_inference(X, y, fit)
    assert summary.names == ("Intercept", "x1", "x2")
