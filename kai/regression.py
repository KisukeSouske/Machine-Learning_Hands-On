"""Linear regression solvers, decoupled from data loading and plotting.

Two ways to fit the same model:

- `fit_gradient_descent` - iterative, the didactic path. Produces a loss
  history, needs a learning rate, and its behaviour depends on how well
  conditioned the problem is.
- `fit_ols` - closed form via the normal equations. No hyperparameters, one
  exact answer. This is the one statistical inference (standard errors,
  t-tests, VIF) must be built on, since those formulas assume the exact
  least-squares solution.

Everything here works on plain numpy arrays: no CSV reading, no scaling, no
matplotlib. That is what makes these callable from a script, from `Model`, or
from a VIF routine that regresses one feature against the others.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from kai.metrics import mean_squared_error, mean_squared_error_derivation, r_squared, squared_loss


@dataclass(frozen=True)
class LinearFit:
    """The result of fitting y ~ X.

    weights: one coefficient per column of X, shape (n_features,).
    bias: the intercept.
    loss_history: MSE per epoch for iterative fits; None for closed-form ones,
        which have no iterations to report.
    """

    weights: np.ndarray
    bias: float
    loss_history: tuple[float, ...] | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predictions for X, in the same feature space the fit was made in."""
        X = np.asarray(X, dtype=float)
        # mirror _as_design_inputs's promotion, so a 1-D X round-trips through
        # fit_ols(x, y).predict(x) the same way it went in
        if X.ndim == 1 and self.weights.shape[0] == 1:
            X = X.reshape(-1, 1)
        return self.bias + X @ self.weights


def _as_design_inputs(X, y) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float arrays, promote 1-D X to a single column, check shapes."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got {X.ndim} dimensions.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got {y.ndim} dimensions.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples "
            f"(got {X.shape[0]} and {y.shape[0]})."
        )
    return X, y


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones, turning X into the design matrix.

    The column of ones is what lets the intercept be estimated as just another
    coefficient, so the whole fit is a single linear system.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(X.shape[0]), X])


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b by Gaussian elimination with partial pivoting.

    Written out rather than delegated to `np.linalg.solve` so the algorithm is
    visible: for each column, pick the largest remaining entry as the pivot
    (that is the "partial pivoting" part, and it is what keeps the arithmetic
    stable), eliminate below it, then substitute backwards.

    Raises:
    ValueError: If A is singular to working precision, i.e. no unique solution.
    """
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    n = b.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A must be square and match b (got {A.shape} and {b.shape}).")

    # scale-aware threshold: what counts as "zero" depends on the magnitudes in A
    singular_threshold = 1e-12 * max(np.abs(A).max(), 1.0)

    # --- forward elimination ---
    for col in range(n):
        # partial pivoting: swap in the row whose entry in this column is largest
        pivot_row = col + int(np.argmax(np.abs(A[col:, col])))
        if abs(A[pivot_row, col]) <= singular_threshold:
            raise ValueError(
                "Matrix is singular to working precision: the system has no unique "
                "solution. With normal equations this usually means perfectly "
                "collinear predictors (or fewer samples than coefficients)."
            )
        if pivot_row != col:
            A[[col, pivot_row]] = A[[pivot_row, col]]
            b[[col, pivot_row]] = b[[pivot_row, col]]

        # subtract multiples of the pivot row from every row beneath it
        for row in range(col + 1, n):
            factor = A[row, col] / A[col, col]
            A[row, col:] -= factor * A[col, col:]
            b[row] -= factor * b[col]

    # --- back substitution ---
    x = np.zeros(n)
    for row in range(n - 1, -1, -1):
        x[row] = (b[row] - A[row, row + 1:] @ x[row + 1:]) / A[row, row]
    return x


def fit_ols(X, y) -> LinearFit:
    """Fit y ~ X exactly, by ordinary least squares (normal equations).

    Minimising the squared residuals leads to the normal equations

        (Xd^T Xd) beta = Xd^T y

    where Xd is the design matrix (X plus a column of ones). That is a square
    linear system, solved here by `solve_linear_system`.

    Parameters:
    X (array-like): Features, shape (n_samples,) or (n_samples, n_features).
    y (array-like): Target, shape (n_samples,).

    Returns:
    LinearFit: with `loss_history=None`, since nothing is iterated.

    Raises:
    ValueError: If the predictors are perfectly collinear, or there are fewer
        samples than coefficients to estimate; in both cases Xd^T Xd is
        singular and no unique solution exists.

    Note:
    Forming Xd^T Xd squares the condition number of the problem. It is the
    textbook formulation and is exact for well-conditioned data; QR or SVD
    based solvers are preferred when the predictors are near-collinear.
    """
    X, y = _as_design_inputs(X, y)
    design = add_intercept(X)
    if design.shape[0] < design.shape[1]:
        raise ValueError(
            f"OLS needs at least as many samples as coefficients "
            f"(got {design.shape[0]} samples for {design.shape[1]} coefficients)."
        )

    normal_matrix = design.T @ design       # Xd^T Xd
    moment_vector = design.T @ y            # Xd^T y
    coefficients = solve_linear_system(normal_matrix, moment_vector)

    return LinearFit(weights=coefficients[1:], bias=float(coefficients[0]))


def fit_gradient_descent(
    X,
    y,
    learning_rate: float,
    batch_size: int = 100,
    epochs: int = 10_000,
    tolerance: float = 1e-4,
    random_state: int | None = None,
) -> LinearFit:
    """Fit y ~ X by mini-batch gradient descent.

    Parameters:
    X (array-like): Features, shape (n_samples,) or (n_samples, n_features).
        Already in whatever space you want to train in - this function does no
        scaling of its own.
    y (array-like): Target, shape (n_samples,).
    learning_rate (float): Step size. Must satisfy lr < 2/lambda_max of the MSE
        Hessian or the run diverges; standardizing X first raises that ceiling
        by orders of magnitude.
    batch_size (int): Samples per gradient step. Values >= n_samples reduce
        this to plain full-batch gradient descent.
    epochs (int): Maximum passes over the data; a safety ceiling, not a target.
    tolerance (float): Relative convergence threshold. Stops when
        ||grad|| <= tolerance * ||grad_initial||, which makes the criterion
        independent of the units of y and of the features.
    random_state (int | None): Seed for the per-epoch shuffle. Pass an int for
        reproducible runs.

    Returns:
    LinearFit: including the per-epoch loss history.

    Raises:
    ValueError: If the loss becomes non-finite, i.e. training diverged.
    """
    X, y = _as_design_inputs(X, y)
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history: list[float] = []
    rng = np.random.default_rng(random_state)

    # Reference magnitude for the convergence test, measured at the STARTING
    # parameters (before any update). Taking it after the first epoch would be
    # self-defeating: a run that converges immediately would compare its
    # already-tiny gradient against itself and never satisfy the threshold.
    initial_gradient_norm = gradient_norm(y, bias + X @ weights, X) or 1.0

    for epoch in range(epochs):
        indices = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            batch = indices[start:start + batch_size]
            x_batch, y_batch = X[batch], y[batch]
            weight_slope, bias_slope = mean_squared_error_derivation(
                y_batch, bias + x_batch @ weights, x_batch
            )
            weights -= learning_rate * weight_slope
            bias -= learning_rate * bias_slope

        y_pred = bias + X @ weights
        epoch_loss = mean_squared_error(y, y_pred)
        if not np.isfinite(epoch_loss):
            raise ValueError(
                f"Training diverged at epoch {epoch}: loss became {epoch_loss}. "
                f"Try a smaller learning_rate (current={learning_rate}) or "
                f"standardize the features first."
            )
        loss_history.append(epoch_loss)

        # Converged when the full-dataset gradient has shrunk to a small
        # FRACTION of where it started. A flat loss delta is not enough on its
        # own: in an ill-conditioned problem the loss can stall while the
        # parameters are still far from the optimum.
        if gradient_norm(y, y_pred, X) <= tolerance * initial_gradient_norm:
            break

    return LinearFit(weights=weights, bias=float(bias), loss_history=tuple(loss_history))


def gradient_norm(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> float:
    """Euclidean norm of the full MSE gradient (weights and bias together)."""
    weight_slope, bias_slope = mean_squared_error_derivation(y_true, y_pred, X)
    return float(np.sqrt(np.sum(weight_slope ** 2) + bias_slope ** 2))

def variance_inflation_factors(features: dict) -> dict:
    """Compute variance inflation factors for each feature.

    For each predictor Xj, the VIF measures how much Xj can be explained by the
    other predictors. The implementation fits one regression of Xj against the
    remaining columns and computes

        VIF(Xj) = 1 / (1 - R^2_{Xj | X_{-j}})

    using the existing OLS solver and R^2 helper.
    """
    if not features:
        raise ValueError("features must not be empty.")

    columns = list(features.keys())
    feature_arrays = {}
    n_samples = None

    for column in columns:
        values = np.asarray(features[column], dtype=float)
        if values.ndim != 1:
            raise ValueError(f"Feature '{column}' must be 1-D, got shape {values.shape}.")
        if n_samples is None:
            n_samples = values.shape[0]
        elif values.shape[0] != n_samples:
            raise ValueError(
                f"All features must contain the same number of samples; "
                f"'{column}' has {values.shape[0]} while the first feature has {n_samples}."
            )
        feature_arrays[column] = values

    vif: dict[str, float] = {}
    for target_name in columns:
        target_values = feature_arrays[target_name]
        other_names = [name for name in columns if name != target_name]
        if not other_names:
            vif[target_name] = 1.0
            continue

        other_values = [feature_arrays[name] for name in other_names]
        design_matrix = np.column_stack(other_values)

        try:
            fit_result = fit_ols(design_matrix, target_values)
        except ValueError as exc:
            raise ValueError(
                f"VIF for '{target_name}' is undefined because the predictors "
                "are perfectly collinear."
            ) from exc

        fitted_values = fit_result.predict(design_matrix)
        r_squared_value = r_squared(target_values, fitted_values)
        if r_squared_value >= 1.0:
            raise ValueError(
                f"VIF for '{target_name}' is undefined because the predictors "
                "are perfectly collinear."
            )
        vif[target_name] = 1.0 / (1.0 - r_squared_value)

    return vif


def _inverse_diagonal(A: np.ndarray) -> np.ndarray:
    """Diagonal of A^-1, via `solve_linear_system` one column at a time.

    Solving A @ x = e_i for each standard basis vector e_i gives the i-th
    column of A^-1; its i-th entry is what coefficient_standard_errors needs.
    Written this way (rather than np.linalg.inv) to reuse the same
    hand-written elimination as `fit_ols`, instead of a second solver.
    """
    n = A.shape[0]
    identity = np.eye(n)
    diagonal = np.empty(n)
    for i in range(n):
        column = solve_linear_system(A, identity[i])
        diagonal[i] = column[i]
    return diagonal


def residual_standard_error(y_true, y_pred, n_features: int) -> float:
    """RSE = sqrt(RSS / (n - p - 1)): the estimate of the error term's standard
    deviation (ISLR eq. 3.15, generalized to p predictors; UFMG-EST-027 Cap.11
    calls the squared version sigma^2_hat).

    Raises:
    ValueError: If n_samples <= n_features + 1, where the degrees of freedom
        (n - p - 1) is zero or negative and RSE is undefined.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    degrees_of_freedom = y_true.size - n_features - 1
    if degrees_of_freedom <= 0:
        raise ValueError(
            f"Residual standard error requires n_samples > n_features + 1 "
            f"(got n_samples={y_true.size}, n_features={n_features})."
        )
    return float(np.sqrt(squared_loss(y_true, y_pred) / degrees_of_freedom))


def coefficient_standard_errors(X, y_true, y_pred) -> np.ndarray:
    """Standard error of every fitted coefficient, intercept first.

    SE(beta_hat) = sqrt(sigma_hat^2 * diag((Xd^T Xd)^-1)), where Xd is the
    design matrix and sigma_hat^2 = RSS/(n-p-1) (ISLR eq. 3.8, generalized
    from simple to multiple regression via the (X^T X)^-1 matrix form
    mentioned on ISLR p.p.75-77 and derived in matrix form in most
    econometrics texts). Reduces exactly to eq. 3.8 when X has one column.

    Returns:
    np.ndarray: shape (n_features + 1,), i.e. [SE(intercept), SE(coef_1), ...].

    Raises:
    ValueError: If n_samples <= n_features + 1 (degenerate degrees of freedom).
    """
    X, y_true = _as_design_inputs(X, y_true)
    y_pred = np.asarray(y_pred, dtype=float)
    design = add_intercept(X)
    n_samples, n_coefficients = design.shape
    degrees_of_freedom = n_samples - n_coefficients
    if degrees_of_freedom <= 0:
        raise ValueError(
            f"Standard errors require n_samples > n_features + 1 "
            f"(got n_samples={n_samples}, n_features={n_coefficients - 1})."
        )
    sigma_squared = squared_loss(y_true, y_pred) / degrees_of_freedom
    normal_matrix = design.T @ design
    return np.sqrt(sigma_squared * _inverse_diagonal(normal_matrix))


def t_statistics(coefficients: np.ndarray, standard_errors: np.ndarray) -> np.ndarray:
    """t = coefficient / SE(coefficient) (ISLR eq. 3.14; UFMG-EST-027 Cap.11 eq.
    for T0), one per coefficient. Tests H0: coefficient = 0 for each."""
    coefficients = np.asarray(coefficients, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    if coefficients.shape != standard_errors.shape:
        raise ValueError(
            f"coefficients and standard_errors must have the same shape "
            f"(got {coefficients.shape} and {standard_errors.shape})."
        )
    return coefficients / standard_errors


def p_values(t_stats: np.ndarray, degrees_of_freedom: int) -> np.ndarray:
    """Two-tailed p-value for each t-statistic, under a t-distribution with
    the given degrees of freedom: p = 2 * P(T > |t|).

    Uses the survival function (`sf`) rather than `1 - cdf`: for a large |t|
    (a strongly significant coefficient), cdf(|t|) rounds to exactly 1.0 in
    float64, and `1 - 1.0` underflows to a false p=0.0. `sf` computes the
    tail probability directly, without that cancellation.

    Raises:
    ValueError: If degrees_of_freedom <= 0.
    """
    if degrees_of_freedom <= 0:
        raise ValueError(f"degrees_of_freedom must be positive (got {degrees_of_freedom}).")
    t_stats = np.asarray(t_stats, dtype=float)
    return 2.0 * scipy_stats.t.sf(np.abs(t_stats), df=degrees_of_freedom)


@dataclass(frozen=True)
class InferenceSummary:
    """Per-coefficient inference for a fitted linear model, intercept first.

    Everything a results table needs in one place: names line up positionally
    with coefficients/standard_errors/t_statistics/p_values.
    """

    names: tuple[str, ...]
    coefficients: np.ndarray
    standard_errors: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray
    residual_standard_error: float
    degrees_of_freedom: int


def summarize_inference(X, y_true, fit: LinearFit, feature_names: list[str] | None = None) -> InferenceSummary:
    """Build the full per-coefficient inference table for a fit produced by
    `fit_ols` (statistically valid only for the exact least-squares solution;
    see the note on `fit_ols`).

    Parameters:
    X (array-like): The same features the fit was made on.
    y_true (array-like): The same target the fit was made on.
    fit (LinearFit): Typically the result of `fit_ols(X, y_true)`.
    feature_names (list[str] | None): Labels for the columns of X, in order.
        Defaults to "x1", "x2", ... if not given.
    """
    X, y_true = _as_design_inputs(X, y_true)
    y_pred = fit.predict(X)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"x{i + 1}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but X has {n_features} columns."
        )

    coefficients = np.concatenate([[fit.bias], fit.weights])
    standard_errors = coefficient_standard_errors(X, y_true, y_pred)
    t_stats = t_statistics(coefficients, standard_errors)
    degrees_of_freedom = X.shape[0] - n_features - 1
    p_vals = p_values(t_stats, degrees_of_freedom)
    rse = residual_standard_error(y_true, y_pred, n_features)

    return InferenceSummary(
        names=("Intercept", *feature_names),
        coefficients=coefficients,
        standard_errors=standard_errors,
        t_statistics=t_stats,
        p_values=p_vals,
        residual_standard_error=rse,
        degrees_of_freedom=degrees_of_freedom,
    )