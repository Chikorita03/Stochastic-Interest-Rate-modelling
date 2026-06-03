"""
yield_curve.py
==============
Production module for Hybrid Empirical-CIR Yield Curve Reconstruction.

Implements the full reconstruction workflow:
  - CIR term-structure mathematics (gamma, B, log_A, yield)
  - Parameter loading from results/cir_parameters.csv
  - Feature construction per maturity
  - Training-matrix assembly (spread formulation)
  - Per-maturity linear/ridge model fitting
  - Yield-curve reconstruction from fitted models
  - Artifact export via joblib

Designed for import into Notebook 03 (yield-curve reconstruction).

Notes
-----
Calibrated kappa = 0.001 is extremely small; all CIR functions use
numerically stable formulations to avoid overflow/underflow.
Yields in the dataset are decimal rates (e.g. 0.005 = 0.5%); no unit
conversion is performed anywhere in this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATURITY_MAP: Dict[str, float] = {
    "3M": 0.25,
    "6M": 0.50,
    "9M": 0.75,
    "1Y": 1.00,
    "2Y": 2.00,
    "5Y": 5.00,
    "10Y": 10.00,
    "20Y": 20.00,
    "30Y": 30.00,
}

SHORT_RATE_MATURITY: str = "3M"
RECONSTRUCTION_MATURITIES: tuple[str, ...] = (
    "6M", "9M", "1Y", "2Y", "5Y", "10Y", "20Y", "30Y"
)

# Smallest τ used as the spread anchor (3-month yield)
SPREAD_ANCHOR_TAU: float = 0.25

# Numerical floor to prevent division-by-zero in CIR bond functions
_GAMMA_MIN: float = 1e-12


# ---------------------------------------------------------------------------
# Section 1 – Maturity utilities
# ---------------------------------------------------------------------------

def maturity_to_years(maturity: str) -> float:
    """Convert a maturity label string to years.

    Parameters
    ----------
    maturity : str
        A maturity label such as ``'3M'``, ``'1Y'``, ``'10Y'``.

    Returns
    -------
    float
        The equivalent number of years.

    Raises
    ------
    KeyError
        If *maturity* is not recognised.

    Examples
    --------
    >>> maturity_to_years("3M")
    0.25
    >>> maturity_to_years("10Y")
    10.0
    """
    if maturity not in MATURITY_MAP:
        raise KeyError(
            f"Unknown maturity label '{maturity}'. "
            f"Recognised labels: {sorted(MATURITY_MAP)}"
        )
    return MATURITY_MAP[maturity]


# ---------------------------------------------------------------------------
# Section 2 – CIR term-structure functions
# ---------------------------------------------------------------------------

def cir_gamma(kappa: float, sigma: float) -> float:
    """Compute the CIR auxiliary parameter γ.

    Under the CIR model the bond-pricing auxiliary parameter is:

        γ = sqrt(κ² + 2σ²)

    A minimum floor is applied so that downstream divisions remain
    numerically stable even when κ and σ are both very small.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed (κ > 0).
    sigma : float
        Volatility coefficient (σ > 0).

    Returns
    -------
    float
        γ = max(sqrt(κ² + 2σ²), _GAMMA_MIN).

    Raises
    ------
    ValueError
        If *kappa* or *sigma* are not strictly positive.
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    gamma = float(np.sqrt(kappa ** 2 + 2.0 * sigma ** 2))
    if gamma < _GAMMA_MIN:
        logger.warning(
            "Computed gamma = %.3e is below floor %.3e; clamping.",
            gamma, _GAMMA_MIN,
        )
        gamma = _GAMMA_MIN
    return gamma


def cir_B(tau: float, kappa: float, sigma: float) -> float:
    """Compute the CIR bond-loading factor B(τ).

    The standard CIR zero-coupon formula gives:

        B(τ) = 2(e^{γτ} - 1) / [(γ + κ)(e^{γτ} - 1) + 2γ]

    For very small γτ the numerator and denominator both approach zero;
    a first-order Taylor expansion is used instead:

        B(τ) ≈ τ / [1 + (κ/2)τ]

    to maintain numerical stability.

    Parameters
    ----------
    tau : float
        Time to maturity in years (τ > 0).
    kappa : float
        Mean-reversion speed (κ > 0).
    sigma : float
        Volatility coefficient (σ > 0).

    Returns
    -------
    float
        B(τ) ≥ 0.

    Raises
    ------
    ValueError
        If *tau* is not strictly positive.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}.")

    gamma = cir_gamma(kappa, sigma)
    gamma_tau = gamma * tau

    # Use Taylor approximation when γτ is small to avoid cancellation error.
    if gamma_tau < 1e-6:
        B = tau / (1.0 + 0.5 * kappa * tau)
        return float(B)

    exp_gt = np.exp(gamma_tau)
    numerator = 2.0 * (exp_gt - 1.0)
    denominator = (gamma + kappa) * (exp_gt - 1.0) + 2.0 * gamma

    if abs(denominator) < _GAMMA_MIN:
        logger.warning(
            "Near-zero denominator in cir_B (tau=%.4f). Returning 0.", tau
        )
        return 0.0

    return float(numerator / denominator)


def cir_log_A(tau: float, kappa: float, theta: float, sigma: float) -> float:
    """Compute log A(τ) in the CIR bond-pricing formula.

    The standard expression is:

        log A(τ) = (2α / σ²) * log[ 2γ exp((γ+κ)τ/2) / ((γ+κ)(e^{γτ}-1) + 2γ) ]

    where α = κθ.

    The log is evaluated directly to avoid computing extremely large
    intermediate exponentials when τ is large.

    Parameters
    ----------
    tau : float
        Time to maturity in years (τ > 0).
    kappa : float
        Mean-reversion speed (κ > 0).
    theta : float
        Long-run mean of the short rate.
    sigma : float
        Volatility coefficient (σ > 0).

    Returns
    -------
    float
        log A(τ) (can be negative for long maturities).

    Raises
    ------
    ValueError
        If *tau* is not strictly positive or if *sigma* is zero.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    alpha = kappa * theta          # reliable composite drift parameter
    gamma = cir_gamma(kappa, sigma)
    gamma_tau = gamma * tau
    sigma2 = sigma ** 2

    two_alpha_over_sigma2 = 2.0 * alpha / sigma2

    # log[ 2γ exp((γ+κ)τ/2) / denominator ]
    # Written as: log(2γ) + (γ+κ)τ/2 - log(denominator)
    # to avoid evaluating exp((γ+κ)τ/2) directly for large τ.

    if gamma_tau < 1e-6:
        # Taylor: denominator ≈ 2γ (1 + (γ+κ)τ/2 + …)
        # Ratio ≈ exp((γ+κ)τ/2) / (1 + (γ+κ)τ/2)
        # log-ratio ≈ (γ+κ)τ/2 - log(1 + (γ+κ)τ/2)
        half_sum_tau = 0.5 * (gamma + kappa) * tau
        log_ratio = half_sum_tau - np.log1p(half_sum_tau)
        return float(two_alpha_over_sigma2 * log_ratio)

    exp_gt = np.exp(gamma_tau)
    denominator = (gamma + kappa) * (exp_gt - 1.0) + 2.0 * gamma

    if denominator <= 0.0:
        raise ValueError(
            f"Non-positive denominator in cir_log_A: {denominator:.6e} "
            f"(tau={tau}, kappa={kappa}, sigma={sigma})."
        )

    log_arg = (
        np.log(2.0 * gamma)
        + 0.5 * (gamma + kappa) * tau
        - np.log(denominator)
    )
    return float(two_alpha_over_sigma2 * log_arg)


def cir_yield(
    tau: float,
    r_t: float,
    kappa: float,
    theta: float,
    sigma: float,
) -> float:
    """Compute the CIR model-implied continuously-compounded yield.

    Under the CIR model the zero-coupon yield is:

        y_CIR(t, τ) = [B(τ) * r_t - log A(τ)] / τ

    Parameters
    ----------
    tau : float
        Time to maturity in years (τ > 0).
    r_t : float
        Observed short rate at time t (decimal, e.g. 0.005 for 0.5%).
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run mean of the short rate.
    sigma : float
        Volatility coefficient.

    Returns
    -------
    float
        CIR-implied yield y_CIR(t, τ) in decimal form.

    Raises
    ------
    ValueError
        If *tau* ≤ 0.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}.")

    B = cir_B(tau, kappa, sigma)
    log_A = cir_log_A(tau, kappa, theta, sigma)
    return float((B * r_t - log_A) / tau)


# ---------------------------------------------------------------------------
# Section 3 – Parameter loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CIRParameters:
    """Immutable container for calibrated CIR model parameters.

    Attributes
    ----------
    alpha : float
        Drift product α = κ × θ.  Treated as the reliable composite.
    kappa : float
        Mean-reversion speed κ.
    theta : float
        Long-run mean θ (may be economically unrealistic due to the
        κ–θ identification ridge; use alpha instead where possible).
    sigma : float
        Volatility coefficient σ.
    """

    alpha: float
    kappa: float
    theta: float
    sigma: float

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}.")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}.")


def load_cir_parameters(
    path: Union[str, Path] = "results/cir_parameters.csv",
) -> CIRParameters:
    """Load calibrated CIR parameters from a CSV file.

    The expected CSV format has columns ``parameter`` and ``value``::

        parameter,value,description
        alpha,0.00240671785519546,Drift product alpha = kappa * theta
        kappa,0.001,Mean reversion speed
        theta,2.40671785519546,Long-run mean of short rate
        sigma,0.0469497250260536,Volatility coefficient

    Parameters
    ----------
    path : str or Path, optional
        Path to the CIR parameters CSV file.
        Default: ``'results/cir_parameters.csv'``.

    Returns
    -------
    CIRParameters
        Dataclass with fields alpha, kappa, theta, sigma.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *path*.
    KeyError
        If required parameters are missing from the file.
    ValueError
        If any parameter value cannot be cast to float, or if kappa/sigma
        fail positivity checks.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CIR parameter file not found: {path.resolve()}"
        )

    df = pd.read_csv(path)

    required_cols = {"parameter", "value"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        raise KeyError(
            f"Parameter file is missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Normalise column names to lowercase for robustness
    df.columns = df.columns.str.lower()
    df["parameter"] = df["parameter"].str.strip().str.lower()

    param_series = df.set_index("parameter")["value"]

    required_params = ["alpha", "kappa", "theta", "sigma"]
    missing_params = [p for p in required_params if p not in param_series.index]
    if missing_params:
        raise KeyError(
            f"Missing parameters in file: {missing_params}. "
            f"Found: {list(param_series.index)}"
        )

    try:
        params = {p: float(param_series[p]) for p in required_params}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not convert parameter values to float: {exc}"
        ) from exc

    logger.info(
        "Loaded CIR parameters: alpha=%.6e  kappa=%.6e  theta=%.6e  sigma=%.6e",
        params["alpha"], params["kappa"], params["theta"], params["sigma"],
    )
    return CIRParameters(**params)


# ---------------------------------------------------------------------------
# Section 4 – Feature construction
# ---------------------------------------------------------------------------

def compute_cir_features(
    short_rate_series: pd.Series,
    maturity: str,
    cir_parameters: CIRParameters,
    *,
    include_squared: bool = True,
) -> pd.DataFrame:
    """Construct CIR-based regression features for a single maturity.

    Features constructed per observation:

    1. ``short_rate``     — observed short rate r_t
    2. ``B_times_r``      — CIR loading factor B(τ) × r_t
    3. ``cir_spread``     — y_CIR(t, τ) − y_CIR(t, 0.25)
    4. ``short_rate_sq``  — r_t² (optional, included by default)

    Parameters
    ----------
    short_rate_series : pd.Series
        Time series of observed 3-month (short) rates in decimal form.
        Index should be datetime-like.
    maturity : str
        Target maturity label (e.g. ``'10Y'``).
    cir_parameters : CIRParameters
        Calibrated CIR parameter dataclass.
    include_squared : bool, optional
        Whether to include the r_t² feature.  Default True.

    Returns
    -------
    pd.DataFrame
        DataFrame with index matching *short_rate_series* and columns:
        ``['short_rate', 'B_times_r', 'cir_spread',
        'short_rate_sq']`` (last column omitted if *include_squared* is False).

    Raises
    ------
    KeyError
        If *maturity* is unrecognised.
    ValueError
        If *short_rate_series* is empty.
    """
    if short_rate_series.empty:
        raise ValueError("short_rate_series must not be empty.")

    tau = maturity_to_years(maturity)
    kappa = cir_parameters.kappa
    theta = cir_parameters.theta
    sigma = cir_parameters.sigma

    # Pre-compute scalar CIR quantities that do not depend on r_t
    B_tau = cir_B(tau, kappa, sigma)
    B_anchor = cir_B(SPREAD_ANCHOR_TAU, kappa, sigma)

    r = short_rate_series.to_numpy(dtype=float)

    # Vectorised log_A is a scalar (independent of r_t)
    log_A_tau = cir_log_A(tau, kappa, theta, sigma)
    log_A_anchor = cir_log_A(SPREAD_ANCHOR_TAU, kappa, theta, sigma)

    # CIR yield: y = (B*r - log_A) / tau
    cir_yld = (B_tau * r - log_A_tau) / tau
    cir_anchor = (B_anchor * r - log_A_anchor) / SPREAD_ANCHOR_TAU
    cir_spr = cir_yld - cir_anchor

    data: Dict[str, np.ndarray] = {
        "short_rate": r,
        "B_times_r": B_tau * r,
        "cir_spread": cir_spr,
    }
    if include_squared:
        data["short_rate_sq"] = r ** 2

    return pd.DataFrame(data, index=short_rate_series.index)


# ---------------------------------------------------------------------------
# Section 5 – Training matrix construction
# ---------------------------------------------------------------------------

def build_reconstruction_dataset(
    training_dataframe: pd.DataFrame,
    cir_parameters: CIRParameters,
    *,
    include_squared: bool = True,
) -> Dict[str, Dict[str, Union[pd.DataFrame, pd.Series]]]:
    """Assemble per-maturity training matrices using the spread formulation.

    For each reconstruction maturity the target variable is:

        spread = target_yield − short_rate

    so that the final yield prediction is:

        yield_hat = short_rate + predicted_spread

    Parameters
    ----------
    training_dataframe : pd.DataFrame
        Cleaned training dataset with columns:
        ``['Date', '3M', '6M', '9M', '1Y', '2Y', '5Y', '10Y', '20Y', '30Y']``.
        The ``Date`` column may be the DataFrame index or a regular column.
        Yields are decimal rates.
    cir_parameters : CIRParameters
        Calibrated CIR parameter dataclass.
    include_squared : bool, optional
        Whether to include the r_t² feature.  Default True.

    Returns
    -------
    dict
        A dictionary keyed by maturity label (e.g. ``'10Y'``).  Each value
        is itself a dict with keys:

        ``'X'``  — pd.DataFrame of regression features.
        ``'y'``  — pd.Series of target spreads.

    Raises
    ------
    KeyError
        If required maturity columns are absent from *training_dataframe*.
    ValueError
        If *training_dataframe* is empty.
    """
    df = _prepare_training_dataframe(training_dataframe)

    if df.empty:
        raise ValueError("training_dataframe is empty after preprocessing.")

    short_rate_col = SHORT_RATE_MATURITY  # "3M"
    if short_rate_col not in df.columns:
        raise KeyError(
            f"Short-rate column '{short_rate_col}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    short_rate_series = df[short_rate_col]
    datasets: Dict[str, Dict[str, Union[pd.DataFrame, pd.Series]]] = {}

    for mat in RECONSTRUCTION_MATURITIES:
        if mat not in df.columns:
            logger.warning(
                "Maturity '%s' not in training data; skipping.", mat
            )
            continue

        target_yield = df[mat]
        spread = target_yield - short_rate_series
        spread.name = f"spread_{mat}"

        X = compute_cir_features(
            short_rate_series,
            mat,
            cir_parameters,
            include_squared=include_squared,
        )

        # Align on common non-NaN rows
        combined = pd.concat([X, spread], axis=1).dropna()
        X_clean = combined[X.columns]
        y_clean = combined[spread.name]

        datasets[mat] = {"X": X_clean, "y": y_clean}
        logger.debug(
            "Built dataset for %s: %d observations, %d features.",
            mat, len(X_clean), X_clean.shape[1],
        )

    return datasets


def _prepare_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Internal helper: normalise the training dataframe.

    - If a 'Date' column is present, set it as the index.
    - Parse the index as datetime if possible.
    - Return a copy.
    """
    df = df.copy()
    if "Date" in df.columns:
        df = df.set_index("Date")
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass  # Keep as-is if not parseable
    return df


# ---------------------------------------------------------------------------
# Section 6 – Model fitting
# ---------------------------------------------------------------------------

def fit_reconstruction_models(
    datasets: Dict[str, Dict[str, Union[pd.DataFrame, pd.Series]]],
    *,
    use_ridge: bool = False,
    ridge_alpha: float = 1.0,
) -> Dict[str, Dict]:
    """Fit one linear regression model per maturity.

    Parameters
    ----------
    datasets : dict
        Output of :func:`build_reconstruction_dataset`.  Each value must
        contain keys ``'X'`` (feature DataFrame) and ``'y'`` (target Series).
    use_ridge : bool, optional
        If True, use Ridge regression instead of OLS.  Default False.
    ridge_alpha : float, optional
        Regularisation strength for Ridge.  Ignored when *use_ridge* is False.
        Default 1.0.

    Returns
    -------
    dict
        Dictionary keyed by maturity label; values are fitted sklearn
        estimator instances.

    Raises
    ------
    ValueError
        If *datasets* is empty or if any dataset is missing required keys.
    """
    if not datasets:
        raise ValueError("datasets dict is empty; nothing to fit.")

    models: Dict[str, Dict] = {}

    for mat, data in datasets.items():
        if "X" not in data or "y" not in data:
            raise ValueError(
                f"Dataset for maturity '{mat}' is missing 'X' or 'y' keys."
            )

        X: pd.DataFrame = data["X"]
        y: pd.Series = data["y"]

        if X.empty or y.empty:
            logger.warning("Empty dataset for maturity '%s'; skipping.", mat)
            continue

        if use_ridge:
            model: Union[LinearRegression, Ridge] = Ridge(alpha=ridge_alpha)
        else:
            model = LinearRegression()

        model.fit(X.to_numpy(), y.to_numpy())
        models[mat] = {
            "model": model,
            "feature_names": list(X.columns)
            }

        logger.info(
            "Fitted %s for maturity %s (n=%d, p=%d).",
            type(model).__name__, mat, len(y), X.shape[1],
        )

    return models


# ---------------------------------------------------------------------------
# Section 7 – Yield reconstruction
# ---------------------------------------------------------------------------

def reconstruct_yield_curve(
    short_rate_series: pd.Series,
    models: Dict[str, Union[LinearRegression, Ridge]],
    cir_parameters: CIRParameters,
    *,
    include_squared: bool = True,
) -> pd.DataFrame:
    """Reconstruct the full yield curve from fitted per-maturity models.

    For each maturity the reconstructed yield is:

        yield_hat(t, τ) = r_t + model.predict(X(t, τ))

    where X(t, τ) is the feature matrix from :func:`compute_cir_features`
    and the model predicts the spread over the short rate.

    Parameters
    ----------
    short_rate_series : pd.Series
        Observed 3-month short rates (decimal).
    models : dict
        Fitted models keyed by maturity label, as returned by
        :func:`fit_reconstruction_models`.
    cir_parameters : CIRParameters
        Calibrated CIR parameter dataclass.
    include_squared : bool, optional
        Must match the flag used during feature construction and training.
        Default True.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed like *short_rate_series* with one column per
        reconstructed maturity (e.g. ``'yield_hat_10Y'``).

    Raises
    ------
    ValueError
        If *short_rate_series* is empty or *models* is empty.
    """
    if short_rate_series.empty:
        raise ValueError("short_rate_series must not be empty.")
    if not models:
        raise ValueError("models dict is empty; cannot reconstruct.")

    r = short_rate_series.to_numpy(dtype=float)
    result_frames: Dict[str, pd.Series] = {}

    for mat, model_info in models.items():
        model = model_info["model"]
        X = compute_cir_features(
            short_rate_series,
            mat,
            cir_parameters,
            include_squared=include_squared,
        )
        predicted_spread = model.predict(X.to_numpy())
        yield_hat = r + predicted_spread
        result_frames[f"yield_hat_{mat}"] = pd.Series(
            yield_hat, index=short_rate_series.index, name=f"yield_hat_{mat}"
        )

    return pd.DataFrame(result_frames)


# ---------------------------------------------------------------------------
# Section 8 – Artifact export
# ---------------------------------------------------------------------------

def save_reconstructed_yields(
    reconstructed: pd.DataFrame,
    path: Union[str, Path] = "results/reconstructed_yields.csv",
) -> Path:
    """Persist the reconstructed yield-curve DataFrame to CSV.

    Parameters
    ----------
    reconstructed : pd.DataFrame
        Output of :func:`reconstruct_yield_curve`.
    path : str or Path, optional
        Output file path.  Parent directories are created if absent.
        Default: ``'results/reconstructed_yields.csv'``.

    Returns
    -------
    Path
        Resolved path of the saved file.

    Raises
    ------
    ValueError
        If *reconstructed* is empty.
    OSError
        If the file cannot be written.
    """
    if reconstructed.empty:
        raise ValueError("reconstructed DataFrame is empty; nothing to save.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reconstructed.to_csv(path)
    logger.info("Reconstructed yields saved to %s", path.resolve())
    return path.resolve()


def save_models(
    models: Dict[str, Union[LinearRegression, Ridge]],
    directory: Union[str, Path] = "results/models",
) -> Dict[str, Path]:
    """Persist fitted per-maturity models using joblib.

    Each model is saved as ``<directory>/model_<maturity>.joblib``.

    Parameters
    ----------
    models : dict
        Fitted models keyed by maturity label.
    directory : str or Path, optional
        Directory in which model files are written.  Created if absent.
        Default: ``'results/models'``.

    Returns
    -------
    dict
        Dictionary mapping maturity label → resolved Path of saved file.

    Raises
    ------
    ValueError
        If *models* is empty.
    OSError
        If any model file cannot be written.
    """
    if not models:
        raise ValueError("models dict is empty; nothing to save.")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}
    for mat, model in models.items():
        filename = directory / f"model_{mat}.joblib"
        joblib.dump(model, filename)
        saved[mat] = filename.resolve()
        logger.info("Saved model for %s → %s", mat, filename.resolve())

    return saved


def load_models(
    maturities: Optional[tuple[str, ...]] = None,
    directory: Union[str, Path] = "results/models",
) -> Dict[str, Union[LinearRegression, Ridge]]:
    """Load previously persisted per-maturity models from disk.

    Parameters
    ----------
    maturities : tuple of str, optional
        Which maturities to load.  Defaults to
        :data:`RECONSTRUCTION_MATURITIES`.
    directory : str or Path, optional
        Directory containing model files.
        Default: ``'results/models'``.

    Returns
    -------
    dict
        Dictionary of loaded models keyed by maturity label.

    Raises
    ------
    FileNotFoundError
        If any expected model file is absent.
    """
    if maturities is None:
        maturities = RECONSTRUCTION_MATURITIES

    directory = Path(directory)
    models: Dict[str, Union[LinearRegression, Ridge]] = {}

    for mat in maturities:
        filename = directory / f"model_{mat}.joblib"
        if not filename.exists():
            raise FileNotFoundError(
                f"Model file not found for maturity '{mat}': {filename.resolve()}"
            )
        models[mat] = {
            "model": joblib.load(filename),
            "feature_names": None
            }
        logger.info("Loaded model for %s from %s", mat, filename.resolve())

    return models