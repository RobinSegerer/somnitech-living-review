"""Frequentist and Bayesian normal-normal random-effects meta-analysis.

All effects are raw mean differences (intervention minus control); negative
values favour intervention. Illustrative educational use; no clinical evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp, ndtr
from scipy.stats import chi2, norm, t as student_t

REQUIRED_COLUMNS = (
    "study_id", "author_year", "n_int", "mean_int", "sd_int",
    "n_ctrl", "mean_ctrl", "sd_ctrl",
)
NUMERICAL_COLUMNS = ("n_int", "mean_int", "sd_int", "n_ctrl", "mean_ctrl", "sd_ctrl")


@dataclass(frozen=True)
class MetaResult:
    k: int
    n_total: int
    mean: float
    ci_low: float
    ci_high: float
    se_hk: float
    tau2: float
    tau: float
    i2: float
    q: float
    q_pvalue: float
    weights_percent: np.ndarray
    pi_low: float | None = None
    pi_high: float | None = None
    method: str = "REML + modified Hartung–Knapp (ad hoc)"


@dataclass(frozen=True)
class BayesResult:
    mean: float
    median: float
    ci_low: float
    ci_high: float
    p_benefit: float
    p_relevant: float
    tau_mean: float
    mu_prior_sd: float
    tau_prior_sd: float
    threshold: float
    pred_ci_low: float
    pred_ci_high: float
    p_pred_benefit: float
    p_pred_relevant: float


def validate_studies(frame: pd.DataFrame, min_studies: int = 2) -> pd.DataFrame:
    """Return a copy with typed study-level data; reject incomplete/invalid input."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {', '.join(sorted(missing))}")
    out = frame.copy()
    if len(out) < min_studies:
        raise ValueError(f"At least {min_studies} complete studies are required.")
    for col in ("study_id", "author_year"):
        if out[col].isna().any() or out[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"'{col}' must not contain missing values.")
    if out["study_id"].duplicated().any():
        raise ValueError("'study_id' must be unique for each study.")
    for col in NUMERICAL_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if not np.isfinite(out[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{col}' contains missing or invalid values.")
    for col in ("n_int", "n_ctrl"):
        values = out[col].to_numpy(float)
        if (values < 2).any() or not np.all(values == np.floor(values)):
            raise ValueError(f"'{col}' must be an integer ≥ 2.")
        out[col] = out[col].astype(int)
    for col in ("mean_int", "mean_ctrl"):
        if not out[col].between(0, 28).all():
            raise ValueError(f"'{col}' must be within the ISI scale (0–28).")
    for col in ("sd_int", "sd_ctrl"):
        if (out[col] <= 0).any():
            raise ValueError(f"'{col}' must be positive.")
    return out.reset_index(drop=True)


def calculate_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Study-level endpoint mean difference and its sampling variance."""
    result = validate_studies(frame)
    result["yi"] = result["mean_int"] - result["mean_ctrl"]
    result["vi"] = result["sd_int"] ** 2 / result["n_int"] + result["sd_ctrl"] ** 2 / result["n_ctrl"]
    result["se"] = np.sqrt(result["vi"])
    result["ci_low"] = result["yi"] - norm.ppf(.975) * result["se"]
    result["ci_high"] = result["yi"] + norm.ppf(.975) * result["se"]
    return result


def _reml_nll(tau2: float, yi: np.ndarray, vi: np.ndarray) -> float:
    w = 1.0 / (vi + tau2)
    wsum = np.sum(w)
    mu = np.sum(w * yi) / wsum
    q = np.sum(w * (yi - mu) ** 2)
    return .5 * (np.sum(np.log(vi + tau2)) + np.log(wsum) + q)


def fit_random_effects(frame: pd.DataFrame) -> MetaResult:
    """REML for tau², with ad-hoc variance floor and t(k-1) critical value.

    The Hartung–Knapp variance is max(q_RE/(k-1), 1) / sum(w), to avoid
    an interval narrower than the unadjusted random-effects standard error.
    I² is the conventional Cochran-Q statistic based estimate.
    """
    effects = calculate_effects(frame)
    yi = effects["yi"].to_numpy(float)
    vi = effects["vi"].to_numpy(float)
    k = len(yi)
    w_fe = 1.0 / vi
    mu_fe = np.sum(w_fe * yi) / np.sum(w_fe)
    q_fe = float(np.sum(w_fe * (yi - mu_fe) ** 2))
    i2 = max(0.0, (q_fe - (k - 1)) / q_fe) * 100.0 if q_fe > 0 else 0.0

    # The likelihood is one-dimensional and can be minimized robustly on [0,∞).
    # An adaptive upper bound avoids silently truncating the optimum.
    upper = max(1.0, 10.0 * float(np.var(yi)), float(np.max(vi)))
    for _ in range(8):
        opt = minimize_scalar(_reml_nll, bounds=(0.0, upper),
                              args=(yi, vi), method="bounded",
                              options={"xatol": 1e-12})
        if opt.x < upper * .95:
            break
        upper *= 4.0
    tau2 = max(0.0, float(opt.x))
    if _reml_nll(0.0, yi, vi) <= _reml_nll(tau2, yi, vi):
        tau2 = 0.0
    w = 1.0 / (vi + tau2)
    mu = float(np.sum(w * yi) / np.sum(w))
    q_re = float(np.sum(w * (yi - mu) ** 2))
    se = np.sqrt(max(1.0, q_re / (k - 1)) / np.sum(w))
    margin = float(student_t.ppf(.975, df=k - 1) * se)
    # An approximate true-effect prediction interval, only when k >= 3.
    # With k=2 modified HK uses t(1) ≈ 12.71, so the CI can be very wide.
    pi_margin = (float(student_t.ppf(.975, df=k - 2) * np.sqrt(tau2 + se**2))
                 if k >= 3 else None)
    return MetaResult(
        k=k,
        n_total=int(effects["n_int"].sum() + effects["n_ctrl"].sum()),
        mean=mu, ci_low=mu - margin, ci_high=mu + margin,
        se_hk=float(se), tau2=tau2, tau=np.sqrt(tau2), i2=i2,
        q=q_fe, q_pvalue=float(chi2.sf(q_fe, df=k - 1)),
        weights_percent=100 * w / np.sum(w),
        pi_low=mu - pi_margin if pi_margin is not None else None,
        pi_high=mu + pi_margin if pi_margin is not None else None,
    )


@dataclass(frozen=True)
class TauGrid:
    """Shared quadrature for p(tau | data) and p(mu | data, tau)."""
    tau: np.ndarray
    weight: np.ndarray
    mu: np.ndarray
    sd_mu: np.ndarray


def _bayesian_tau_grid(frame: pd.DataFrame, mu_prior_sd: float,
                       tau_prior_sd: float, grid_size: int = 2001) -> TauGrid:
    """Analytically integrate mu, then integrate tau on a quadratic grid.

    Nonuniform trapezoidal weights preserve integration accuracy near tau=0
    even if the maximum plausible heterogeneity is large.
    """
    if not (np.isfinite(mu_prior_sd) and np.isfinite(tau_prior_sd)
            and mu_prior_sd > 0 and tau_prior_sd > 0):
        raise ValueError("Prior standard deviations must be positive and finite.")
    if grid_size < 401:
        raise ValueError("At least 401 grid points are required for integration.")
    effects = calculate_effects(frame)
    yi, vi = effects["yi"].to_numpy(float), effects["vi"].to_numpy(float)
    max_tau = max(10.0, 9.0 * tau_prior_sd, np.ptp(yi) * 5.0)
    taus = max_tau * np.linspace(0, 1, grid_size)**2
    vtot = vi[None, :] + taus[:, None] ** 2
    w = 1.0 / vtot
    precision = w.sum(axis=1) + 1.0 / mu_prior_sd**2
    b = (w * yi[None, :]).sum(axis=1)
    mu_cond = b / precision
    sigma_cond = np.sqrt(1.0 / precision)
    log_evidence = -0.5 * (
        len(yi) * np.log(2 * np.pi)
        + np.log(vtot).sum(axis=1)
        + np.log(mu_prior_sd**2 * precision)
        + (w * yi[None, :] ** 2).sum(axis=1)
        - b**2 / precision
    )
    log_prior_tau = np.log(np.sqrt(2 / np.pi) / tau_prior_sd) - .5 * (taus / tau_prior_sd)**2
    delta = np.diff(taus)
    traps = np.empty(grid_size)
    traps[0], traps[-1] = delta[0] / 2, delta[-1] / 2
    traps[1:-1] = (delta[:-1] + delta[1:]) / 2
    logweight = log_evidence + log_prior_tau + np.log(traps)
    normalized = np.exp(logweight - logsumexp(logweight))
    return TauGrid(taus, normalized, mu_cond, sigma_cond)


def _mixture_cdf(x: float, centers: np.ndarray, sigmas: np.ndarray,
                 weights: np.ndarray) -> float:
    return float(np.sum(weights * ndtr((x - centers) / sigmas)))


def _mixture_quantile(prob: float, centers: np.ndarray, sigmas: np.ndarray,
                      weights: np.ndarray) -> float:
    left = float(np.min(centers - 15 * sigmas))
    right = float(np.max(centers + 15 * sigmas))
    return float(brentq(
        lambda x: _mixture_cdf(x, centers, sigmas, weights) - prob, left, right))


def fit_bayesian(frame: pd.DataFrame, threshold: float = -2.0,
                 mu_prior_sd: float = 5.0, tau_prior_sd: float = 2.0,
                 grid_size: int = 2001) -> BayesResult:
    """Bayesian random-effects posterior and true-effect predictive distribution.

    theta_new describes the true effect in a new implementation, not a noisy
    future study estimate. Includes heterogeneity and parameter uncertainty.
    """
    grid = _bayesian_tau_grid(frame, mu_prior_sd, tau_prior_sd, grid_size)
    pred_sd = np.sqrt(grid.sd_mu**2 + grid.tau**2)
    cdf_mu = lambda x: _mixture_cdf(x, grid.mu, grid.sd_mu, grid.weight)
    cdf_pred = lambda x: _mixture_cdf(x, grid.mu, pred_sd, grid.weight)
    return BayesResult(
        mean=float(np.sum(grid.weight * grid.mu)),
        median=_mixture_quantile(.5, grid.mu, grid.sd_mu, grid.weight),
        ci_low=_mixture_quantile(.025, grid.mu, grid.sd_mu, grid.weight),
        ci_high=_mixture_quantile(.975, grid.mu, grid.sd_mu, grid.weight),
        p_benefit=cdf_mu(0), p_relevant=cdf_mu(threshold),
        tau_mean=float(np.sum(grid.weight * grid.tau)),
        mu_prior_sd=mu_prior_sd, tau_prior_sd=tau_prior_sd,
        threshold=threshold,
        pred_ci_low=_mixture_quantile(.025, grid.mu, pred_sd, grid.weight),
        pred_ci_high=_mixture_quantile(.975, grid.mu, pred_sd, grid.weight),
        p_pred_benefit=cdf_pred(0), p_pred_relevant=cdf_pred(threshold),
    )


def bayesian_density(frame: pd.DataFrame, mu_prior_sd: float = 5.0,
                     tau_prior_sd: float = 2.0, x_min: float | None = None,
                     x_max: float | None = None, points: int = 450,
                     threshold: float = -2.0) -> pd.DataFrame:
    """Marginal posterior for mu; adaptive bounds show credible tails."""
    grid = _bayesian_tau_grid(frame, mu_prior_sd, tau_prior_sd)
    if points < 2:
        raise ValueError("At least two density points are required.")
    if x_min is None or x_max is None:
        left = _mixture_quantile(.001, grid.mu, grid.sd_mu, grid.weight)
        right = _mixture_quantile(.999, grid.mu, grid.sd_mu, grid.weight)
        span = max(1.0, right - left)
        if x_min is None:
            x_min = min(left - span * .1, threshold, 0.0)
        if x_max is None:
            x_max = max(right + span * .1, 0.0, threshold)
    if not (np.isfinite(x_min) and np.isfinite(x_max) and x_min < x_max):
        raise ValueError("Density axis bounds must be finite and increasing.")
    x = np.linspace(x_min, x_max, points)
    density = np.sum(grid.weight[:, None] * norm.pdf(
        x[None, :], grid.mu[:, None], grid.sd_mu[:, None]), axis=0)
    return pd.DataFrame({"MD": x, "Posterior density": density})
