"""A faithful visual explanation of sequential Bayesian review updating.

The normal-normal random-effects model involves two unknown parameters:
the overall mean difference mu and between-study SD tau.

For an addition of new studies, the *joint* posterior from the preceding
version is updated with their likelihood. Marginal densities for mu obtained
by fitting each complete snapshot are therefore identical to the sequential
result (given the same initial priors and model).

The ratio p(mu | current) / p(mu | previous) is an effective, relative
updating factor for mu, integrating over uncertainty about tau conditional
on mu and the preceding studies. It is not a standalone probability density
or a likelihood evaluated at a fixed tau.

A data correction replaces the erroneous contribution rather than adding
another study. The ratio of full posteriors is still a valid relative
revision factor, but it must not be interpreted as new-study evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.analysis import bayesian_density, fit_bayesian
from src.data import VERSIONS, snapshot


@dataclass(frozen=True)
class UpdatingStory:
    version: str
    previous_version: str | None
    kind: str
    added_studies: tuple[str, ...]
    corrected_studies: tuple[str, ...]
    study_count_before: int
    study_count_after: int
    prior_benefit: float
    posterior_benefit: float
    prior_relevant: float
    posterior_relevant: float
    prior_mean: float
    posterior_mean: float
    prior_tau_mean: float
    posterior_tau_mean: float
    curves: pd.DataFrame


def _difference_between_snapshots(before: pd.DataFrame, after: pd.DataFrame):
    """Identify genuine new records and revised outcome/variance inputs."""
    old = before.set_index("study_id")
    new = after.set_index("study_id")
    removed = set(old.index) - set(new.index)
    if removed:
        raise ValueError(f"Unexpected removed studies: {sorted(removed)}")
    added_ids = [sid for sid in new.index if sid not in old.index]
    revised_ids = []
    compare_cols = ["n_int", "mean_int", "sd_int", "n_ctrl", "mean_ctrl", "sd_ctrl"]
    for sid in old.index:
        if any(not np.isclose(float(old.at[sid, c]), float(new.at[sid, c]))
               for c in compare_cols):
            revised_ids.append(str(sid))
    added_names = tuple(str(new.at[sid, "author_year"]) for sid in added_ids)
    return added_names, tuple(revised_ids)


def build_updating_story(
    version: str,
    *,
    threshold: float = -2.0,
    mu_prior_sd: float = 5.0,
    tau_prior_sd: float = 2.0,
    points: int = 601,
) -> UpdatingStory:
    """Compute prior/posterior curves and the relative effective update factor.

    All snapshots are analysed with the *same original joint prior*.
    This is algebraically equivalent to sequentially updating the full
    p(mu, tau) posterior for additions; no study is counted more than once.
    """
    if version not in VERSIONS:
        raise ValueError(f"Unknown review version: {version}")
    if mu_prior_sd <= 0 or tau_prior_sd <= 0:
        raise ValueError("Prior standard deviations must be positive.")
    if points < 200:
        raise ValueError("Provide at least 200 plot points.")

    idx = VERSIONS.index(version)
    current = snapshot(version)
    previous_version = VERSIONS[idx - 1] if idx > 0 else None
    previous = snapshot(previous_version) if previous_version else None
    posterior = fit_bayesian(
        current, threshold=threshold,
        mu_prior_sd=mu_prior_sd, tau_prior_sd=tau_prior_sd,
    )

    # Focus on the posterior region and the adjustable relevance threshold.
    # We do NOT renormalise a truncated prior; both curves are valid densities.
    xmin = min(-5.5, threshold - 1.5, posterior.ci_low - 1.2)
    xmax = max(3.0, posterior.ci_high + 1.2)
    if previous is None:
        prior_benefit = float(norm.cdf(0, loc=0, scale=mu_prior_sd))
        prior_relevant = float(norm.cdf(threshold, loc=0, scale=mu_prior_sd))
        prior_mean = 0.0
        prior_tau_mean = float(tau_prior_sd * np.sqrt(2 / np.pi))
        added_names = tuple(str(x) for x in current["author_year"])
        corrected_ids: tuple[str, ...] = ()
        kind = "initial"
        before_count = 0
    else:
        before = fit_bayesian(
            previous, threshold=threshold,
            mu_prior_sd=mu_prior_sd, tau_prior_sd=tau_prior_sd,
        )
        prior_benefit = before.p_benefit
        prior_relevant = before.p_relevant
        prior_mean = before.mean
        prior_tau_mean = before.tau_mean
        xmin = min(xmin, before.ci_low - 1.2)
        xmax = max(xmax, before.ci_high + 1.2)
        added_names, corrected_ids = _difference_between_snapshots(previous, current)
        kind = ("mixed" if added_names and corrected_ids else
                "correction" if corrected_ids else
                "addition" if added_names else "no_change")
        before_count = len(previous)

    x = np.linspace(xmin, xmax, points)
    if previous is None:
        prior_density = norm.pdf(x, loc=0, scale=mu_prior_sd)
    else:
        prior_density = bayesian_density(
            previous, mu_prior_sd=mu_prior_sd, tau_prior_sd=tau_prior_sd,
            x_min=xmin, x_max=xmax, points=points,
        )["Posterior density"].to_numpy(float)

    posterior_density = bayesian_density(
        current, mu_prior_sd=mu_prior_sd, tau_prior_sd=tau_prior_sd,
        x_min=xmin, x_max=xmax, points=points,
    )["Posterior density"].to_numpy(float)

    # p(mu | current) / p(mu | previous) is proportional to the effective
    # incremental likelihood (or replacement factor in V4), *after* tau has
    # been integrated out with the previous conditional distribution.
    # Work in log space in the meaningful overlap to avoid underflow in
    # ratios of small densities. The resulting factor is peak-normalised.
    overlap = (
        (prior_density >= np.max(prior_density) * .0005)
        & (posterior_density >= np.max(posterior_density) * .0005)
        & (prior_density > 0) & (posterior_density > 0)
    )
    factor = np.full_like(prior_density, np.nan)
    if np.any(overlap):
        log_ratio = np.log(posterior_density[overlap]) - np.log(prior_density[overlap])
        factor[overlap] = np.exp(log_ratio - np.max(log_ratio))

    curves = pd.DataFrame({
        "MD": x,
        "Prior density": prior_density,
        "Posterior density": posterior_density,
        "Relative update factor": factor,
    })
    return UpdatingStory(
        version=version,
        previous_version=previous_version,
        kind=kind,
        added_studies=added_names,
        corrected_studies=corrected_ids,
        study_count_before=before_count,
        study_count_after=len(current),
        prior_benefit=float(prior_benefit),
        posterior_benefit=float(posterior.p_benefit),
        prior_relevant=float(prior_relevant),
        posterior_relevant=float(posterior.p_relevant),
        prior_mean=float(prior_mean),
        posterior_mean=float(posterior.mean),
        prior_tau_mean=float(prior_tau_mean),
        posterior_tau_mean=float(posterior.tau_mean),
        curves=curves,
    )
