"""Regression checks for the educational sequential Bayesian update."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from src.analysis import fit_bayesian
from src.data import VERSIONS, snapshot
from src.updating import build_updating_story


@pytest.mark.parametrize(
    "version,previous,k_before,k_after,n_added,n_corrected",
    [
        ("V0", None, 0, 4, 4, 0),
        ("V1", "V0", 4, 6, 2, 0),
        ("V2", "V1", 6, 9, 3, 0),
        ("V3", "V2", 9, 13, 4, 0),
        ("V4", "V3", 13, 13, 0, 1),
    ],
)
def test_new_studies_and_revisions_are_identified(
    version, previous, k_before, k_after, n_added, n_corrected
):
    story = build_updating_story(version, points=251)
    assert story.version == version
    assert story.previous_version == previous
    assert story.study_count_before == k_before
    assert story.study_count_after == k_after
    assert len(story.added_studies) == n_added
    assert len(story.corrected_studies) == n_corrected
    assert 0 <= story.posterior_relevant <= story.posterior_benefit <= 1
    assert 0 <= story.prior_relevant <= story.prior_benefit <= 1
    assert np.isfinite(story.curves[["Prior density", "Posterior density"]]).all().all()


@pytest.mark.parametrize("version", VERSIONS)
def test_posterior_probabilities_reproduce_full_snapshot_fit(version):
    story = build_updating_story(version, threshold=-1.5, points=251)
    reference = fit_bayesian(snapshot(version), threshold=-1.5)
    assert story.posterior_mean == pytest.approx(reference.mean, abs=1e-10)
    assert story.posterior_benefit == pytest.approx(reference.p_benefit, abs=1e-10)
    assert story.posterior_relevant == pytest.approx(reference.p_relevant, abs=1e-10)


def test_initial_prior_is_not_recycled_posterior():
    story = build_updating_story("V0", mu_prior_sd=5, threshold=-2)
    assert story.prior_mean == 0
    assert story.prior_benefit == pytest.approx(.5)
    assert story.prior_relevant == pytest.approx(norm.cdf(-2 / 5), abs=1e-9)


@pytest.mark.parametrize("version", ("V1", "V2", "V3", "V4"))
def test_previous_posterior_is_prior_for_next_step(version):
    story = build_updating_story(version, points=251)
    prior_fit = fit_bayesian(snapshot(story.previous_version))
    assert story.prior_mean == pytest.approx(prior_fit.mean, abs=1e-10)
    assert story.prior_benefit == pytest.approx(prior_fit.p_benefit, abs=1e-10)
    assert story.prior_relevant == pytest.approx(prior_fit.p_relevant, abs=1e-10)


@pytest.mark.parametrize("version", VERSIONS)
def test_relative_update_factor_reproduces_posterior_shape(version):
    story = build_updating_story(version, points=451)
    curves = story.curves
    mask = curves["Relative update factor"].notna()
    assert mask.sum() > 20
    raw_reconstructed = (
        curves.loc[mask, "Prior density"].to_numpy()
        * curves.loc[mask, "Relative update factor"].to_numpy()
    )
    posterior = curves.loc[mask, "Posterior density"].to_numpy()
    # The factor is rescaled for its own secondary axis; therefore compare
    # normalized shapes, not their absolute vertical positions.
    assert np.allclose(raw_reconstructed / raw_reconstructed.max(),
                       posterior / posterior.max(), rtol=1e-9, atol=1e-9)


def test_v4_is_a_replacement_not_an_extra_study():
    story = build_updating_story("V4", points=251)
    assert story.kind == "correction"
    assert story.corrected_studies == ("S06",)
    assert story.added_studies == ()
    assert story.study_count_before == story.study_count_after == 13


def test_prior_settings_propagate_through_update():
    story_a = build_updating_story("V1", mu_prior_sd=1.0, tau_prior_sd=0.5, points=251)
    story_b = build_updating_story("V1", mu_prior_sd=10.0, tau_prior_sd=4.0, points=251)
    assert abs(story_a.posterior_mean - story_b.posterior_mean) > 0.001
