"""Tests for the explicitly hypothetical Living Evidence teaching activities."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import fit_random_effects
from src.data import VERSIONS, snapshot
from src.teaching import (
    add_file_drawer_trials, bonferroni_limits,
    compare_weights, load_illustrative_monitoring,
)


def test_common_and_random_effects_weights_sum_to_100():
    frame = snapshot("V2")
    result = fit_random_effects(frame)
    comparison = compare_weights(frame, result)
    assert len(comparison) == 9
    assert comparison["Common-effect (%)"].sum() == pytest.approx(100)
    assert comparison["Random-effects (%)"].sum() == pytest.approx(100)
    assert comparison["Change (pp)"].sum() == pytest.approx(0, abs=1e-8)
    assert result.tau2 > 0
    assert comparison["Common-effect (%)"].max() > comparison["Random-effects (%)"].max()


def test_weight_comparison_matches_fitted_random_effects_weights():
    for version in VERSIONS:
        data = snapshot(version)
        fit = fit_random_effects(data)
        weights = compare_weights(data, fit)
        np.testing.assert_allclose(weights["Random-effects (%)"], fit.weights_percent)


def test_bonferroni_intervals_are_wider_and_keep_original_centres():
    rows = []
    for version in VERSIONS:
        fit = fit_random_effects(snapshot(version))
        rows.append({
            "version": version, "k": fit.k, "mean": fit.mean,
            "se_hk": fit.se_hk, "ci_low": fit.ci_low, "ci_high": fit.ci_high,
        })
    history = pd.DataFrame(rows)
    adjusted = bonferroni_limits(history)
    np.testing.assert_allclose(adjusted["mean"], history["mean"])
    assert (adjusted["adjusted_low"] < history["ci_low"]).all()
    assert (adjusted["adjusted_high"] > history["ci_high"]).all()


def test_bonferroni_rejects_missing_se_or_invalid_alpha():
    with pytest.raises(ValueError, match="SE"):
        bonferroni_limits(pd.DataFrame({"k": [4]}))
    with pytest.raises(ValueError, match="family_alpha"):
        bonferroni_limits(pd.DataFrame({"k": [4]}), family_alpha=0)


def test_file_drawer_creates_two_null_studies_without_touching_original():
    original = snapshot("V1").assign(include=True)
    augmented = add_file_drawer_trials(original)
    assert len(augmented) == len(original) + 2
    assert set(augmented["study_id"]) - set(original["study_id"]) == {"FD-01", "FD-02"}
    assert original["study_id"].str.startswith("FD-").sum() == 0
    fabricated = augmented[augmented["study_id"].str.startswith("FD-")]
    assert (fabricated["include"] == True).all()
    assert (fabricated["n_int"] == 500).all()
    assert (fabricated["n_ctrl"] == 500).all()
    assert (fabricated["mean_int"] == fabricated["mean_ctrl"]).all()
    assert fit_random_effects(augmented).mean != pytest.approx(
        fit_random_effects(original).mean
    )


def test_file_drawer_button_is_idempotent_and_include_column_is_preserved():
    once = add_file_drawer_trials(snapshot("V0").assign(include=True))
    twice = add_file_drawer_trials(once)
    assert len(twice) == len(once)
    assert len(once.loc[once["study_id"].str.startswith("FD-")]) == 2
    assert twice["include"].dtype == bool


def test_monitoring_counts_are_internally_consistent_and_not_real_data():
    df = load_illustrative_monitoring()
    assert set(df["version"]) == set(VERSIONS)
    assert df["included"].sum() == 13
    assert df.loc[df["version"] == "V4", "screened"].sum() == 0
    assert df.loc[df["version"] == "V4", "included"].sum() == 0
    assert (df["identified"] >= df["screened"]).all()
    assert (df["screened"] >= df["full_texts"]).all()
    assert (df["full_texts"] >= df["included"]).all()
    assert ((df["version"] == "V0") & (df["included"] == 0)).any()
