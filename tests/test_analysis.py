"""Regression tests for analysis and as-known-at-time versioning."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import calculate_effects, fit_bayesian, fit_random_effects, validate_studies
from src.data import VERSIONS, snapshot

EXPECTED = {
    "V0": (4, -1.097395, -2.478731, 0.283941),
    "V1": (6, -1.690537, -2.743725, -0.637349),
    "V2": (9, -0.830461, -1.785264, 0.124343),
    "V3": (13, -0.597370, -1.249715, 0.054976),
    "V4": (13, -0.444784, -0.948689, 0.059121),
}


@pytest.mark.parametrize("version", VERSIONS)
def test_frequentist_reference_values(version):
    k, mu, lo, hi = EXPECTED[version]
    fit = fit_random_effects(snapshot(version))
    assert fit.k == k
    assert fit.mean == pytest.approx(mu, abs=1e-4)
    assert fit.ci_low == pytest.approx(lo, abs=1e-4)
    assert fit.ci_high == pytest.approx(hi, abs=1e-4)
    assert np.isclose(fit.weights_percent.sum(), 100)
    assert fit.tau2 >= 0
    assert fit.ci_low <= fit.mean <= fit.ci_high


def test_correction_only_applies_to_v4():
    v3 = snapshot("V3").set_index("study_id")
    v4 = snapshot("V4").set_index("study_id")
    assert v3.loc["S06", "mean_int"] == 10.2
    assert v4.loc["S06", "mean_int"] == 11.3
    assert v3.loc["S06", "sd_int"] == 3.1
    assert v4.loc["S06", "sd_int"] == 3.4
    assert set(v4.index) == set(v3.index)
    assert v3.drop(index="S06")[["mean_int", "sd_int"]].equals(
        v4.drop(index="S06")[["mean_int", "sd_int"]])


def test_md_variance_formula():
    row = calculate_effects(snapshot("V0")).iloc[0]
    assert row["yi"] == pytest.approx(-1.7)
    assert row["vi"] == pytest.approx(4.1**2/35 + 4.3**2/34)


def test_bayesian_probabilities_and_order():
    b1 = fit_bayesian(snapshot("V1"))
    b4 = fit_bayesian(snapshot("V4"))
    assert b1.ci_low < b1.mean < b1.ci_high
    assert 0 <= b1.p_relevant <= b1.p_benefit <= 1
    assert b1.p_relevant > b4.p_relevant
    assert b1.tau_mean > 0
    assert b1.mean == pytest.approx(-1.6552, abs=.02)


def test_input_validation_rejects_zero_sd_and_repeated_id():
    data = snapshot("V0")
    bad = data.copy()
    bad.loc[0, "sd_int"] = 0
    with pytest.raises(ValueError):
        validate_studies(bad)
    bad = data.copy()
    bad.loc[1, "study_id"] = bad.loc[0, "study_id"]
    with pytest.raises(ValueError):
        validate_studies(bad)


def test_experiment_is_separate_from_canonical_data():
    original = snapshot("V1")
    modified = original.copy()
    modified.loc[0, "mean_int"] = -4.0
    assert fit_random_effects(modified).mean != pytest.approx(fit_random_effects(original).mean)
    assert snapshot("V1").loc[0, "mean_int"] == 12.4
