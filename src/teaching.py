"""Teaching-only scenarios for SomniTech: weights, repeated looks and monitoring.

All non-study monitoring counts and unpublished trials are explicitly invented.
None of these routines estimates publication bias or performs a real PRISMA audit.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src.analysis import MetaResult, calculate_effects, validate_studies


def compare_weights(frame: pd.DataFrame, fit: MetaResult) -> pd.DataFrame:
    """Common-effect inverse-variance weights vs fitted random-effects weights."""
    effects = calculate_effects(frame)
    inv_var = 1.0 / effects["vi"].to_numpy(dtype=float)
    common = 100.0 * inv_var / inv_var.sum()
    random = np.asarray(fit.weights_percent, dtype=float)
    if len(random) != len(effects):
        raise ValueError("The fitted model and study data have different lengths.")
    return pd.DataFrame({
        "Study": effects["author_year"].astype(str).to_numpy(),
        "Common-effect (%)": common,
        "Random-effects (%)": random,
        "Change (pp)": random - common,
    })


def bonferroni_limits(history: pd.DataFrame, family_alpha: float = .05) -> pd.DataFrame:
    """Demonstrate Bonferroni across the planned number of displayed looks.

    Not a validated prospective alpha-spending design; nominal modified-HK
    standard errors are held fixed and the critical t quantile is changed.
    """
    if not (0 < family_alpha < 1):
        raise ValueError("family_alpha must lie between zero and one.")
    required = {"version", "k", "mean", "se_hk", "ci_low", "ci_high"}
    if not required.issubset(history.columns):
        raise ValueError("History is missing the SE required for adjusted limits.")
    n_looks = len(history)
    if n_looks < 1 or (history["k"] < 2).any():
        raise ValueError("At least one look with two studies is required.")
    out = history.copy()
    critical = student_t.ppf(1.0 - family_alpha / (2.0 * n_looks),
                             out["k"].to_numpy(float) - 1.0)
    margin = critical * out["se_hk"].to_numpy(float)
    out["adjusted_low"] = out["mean"] - margin
    out["adjusted_high"] = out["mean"] + margin
    return out


def add_file_drawer_trials(frame: pd.DataFrame) -> pd.DataFrame:
    """Add exactly two hypothetical, previously unseen, null-effect RCTs.

    Idempotent for a second button click. These are not recovered real trials.
    """
    base = frame.copy()
    if "include" not in base:
        base["include"] = True
    existing = set(base["study_id"].astype(str))
    added = []
    for idx in (1, 2):
        sid = f"FD-{idx:02d}"
        if sid in existing:
            continue
        row = {column: np.nan for column in base.columns}
        row.update({
            "study_id": sid,
            "author_year": f"Hypothetical unpublished null trial {idx}",
            "n_int": 500, "n_ctrl": 500,
            "mean_int": 12.0, "mean_ctrl": 12.0,
            "sd_int": 4.0, "sd_ctrl": 4.0,
            "rob": "Some concerns", "phase": 4, "include": True,
        })
        added.append(row)
    if added:
        base = pd.concat([base, pd.DataFrame(added)], ignore_index=True)
    return validate_studies(base)


def load_illustrative_monitoring() -> pd.DataFrame:
    """Synthetic screening activity, NOT an actual search or PRISMA flow."""
    path = Path(__file__).resolve().parents[1] / "data" / "monitoring.csv"
    df = pd.read_csv(path)
    for key in ("identified", "screened", "full_texts", "included"):
        df[key] = df[key].astype(int)
    if (df[["identified", "screened", "full_texts", "included"]] < 0).any().any():
        raise ValueError("Screening counts must be non-negative.")
    if not ((df["identified"] >= df["screened"])
            & (df["screened"] >= df["full_texts"])
            & (df["full_texts"] >= df["included"])).all():
        raise ValueError("Illustrative PRISMA-like counts are inconsistent.")
    return df
