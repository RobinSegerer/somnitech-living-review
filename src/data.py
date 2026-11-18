"""Versioned, reproducible snapshots of the fictional SomniTech dataset."""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
VERSIONS = ("V0", "V1", "V2", "V3", "V4")
VERSION_METADATA = {
    "V0": {"date": "2024-06-20", "title": "Initial Review", "description": "Four small RCTs; the evidence remains uncertain.", "type": "Initial publication"},
    "V1": {"date": "2025-07-18", "title": "New evidence", "description": "Two additional RCTs, including SOMNI-TRIAL, strengthen the estimated benefit.", "type": "New studies"},
    "V2": {"date": "2026-09-22", "title": "Contradictory evidence", "description": "Three large trials weaken the pooled estimate and increase heterogeneity.", "type": "Contradictory evidence"},
    "V3": {"date": "2027-07-15", "title": "Further evidence", "description": "Four additional RCTs reduce the pooled effect further.", "type": "New studies"},
    "V4": {"date": "2027-10-10", "title": "Data correction", "description": "Correction of SOMNI-TRIAL's primary outcome without adding a new study.", "type": "Data correction"},
}


def load_studies() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / "data" / "studies.csv", encoding="utf-8")


def load_events() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / "data" / "events.csv", encoding="utf-8").fillna("")


def snapshot(version: str, studies: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the *as-known-at-version* data; revisions apply from V4 onward."""
    if version not in VERSIONS:
        raise ValueError(f"Unknown version: {version}")
    if studies is None:
        studies = load_studies()
    current_phase = int(version[-1])
    frame = studies.loc[studies["phase"] <= min(current_phase, 3)].copy()
    if version == "V4":
        mask = frame["study_id"].eq("S06")
        frame.loc[mask, "mean_int"] = frame.loc[mask, "revised_mean_int"]
        frame.loc[mask, "sd_int"] = frame.loc[mask, "revised_sd_int"]
    return frame.reset_index(drop=True)
