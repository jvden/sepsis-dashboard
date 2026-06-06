"""
Rule-based sepsis risk score (modified qSOFA + clinical indicators).

This is a transparent, interpretable scoring system — NOT a black-box ML model.
Each point is traceable to published clinical criteria, supporting explainability
and human oversight as required by the EU AI Act (Art. 13) and Isala's clinical
governance requirements.

Points:
  Vitals (qSOFA-basis):
    +1  Resp >= 22/min
    +1  SBP <= 100 mmHg
  Extended indicators:
    +1  Temp < 36.0 or > 38.3 °C
    +1  HR > 90 bpm
  Laboratory:
    +1  Lactate > 2.0 mmol/L   (tissue hypoperfusion)
    +1  WBC < 4 or > 12 *10^3  (SIRS criterion)

Score 0–2 = Laag risico (groen)
Score 3–4 = Matig risico (oranje)
Score 5–6 = Hoog risico (rood)
"""

import pandas as pd
import numpy as np

SCORE_COMPONENTS = {
    "Resp ≥ 22/min":        lambda r: _gt(r.get("Resp"), 22),
    "SBP ≤ 100 mmHg":       lambda r: _le(r.get("SBP"), 100),
    "Temp afwijkend":        lambda r: _temp_abnormal(r.get("Temp")),
    "HR > 90 bpm":           lambda r: _gt(r.get("HR"), 90),
    "Lactaat > 2 mmol/L":    lambda r: _gt(r.get("Lactate"), 2.0),
    "WBC afwijkend":         lambda r: _wbc_abnormal(r.get("WBC")),
}


def _gt(val, threshold):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    return int(val > threshold)


def _le(val, threshold):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    return int(val <= threshold)


def _temp_abnormal(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    return int(val < 36.0 or val > 38.3)


def _wbc_abnormal(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    return int(val < 4.0 or val > 12.0)


def compute_score_series(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Add risk score and component columns to a per-patient DataFrame."""
    df = patient_df.copy()
    for name, fn in SCORE_COMPONENTS.items():
        df[f"_sc_{name}"] = df.apply(fn, axis=1)
    score_cols = [c for c in df.columns if c.startswith("_sc_")]
    df["risk_score"] = df[score_cols].sum(axis=1)
    df["risk_level"] = df["risk_score"].apply(_level)
    df["risk_color"] = df["risk_score"].apply(_color)
    return df


def _level(score: float) -> str:
    if score <= 2:
        return "Laag"
    if score <= 4:
        return "Matig"
    return "Hoog"


def _color(score: float) -> str:
    if score <= 2:
        return "#28a745"
    if score <= 4:
        return "#fd7e14"
    return "#dc3545"


def score_contributions(row: pd.Series) -> dict[str, int]:
    """Return the score breakdown for a single patient row."""
    return {name: fn(row) for name, fn in SCORE_COMPONENTS.items()}


def risk_badge(score: float) -> str:
    level = _level(score)
    color = _color(score)
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-weight:bold">{level} ({int(score)}/6)</span>'
