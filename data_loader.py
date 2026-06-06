"""Data loading and preprocessing for the Sepsis Monitor dashboard."""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

_FULL_DATA = Path(__file__).parent.parent / "vital-ai" / "data" / "Prediction-of-sepsis" / "train_data.csv"
_SAMPLE_DATA = Path(__file__).parent / "sample_data.csv"
DATA_PATH = _FULL_DATA if _FULL_DATA.exists() else _SAMPLE_DATA

VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
    "Creatinine", "Glucose", "Lactate", "Magnesium",
    "Phosphate", "Potassium", "WBC", "Hct", "Hgb",
    "PTT", "Platelets", "Bilirubin_total", "Fibrinogen",
]
DEMO_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
LABEL_COL = "SepsisLabel"

VITAL_RANGES = {
    "HR":    {"unit": "bpm",   "normal": (60, 100),  "warn": (50, 120),  "label": "Hartfrequentie"},
    "O2Sat": {"unit": "%",     "normal": (95, 100),  "warn": (90, 100),  "label": "SpO₂"},
    "Temp":  {"unit": "°C",    "normal": (36.1, 38.0), "warn": (35.0, 39.0), "label": "Temperatuur"},
    "SBP":   {"unit": "mmHg",  "normal": (90, 140),  "warn": (80, 160),  "label": "Systolische BD"},
    "MAP":   {"unit": "mmHg",  "normal": (70, 100),  "warn": (60, 110),  "label": "MAP"},
    "DBP":   {"unit": "mmHg",  "normal": (60, 90),   "warn": (50, 100),  "label": "Diastolische BD"},
    "Resp":  {"unit": "/min",  "normal": (12, 20),   "warn": (10, 24),   "label": "Ademfrequentie"},
}


@st.cache_data(show_spinner="Gegevens laden…", ttl=3600)
def load_sample(n_patients: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Load a reproducible sample of patients for the dashboard."""
    df = pd.read_csv(DATA_PATH, dtype={"Patient_ID": str})
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Determine per-patient sepsis status
    df["is_sepsis"] = df.groupby("Patient_ID")[LABEL_COL].transform("max")

    # If using the bundled sample file, skip re-sampling (already 600 patients)
    n_unique = df["Patient_ID"].nunique()
    if n_unique <= n_patients:
        df["Unit"] = df.apply(
            lambda r: "MICU" if r["Unit1"] == 1 else ("SICU" if r["Unit2"] == 1 else "Overig"),
            axis=1,
        )
        return df.sort_values(["Patient_ID", "Hour"]).reset_index(drop=True)

    # Stratified sample: half sepsis, half non-sepsis
    half = n_patients // 2
    rng = np.random.default_rng(random_state)
    sepsis_patients = df.groupby("Patient_ID")[LABEL_COL].max()

    sep_ids = sepsis_patients[sepsis_patients == 1].index.tolist()
    non_ids = sepsis_patients[sepsis_patients == 0].index.tolist()

    sample_sep = rng.choice(sep_ids, size=min(half, len(sep_ids)), replace=False)
    sample_non = rng.choice(non_ids, size=min(half, len(non_ids)), replace=False)
    selected = list(sample_sep) + list(sample_non)

    df = df[df["Patient_ID"].isin(selected)].copy()
    df["Unit"] = df.apply(
        lambda r: "MICU" if r["Unit1"] == 1 else ("SICU" if r["Unit2"] == 1 else "Overig"),
        axis=1,
    )
    df = df.sort_values(["Patient_ID", "Hour"]).reset_index(drop=True)
    return df


def latest_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent recorded row per patient."""
    return (
        df.sort_values("Hour")
        .groupby("Patient_ID")
        .last()
        .reset_index()
    )


def missing_rate(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Fraction of missing values per column, across the dataset."""
    return df[cols].isna().mean().sort_values(ascending=False)


def sepsis_onset_hour(df: pd.DataFrame) -> dict[str, int | None]:
    """Return the first ICULOS hour where SepsisLabel == 1 per patient."""
    onset = (
        df[df[LABEL_COL] == 1]
        .groupby("Patient_ID")["ICULOS"]
        .min()
    )
    return onset.to_dict()
