# src/modeling/core_utils.py
"""Core utility functions for modeling tasks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_stratified_cv(y: pd.Series, max_splits: int, random_state: int):
    counts = y.value_counts(dropna=False)
    if counts.empty:
        return None

    min_class = int(counts.min())
    n_splits = min(max_splits, min_class)
    if n_splits < 2:
        return None

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def pick_kfold(n_samples: int, max_splits: int, random_state: int):
    n_splits = min(max_splits, n_samples)
    if n_splits < 2:
        return None

    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def drop_task1_columns_from_task2(df_task2: pd.DataFrame, df_task1: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated task1 columns that were merged into task2 before subject-level aggregation."""
    task1_cols = set(df_task1.columns)
    cols_to_drop = [c for c in df_task2.columns if (c in task1_cols) and (c != "Name")]
    return df_task2.drop(columns=cols_to_drop, errors="ignore")


def collapse_to_subject_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per subject by taking the first record per `Name`."""
    if "Name" not in df.columns:
        raise KeyError("Missing `Name` column; cannot aggregate to subject level.")
    return df.groupby("Name", as_index=False).first()


def pick_baseline_scale_columns(df: pd.DataFrame) -> list[str]:
    """Keep baseline (T1) scale features only, to avoid leakage from T2/T3."""
    baseline: list[str] = []
    for c in df.columns:
        if c == "Name":
            continue
        if re.search(r"_T2$|_T3$", c):
            continue
        if re.search(r"_(change|mean|std)$", c) and re.match(r"^(PCL|GAD|PHQ|SDQ|PCL_)", c):
            continue
        if re.search(r"_T1$", c) and re.match(r"^(PCL|GAD|PHQ|SDQ|PCL_)", c):
            baseline.append(c)
    return baseline


def build_X(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    X_raw = df[feature_cols].copy()
    X = pd.get_dummies(X_raw, drop_first=True)
    return X, list(X.columns)


def calculate_slope(series: pd.Series) -> float:
    y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])


def safe_std0(series: pd.Series) -> float:
    y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    return float(np.std(y, ddof=0))


def choose_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5
    f1s = 2 * precision[:-1] * recall[:-1] / np.clip((precision[:-1] + recall[:-1]), 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1s))])
