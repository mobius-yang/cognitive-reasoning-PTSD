#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid OpenMP shared-memory issues in constrained environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DISABLE_SHM", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_SHM_DISABLE", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_PROC_BIND", "false")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling.core_utils import (
    calculate_slope,
    choose_threshold_by_f1,
    collapse_to_subject_level,
    drop_task1_columns_from_task2,
    pick_baseline_scale_columns,
    pick_stratified_cv,
    safe_std0,
)


def _select_followup_pcl_column(df: pd.DataFrame) -> str:
    for candidate in ["PCL_T3", "PCL_T2"]:
        if candidate in df.columns:
            return candidate
    raise KeyError("Missing follow-up PCL column: expected `PCL_T2` or `PCL_T3`.")


def _as_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _aggregate_task1_subject_level(
    df_task1: pd.DataFrame,
    *,
    dims: list[str],
    add_weighted_composite: bool,
    weight_dim: str | None,
    weight_value: float,
) -> pd.DataFrame:
    df = df_task1.copy()
    if "name" not in df.columns and "Name" in df.columns:
        df = df.rename(columns={"Name": "name"})
    if "name" not in df.columns:
        raise KeyError("Missing `name`/`Name` column in task1.")

    df["name"] = df["name"].astype(str)

    for col in set(dims) | {"temporal", "coherence", "reflection", "language", "perspective", "sensory", "arousal"}:
        if col in df.columns:
            df[col] = _as_float_series(df, col)

    if add_weighted_composite:
        weight_dim = str(weight_dim) if weight_dim is not None else None
        weight_value = float(weight_value)

        # Composite definitions follow the heuristic in `src/feature_extraction.py`.
        vam_dims = ["temporal", "coherence", "reflection", "language", "perspective"]
        sam_dims = ["sensory", "arousal"]

        wam = {d: 1.0 for d in vam_dims}
        wsa = {d: 1.0 for d in sam_dims}
        if weight_dim in wam:
            wam[weight_dim] = weight_value
        if weight_dim in wsa:
            wsa[weight_dim] = weight_value

        vam_denom = float(sum(wam.values()))
        sam_denom = float(sum(wsa.values()))

        df["VAM_index_w"] = sum(wam[d] * df[d] for d in vam_dims) / vam_denom
        df["SAM_index_w"] = sum(wsa[d] * df[d] for d in sam_dims) / sam_denom

    # Keep only columns that exist in the sheet.
    dims_present = [c for c in dims if c in df.columns]
    if add_weighted_composite:
        dims_present += [c for c in ["VAM_index_w", "SAM_index_w"] if c in df.columns]

    if not dims_present:
        raise ValueError("No usable task1 dimensions found; check task1_scored.xlsx columns.")

    agg_funcs = {feat: ["mean", safe_std0, calculate_slope] for feat in dims_present}
    df_agg = df.groupby("name").agg(agg_funcs)
    df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
    df_agg = df_agg.reset_index().rename(columns={"name": "Name"})

    feature_cols = [c for c in df_agg.columns if c != "Name"]
    df_agg[feature_cols] = df_agg[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df_agg


def _prepare_subject_level_dataset(df_task1: pd.DataFrame, df_task2: pd.DataFrame, df_task1_agg: pd.DataFrame) -> pd.DataFrame:
    df2 = df_task2.copy()
    if "Name" not in df2.columns and "name" in df2.columns:
        df2 = df2.rename(columns={"name": "Name"})
    if "Name" not in df2.columns:
        raise KeyError("Missing `Name`/`name` column in task2.")
    df2["Name"] = df2["Name"].astype(str)

    df_task1_for_drop = df_task1.copy()
    if "name" not in df_task1_for_drop.columns and "Name" in df_task1_for_drop.columns:
        df_task1_for_drop = df_task1_for_drop.rename(columns={"Name": "name"})
    df_task1_for_drop = df_task1_for_drop.rename(columns={"name": "Name"})

    df2_clean = drop_task1_columns_from_task2(df2, df_task1_for_drop)
    df2_subj = collapse_to_subject_level(df2_clean)

    df_merged = pd.merge(df2_subj, df_task1_agg, on="Name", how="inner")

    followup_col = _select_followup_pcl_column(df_merged)
    if "PCL_T1" not in df_merged.columns:
        raise KeyError("Missing baseline PCL column: expected `PCL_T1`.")

    threshold = -10
    df_merged["PCL_delta"] = df_merged[followup_col] - df_merged["PCL_T1"]
    df_merged["Is_Responder"] = (df_merged["PCL_delta"] <= threshold).astype(int)
    df_merged["Is_NonResponder"] = (1 - df_merged["Is_Responder"]).astype(int)

    df_final = df_merged.dropna(subset=["PCL_delta", "Is_NonResponder"]).copy()
    df_final["Is_NonResponder"] = df_final["Is_NonResponder"].astype(int)
    return df_final


@dataclass(frozen=True)
class EvalResult:
    seed: int
    experiment: str
    n_samples: int
    n_pos: int
    n_features_after_dummies: int
    auprc: float
    roc_auc: float
    brier: float
    balanced_accuracy: float
    f1: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int


def _eval_logreg_oof(df_final: pd.DataFrame, feature_cols: list[str], *, seed: int, experiment: str) -> EvalResult:
    X = pd.get_dummies(df_final[feature_cols].copy(), drop_first=True)
    y = df_final["Is_NonResponder"].astype(int)

    cv = pick_stratified_cv(y, max_splits=5, random_state=seed)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]
    )
    grid = GridSearchCV(
        pipe,
        param_grid={"clf__C": [0.1, 1.0, 10.0]},
        scoring="average_precision",
        cv=cv if cv is not None else 3,
        n_jobs=1,
        refit=True,
    )

    if cv is not None:
        y_prob = cross_val_predict(grid, X, y, cv=cv, method="predict_proba")[:, 1]
    else:
        grid.fit(X, y)
        y_prob = grid.predict_proba(X)[:, 1]

    thr = choose_threshold_by_f1(y.to_numpy(), y_prob)
    y_pred = (y_prob >= thr).astype(int)

    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, y_pred).ravel()]

    return EvalResult(
        seed=int(seed),
        experiment=str(experiment),
        n_samples=int(len(y)),
        n_pos=int(y.sum()),
        n_features_after_dummies=int(X.shape[1]),
        auprc=float(average_precision_score(y, y_prob)),
        roc_auc=float(roc_auc_score(y, y_prob)),
        brier=float(brier_score_loss(y, y_prob)),
        balanced_accuracy=float(balanced_accuracy_score(y, y_pred)),
        f1=float(f1_score(y, y_pred)),
        threshold=float(thr),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )


def _parse_csv_floats(s: str) -> list[float]:
    out: list[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _parse_csv_ints(s: str) -> list[int]:
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Ablation & sensitivity analysis for VAM/SAM heuristics")
    p.add_argument("--task1", default="data/task1_scored.xlsx")
    p.add_argument("--task2", default="data/task2_processing_data.xlsx")
    p.add_argument("--out-json", default="results/ablation_vam_sam_logreg.json")
    p.add_argument("--out-csv", default="results/ablation_vam_sam_logreg.csv")
    p.add_argument("--seeds", default="42", help="Comma-separated seeds, e.g. 42,43,44")
    p.add_argument(
        "--mode",
        default="pipeline",
        choices=["pipeline", "full"],
        help="pipeline: match current modeling pipeline raw dims; full: include language/perspective as raw dims too",
    )
    p.add_argument(
        "--weight-grid",
        default="1,1.5,2,3",
        help="Comma-separated weight grid for OAT sensitivity (applied to each component dim)",
    )
    p.add_argument(
        "--weight-dims",
        default="temporal,coherence,reflection,language,perspective,sensory,arousal",
        help="Comma-separated component dims to sweep (OAT).",
    )

    args = p.parse_args()

    seeds = _parse_csv_ints(args.seeds)
    weight_grid = _parse_csv_floats(args.weight_grid)
    weight_dims = [d.strip() for d in args.weight_dims.split(",") if d.strip()]

    df_task1 = pd.read_excel(args.task1)
    df_task2 = pd.read_excel(args.task2)

    if args.mode == "pipeline":
        raw_dims = ["temporal", "coherence", "reflection", "sensory", "arousal"]
    else:
        raw_dims = ["temporal", "coherence", "reflection", "sensory", "arousal", "language", "perspective"]

    composite_dims = ["VAM_index", "SAM_index"]

    started = time.time()
    all_rows: list[EvalResult] = []

    for seed in seeds:
        df_task1_agg = _aggregate_task1_subject_level(
            df_task1,
            dims=raw_dims + composite_dims,
            add_weighted_composite=False,
            weight_dim=None,
            weight_value=1.0,
        )
        df_final = _prepare_subject_level_dataset(df_task1, df_task2, df_task1_agg)

        baseline_cols = sorted(pick_baseline_scale_columns(df_final))
        raw_agg_cols = sorted([c for c in df_final.columns if any(c.startswith(f"{d}_") for d in raw_dims)])
        composite_agg_cols = sorted(
            [c for c in df_final.columns if c.startswith("VAM_index_") or c.startswith("SAM_index_")]
        )

        experiments: list[tuple[str, list[str]]] = [
            ("baseline_only", baseline_cols),
            ("raw_only(+baseline)", baseline_cols + raw_agg_cols),
            ("composite_only(+baseline)", baseline_cols + composite_agg_cols),
            ("raw+composite(+baseline)", baseline_cols + raw_agg_cols + composite_agg_cols),
        ]

        for exp_name, cols in experiments:
            all_rows.append(_eval_logreg_oof(df_final, cols, seed=seed, experiment=exp_name))

        for dim in weight_dims:
            for w in weight_grid:
                df_task1_agg_w = _aggregate_task1_subject_level(
                    df_task1,
                    dims=raw_dims + composite_dims,
                    add_weighted_composite=True,
                    weight_dim=dim,
                    weight_value=w,
                )
                df_final_w = _prepare_subject_level_dataset(df_task1, df_task2, df_task1_agg_w)

                baseline_cols_w = sorted(pick_baseline_scale_columns(df_final_w))
                comp_w_cols = sorted(
                    [c for c in df_final_w.columns if c.startswith("VAM_index_w_") or c.startswith("SAM_index_w_")]
                )
                all_rows.append(
                    _eval_logreg_oof(
                        df_final_w,
                        baseline_cols_w + comp_w_cols,
                        seed=seed,
                        experiment=f"composite_w({dim}={w})_only(+baseline)",
                    )
                )

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "task1": str(args.task1),
            "task2": str(args.task2),
            "mode": args.mode,
            "seeds": seeds,
            "weight_grid": weight_grid,
            "weight_dims": weight_dims,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(time.time() - started, 3),
        },
        "rows": [r.__dict__ for r in all_rows],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(EvalResult.__annotations__.keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r.__dict__)

    # Compact stdout summary.
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    print(
        f"Rows: {len(all_rows)} | Seeds: {len(seeds)} | Mode: {args.mode} | weight_dims: {weight_dims} | weight_grid: {weight_grid}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
