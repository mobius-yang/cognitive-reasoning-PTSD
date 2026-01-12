#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

# Avoid OpenMP shared-memory issues in constrained environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DISABLE_SHM", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_SHM_DISABLE", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_PROC_BIND", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt


def _try_set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _select_followup_pcl_column(df: pd.DataFrame) -> str:
    for candidate in ["PCL_T3", "PCL_T2"]:
        if candidate in df.columns:
            return candidate
    raise KeyError("Missing follow-up PCL column: expected `PCL_T2` or `PCL_T3`.")


def _aggregate_task1_features(df_task1: pd.DataFrame) -> pd.DataFrame:
    from src.modeling.core_utils import calculate_slope, safe_std0
    from src.modeling.feature_mining import mine_advanced_features
    from src.modeling.time_series import extract_suds_dynamics_features

    df = df_task1.copy()
    if "name" not in df.columns and "Name" in df.columns:
        df = df.rename(columns={"Name": "name"})
    if "name" not in df.columns:
        raise KeyError("Missing `name` column in task1.")

    df["name"] = df["name"].astype(str)

    base_features = [
        "VAM_index",
        "SAM_index",
        "temporal",
        "coherence",
        "reflection",
        "sensory",
        "arousal",
        "language",
        "perspective",
    ]
    features = [c for c in base_features if c in df.columns]
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_funcs = {feat: ["mean", safe_std0, calculate_slope] for feat in features}
    df_agg = df.groupby("name").agg(agg_funcs)
    df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
    df_agg.reset_index(inplace=True)
    df_agg.rename(columns={"name": "Name"}, inplace=True)

    feature_cols = [c for c in df_agg.columns if c != "Name"]
    df_agg[feature_cols] = df_agg[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df_with_name = df.rename(columns={"name": "Name"})

    # Advanced mining features.
    df_advanced = mine_advanced_features(df_with_name)
    if not df_advanced.empty:
        df_agg = pd.merge(df_agg, df_advanced, on="Name", how="left")

    # SUDS time-series dynamics features.
    df_suds_ts = extract_suds_dynamics_features(df_with_name)
    if not df_suds_ts.empty:
        df_agg = pd.merge(df_agg, df_suds_ts, on="Name", how="left")

    feature_cols = [c for c in df_agg.columns if c != "Name"]
    df_agg[feature_cols] = df_agg[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df_agg


def _load_subject_level_df(
    *,
    task1_path: Path,
    task2_path: Path,
    followup_col: str | None,
    threshold: float,
) -> pd.DataFrame:
    from src.modeling.core_utils import collapse_to_subject_level, drop_task1_columns_from_task2

    df_task1 = pd.read_excel(task1_path)
    df_task2 = pd.read_excel(task2_path)

    if "name" not in df_task1.columns and "Name" in df_task1.columns:
        df_task1.rename(columns={"Name": "name"}, inplace=True)
    if "Name" not in df_task2.columns and "name" in df_task2.columns:
        df_task2.rename(columns={"name": "Name"}, inplace=True)

    df_task1["name"] = df_task1["name"].astype(str)
    df_task2["Name"] = df_task2["Name"].astype(str)

    df_task1_agg = _aggregate_task1_features(df_task1)

    df_task1_for_drop = df_task1.rename(columns={"name": "Name"})
    df_task2_clean = drop_task1_columns_from_task2(df_task2, df_task1_for_drop)
    df_task2_subj = collapse_to_subject_level(df_task2_clean)

    df_merged = pd.merge(df_task2_subj, df_task1_agg, on="Name", how="inner")
    follow = followup_col or _select_followup_pcl_column(df_merged)

    if "PCL_T1" not in df_merged.columns:
        raise KeyError("Missing baseline PCL column: expected `PCL_T1`.")

    df_merged["PCL_delta"] = df_merged[follow] - df_merged["PCL_T1"]
    df_merged["Is_Responder"] = (df_merged["PCL_delta"] <= threshold).astype(int)
    df_merged["Is_NonResponder"] = (1 - df_merged["Is_Responder"]).astype(int)
    df_final = df_merged.dropna(subset=["PCL_delta", "Is_NonResponder"]).copy()
    return df_final


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _plot_logreg_diagnostics(results_dir: Path, out_dir: Path) -> None:
    diag_path = results_dir / "binary_logreg_diagnostics.json"
    if not diag_path.exists():
        return

    diag = _read_json(diag_path)
    cm = np.array(diag["confusion_matrix"], dtype=int)
    fpr = np.array(diag["roc"]["fpr"], dtype=float)
    tpr = np.array(diag["roc"]["tpr"], dtype=float)
    auc = float(diag["roc"]["auc"])
    prec = np.array(diag["pr"]["precision"], dtype=float)
    rec = np.array(diag["pr"]["recall"], dtype=float)
    ap = float(diag["pr"]["ap"])
    frac_pos = np.array(diag["calibration"]["frac_pos"], dtype=float)
    mean_pred = np.array(diag["calibration"]["mean_pred"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix (OOF)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(["Responder", "Non-responder"])
    ax.set_yticklabels(["Responder", "Non-responder"], rotation=0)

    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.plot(fpr, tpr, color="#4C78A8", linewidth=2)
    ax.set_title(f"ROC (AUC={auc:.3f})")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 0]
    baseline = cm[1].sum() / max(cm.sum(), 1)
    ax.hlines(baseline, 0, 1, linestyles="--", color="gray", linewidth=1, label="Baseline")
    ax.plot(rec, prec, color="#F58518", linewidth=2, label="Model")
    ax.set_title(f"PR (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1, 1]
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.plot(mean_pred, frac_pos, marker="o", color="#54A24B", linewidth=2)
    ax.set_title("Calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("Binary Model Diagnostics (LogReg)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "model_logreg_diagnostics_panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_group_importance(results_dir: Path, out_dir: Path) -> None:
    imp_path = results_dir / "feature_importance_importances.json"
    if not imp_path.exists():
        return

    payload = _read_json(imp_path)
    names = payload["feature_names"]
    imps = np.asarray(payload["importances"], dtype=float)

    def group(name: str) -> str:
        if name.startswith("SUDS_"):
            return "SUDS dynamics"
        if name.startswith("Advanced_"):
            return "Advanced mining"
        if name.endswith("_T1") and name != "PCL_T1":
            return "Baseline scales"
        if name.startswith("PCL_") or name in {"PCL_T1"}:
            return "Baseline scales"
        if name.startswith(("VAM_index_", "SAM_index_")):
            return "VAM/SAM composite"
        if name.startswith(("temporal_", "coherence_", "reflection_", "sensory_", "arousal_", "language_", "perspective_")):
            return "Raw dimensions"
        return "Other"

    df = pd.DataFrame({"feature": names, "importance": np.abs(imps)})
    df["group"] = df["feature"].map(group)
    grouped = df.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=grouped, x="importance", y="group", ax=ax, palette="viridis")
    ax.set_title("Feature Importance by Group (abs importance sum)")
    ax.set_xlabel("Sum of |importance|")
    ax.set_ylabel("")
    for i, row in grouped.reset_index(drop=True).iterrows():
        ax.text(row["importance"], i, f"  {row['importance']:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance_by_group.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_vs_response(df: pd.DataFrame, out_dir: Path) -> None:
    if "Cluster_Label" not in df.columns:
        return

    tab = pd.crosstab(df["Cluster_Label"], df["Is_NonResponder"])
    tab = tab.rename(columns={0: "Responder", 1: "Non-responder"})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    sns.heatmap(tab, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Cluster × Outcome (counts)")
    ax.set_xlabel("")
    ax.set_ylabel("Cluster")

    ax = axes[1]
    tab_pct = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    sns.heatmap(tab_pct, annot=True, fmt=".1f", cmap="Oranges", cbar=False, ax=ax)
    ax.set_title("Cluster × Outcome (row %)")
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()
    fig.savefig(out_dir / "spectral_cluster_vs_outcome.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_feature_matrix(df: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> None:
    cols = [c for c in feature_cols if c in df.columns]
    if not cols or "Cluster_Label" not in df.columns:
        return

    tmp = df[["Name", "Cluster_Label", "Is_NonResponder", *cols]].copy()
    tmp = tmp.sort_values(["Cluster_Label", "Is_NonResponder", "Name"])
    X = tmp[cols].to_numpy(dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    Z = (X - mu) / sd

    fig, ax = plt.subplots(figsize=(10, max(4, 0.18 * len(tmp))))
    sns.heatmap(
        Z,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        cbar_kws={"label": "Z-score"},
        yticklabels=False,
        xticklabels=cols,
    )
    ax.set_title("Subjects × Features (sorted by cluster & outcome)")
    ax.set_xlabel("")
    ax.set_ylabel("Subjects")

    # Draw cluster separators
    boundaries = tmp["Cluster_Label"].ne(tmp["Cluster_Label"].shift()).to_numpy()
    idxs = np.where(boundaries)[0]
    for i in idxs[1:]:
        ax.hlines(i, *ax.get_xlim(), colors="black", linewidth=1)

    fig.tight_layout()
    fig.savefig(out_dir / "spectral_cluster_feature_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_radar(df: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> None:
    cols = [c for c in feature_cols if c in df.columns]
    if not cols or "Cluster_Label" not in df.columns:
        return

    tmp = df[["Cluster_Label", *cols]].copy()
    X = tmp[cols].to_numpy(dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    tmp[cols] = (tmp[cols] - mu) / sd

    means = tmp.groupby("Cluster_Label")[cols].mean()
    labels = list(means.index)

    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, polar=True)

    for k in labels:
        vals = means.loc[k].to_numpy(dtype=float)
        vals = np.concatenate([vals, vals[:1]])
        ax.plot(angles, vals, linewidth=2, label=f"Cluster {k}")
        ax.fill(angles, vals, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_title("Cluster Profile (Z-scored means)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "spectral_cluster_profile_radar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_suds_dynamics(df: pd.DataFrame, out_dir: Path) -> None:
    need = ["SUDS_ARIMA_ar1", "SUDS_GARCH_sigma2_mean", "Is_NonResponder"]
    if any(c not in df.columns for c in need):
        return

    tmp = df[["Is_NonResponder", "SUDS_ARIMA_ar1", "SUDS_GARCH_sigma2_mean"]].copy()
    tmp["Outcome"] = tmp["Is_NonResponder"].map({0: "Responder", 1: "Non-responder"})
    tmp["Volatility"] = np.log1p(pd.to_numeric(tmp["SUDS_GARCH_sigma2_mean"], errors="coerce"))
    tmp["Inertia"] = pd.to_numeric(tmp["SUDS_ARIMA_ar1"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    sns.scatterplot(data=tmp, x="Inertia", y="Volatility", hue="Outcome", ax=ax, s=60)
    ax.set_title("SUDS Inertia vs Volatility")
    ax.set_xlabel("ARIMA AR(1) (inertia)")
    ax.set_ylabel("log(1 + GARCH sigma2_mean)")

    ax = axes[1]
    sns.violinplot(data=tmp, x="Outcome", y="Volatility", ax=ax, inner="box", cut=0)
    ax.set_title("Volatility by Outcome")
    ax.set_xlabel("")
    ax.set_ylabel("log(1 + GARCH sigma2_mean)")

    fig.tight_layout()
    fig.savefig(out_dir / "suds_inertia_volatility.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_example_suds_trajectories(
    *,
    task1_path: Path,
    df_subject: pd.DataFrame,
    out_dir: Path,
    top_k: int = 6,
) -> None:
    if "SUDS_GARCH_sigma2_mean" not in df_subject.columns:
        return

    df_task1 = pd.read_excel(task1_path)
    if "Name" in df_task1.columns and "name" not in df_task1.columns:
        df_task1.rename(columns={"Name": "name"}, inplace=True)
    if "name" not in df_task1.columns:
        return

    df_task1["name"] = df_task1["name"].astype(str)
    df_task1["session"] = pd.to_numeric(df_task1.get("session"), errors="coerce")
    df_task1["suds_after"] = pd.to_numeric(df_task1.get("suds_after"), errors="coerce")

    top = (
        df_subject[["Name", "SUDS_GARCH_sigma2_mean"]]
        .assign(SUDS_GARCH_sigma2_mean=pd.to_numeric(df_subject["SUDS_GARCH_sigma2_mean"], errors="coerce"))
        .dropna()
        .sort_values("SUDS_GARCH_sigma2_mean", ascending=False)
        .head(top_k)
    )
    if top.empty:
        return

    ncols = 2
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.2 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, v) in zip(axes, top[["Name", "SUDS_GARCH_sigma2_mean"]].itertuples(index=False, name=None)):
        g = df_task1[df_task1["name"] == name][["session", "suds_after"]].dropna()
        if g.empty:
            ax.set_axis_off()
            continue
        g = g.groupby("session", as_index=False)["suds_after"].mean().sort_values("session")
        ax.plot(g["session"], g["suds_after"], marker="o", linewidth=2, color="#4C78A8")
        ax.set_title(f"{name} (vol={v:.1f})", fontsize=10)
        ax.set_xlabel("Session")
        ax.set_ylabel("SUDS_after")
        ax.grid(True, alpha=0.25)

    for ax in axes[len(top) :]:
        ax.set_axis_off()

    fig.suptitle("Example SUDS Trajectories (Top volatility subjects)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "suds_example_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_index_html(out_dir: Path, images: list[str]) -> None:
    items = "\n".join(
        f'<div class=\"card\"><div class=\"name\">{name}</div><a href=\"{name}\"><img src=\"{name}\" /></a></div>'
        for name in images
    )
    html = f"""<!doctype html>
<html lang=\"zh\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Report Images</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial; margin: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; }}
    .name {{ font-size: 13px; color: #374151; margin-bottom: 8px; word-break: break-all; }}
    img {{ width: 100%; height: auto; border-radius: 6px; }}
  </style>
</head>
<body>
  <h2>Report Images</h2>
  <div class=\"grid\">{items}</div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    warnings.filterwarnings("ignore")
    _try_set_chinese_font()

    p = argparse.ArgumentParser(description="Generate more intuitive plots into report_image/")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="report_image")
    p.add_argument("--task1", default="data/task1_scored.xlsx")
    p.add_argument("--task2", default="data/task2_processing_data.xlsx")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    task1_path = Path(args.task1)
    task2_path = Path(args.task2)

    _ensure_dir(out_dir)

    metrics_path = results_dir / "metrics.json"
    metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    followup_col = metrics.get("followup_col")
    threshold = float(metrics.get("threshold_delta", -10.0))

    df_final = _load_subject_level_df(
        task1_path=task1_path,
        task2_path=task2_path,
        followup_col=followup_col,
        threshold=threshold,
    )

    cluster_path = results_dir / "spectral_clusters.csv"
    if cluster_path.exists():
        clusters = pd.read_csv(cluster_path)
        df_final = pd.merge(df_final, clusters, on="Name", how="left")

    # Copy a few existing key images for convenience.
    for fname in [
        "spectral_clustering_pca.png",
        "spectral_clustering_heatmap.png",
        "feature_importance.png",
        "feature_importance_regression.png",
    ]:
        _copy_if_exists(results_dir / fname, out_dir / fname)

    _plot_logreg_diagnostics(results_dir, out_dir)
    _plot_feature_group_importance(results_dir, out_dir)
    _plot_cluster_vs_response(df_final, out_dir)

    cluster_feats = [
        "Advanced_dissociation_index_mean",
        "Advanced_dissociation_index_max",
        "Advanced_reflection_max",
        "VAM_index_calculate_slope",
        "reflection_calculate_slope",
        "coherence_mean",
        "SUDS_ARIMA_ar1",
        "SUDS_GARCH_sigma2_mean",
        "SUDS_GARCH_sigma2_max",
    ]
    _plot_cluster_feature_matrix(df_final, cluster_feats, out_dir)
    _plot_cluster_radar(df_final, cluster_feats, out_dir)
    _plot_suds_dynamics(df_final, out_dir)
    _plot_example_suds_trajectories(task1_path=task1_path, df_subject=df_final, out_dir=out_dir)

    images = sorted([p.name for p in out_dir.glob("*.png")])
    _write_index_html(out_dir, images)
    print(f"Wrote {len(images)} images to {out_dir}/ (see index.html)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
