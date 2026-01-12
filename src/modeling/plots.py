# src/modeling/plots.py
"""
Plotting utilities for model diagnostics and feature importance visualization.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
import json


def plot_binary_diagnostics(
    results_dir: Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    title_suffix: str,
    out_prefix: str,
) -> None:
    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix ({title_suffix})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(results_dir / f"{out_prefix}_confusion_matrix.png")
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = float(roc_auc_score(y_true, y_prob))
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.legend()
    plt.title(f"ROC ({title_suffix})")
    plt.savefig(results_dir / f"{out_prefix}_roc_curve.png")
    plt.close()

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.legend()
    plt.title(f"Precision-Recall ({title_suffix})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(results_dir / f"{out_prefix}_pr_curve.png")
    plt.close()

    # Calibration
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="uniform")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(f"Calibration ({title_suffix})")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.savefig(results_dir / f"{out_prefix}_calibration.png")
    plt.close()

    # prepare numeric diagnostics to save as JSON for downstream reporting/LLM parsing
    diagnostics = {
        "confusion_matrix": cm.tolist(),
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc)},
        "pr": {"precision": prec.tolist(), "recall": rec.tolist(), "ap": float(ap)},
        "calibration": {"frac_pos": frac_pos.tolist(), "mean_pred": mean_pred.tolist()},
    }

    # write diagnostics json

    with open(results_dir / f"{out_prefix}_diagnostics.json", "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, ensure_ascii=False, indent=2)


    return diagnostics


def plot_triclass_confusion(
    results_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    class_names: list[str],
    out_name: str,
) -> None:
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix (3-class)")
    plt.savefig(results_dir / out_name)
    plt.close()
    # save numeric confusion matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with open(results_dir / (out_name.replace('.png', '') + "_matrix.json"), "w", encoding="utf-8") as fh:
        json.dump({"labels": labels, "class_names": class_names, "confusion_matrix": cm.tolist()}, fh, ensure_ascii=False, indent=2)



def plot_top_feature_importance(
    results_dir: Path,
    importances: np.ndarray,
    feature_names: list[str],
    title: str,
    out_name: str,
    top_k: int = 20,
) -> None:
    top_idx = np.argsort(importances)[-top_k:]
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_idx)), importances[top_idx], color="#4c72b0", align="center")
    plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
    plt.xlabel("Relative Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(results_dir / out_name)
    plt.close()
    # save numeric importances

    importance_payload = {
        "feature_names": list(feature_names),
        "importances": [float(x) for x in importances.tolist()],
    }
    with open(results_dir / (out_name.replace('.png', '') + "_importances.json"), "w", encoding="utf-8") as fh:
        json.dump(importance_payload, fh, ensure_ascii=False, indent=2)


    return importance_payload

