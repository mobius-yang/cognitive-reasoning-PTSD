from __future__ import annotations

import json
from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
RANDOM_STATE = 42


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def pick_stratified_cv(y: pd.Series, max_splits: int = 5):
    counts = y.value_counts(dropna=False)
    if counts.empty:
        return None
    min_class = int(counts.min())
    n_splits = min(max_splits, min_class)
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def pick_kfold(n_samples: int, max_splits: int = 5):
    n_splits = min(max_splits, n_samples)
    if n_splits < 2:
        return None
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def drop_task1_columns_from_task2(df_task2: pd.DataFrame, df_task1: pd.DataFrame) -> pd.DataFrame:
    """
    `data_processing.py` 会把 `task1_scored.xlsx`（按 session 的文本数据）merge 进 task2，
    导致每个被试在 task2 中重复多行。这里把 task1 列从 task2 中剔除，再按 Name 聚合到被试级。
    """
    task1_cols = set(df_task1.columns)
    cols_to_drop = [c for c in df_task2.columns if (c in task1_cols) and (c != "Name")]
    return df_task2.drop(columns=cols_to_drop, errors="ignore")


def collapse_to_subject_level(df: pd.DataFrame) -> pd.DataFrame:
    if "Name" not in df.columns:
        raise KeyError("缺少 Name 列，无法聚合到被试级")
    return df.groupby("Name", as_index=False).first()


def pick_baseline_scale_columns(df: pd.DataFrame) -> list[str]:
    """
    仅保留基线量表特征（T1），避免把 T2/T3 或其派生统计量当作特征造成标签泄露。
    """
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


def choose_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5
    f1s = 2 * precision[:-1] * recall[:-1] / np.clip((precision[:-1] + recall[:-1]), 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1s))])


def plot_binary_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    title_suffix: str,
    out_prefix: str,
) -> None:
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"混淆矩阵 ({title_suffix})")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.savefig(RESULTS_DIR / f"{out_prefix}_confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = float(roc_auc_score(y_true, y_prob))
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.legend()
    plt.title(f"ROC ({title_suffix})")
    plt.savefig(RESULTS_DIR / f"{out_prefix}_roc_curve.png")
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.legend()
    plt.title(f"PR Curve ({title_suffix})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(RESULTS_DIR / f"{out_prefix}_pr_curve.png")
    plt.close()

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="uniform")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(f"Calibration ({title_suffix})")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.savefig(RESULTS_DIR / f"{out_prefix}_calibration.png")
    plt.close()


def main() -> None:
    ensure_results_dir()

    print("正在加载数据...")
    task1_path = "task1_scored.xlsx"
    task2_path = "task2_processing_data.xlsx"
    df_task1 = pd.read_excel(task1_path)
    df_task2 = pd.read_excel(task2_path)

    if "name" not in df_task1.columns and "Name" in df_task1.columns:
        df_task1.rename(columns={"Name": "name"}, inplace=True)
    if "Name" not in df_task2.columns and "name" in df_task2.columns:
        df_task2.rename(columns={"name": "Name"}, inplace=True)

    df_task1["name"] = df_task1["name"].astype(str)
    df_task2["Name"] = df_task2["Name"].astype(str)

    print("正在聚合文本特征（被试级）...")
    nlp_features = ["VAM_index", "SAM_index", "temporal", "coherence", "reflection", "sensory", "arousal"]
    agg_funcs = {feat: ["mean", safe_std0, calculate_slope] for feat in nlp_features}
    df_task1_agg = df_task1.groupby("name").agg(agg_funcs)
    df_task1_agg.columns = ["_".join(col).strip() for col in df_task1_agg.columns.values]
    df_task1_agg.reset_index(inplace=True)
    df_task1_agg.rename(columns={"name": "Name"}, inplace=True)

    text_feature_cols = [c for c in df_task1_agg.columns if c != "Name"]
    df_task1_agg[text_feature_cols] = (
        df_task1_agg[text_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    print("正在将 task2 聚合到被试级（避免 CV 泄露）...")
    df_task1_for_drop = df_task1.rename(columns={"name": "Name"})
    df_task2_clean = drop_task1_columns_from_task2(df_task2, df_task1_for_drop)
    df_task2_subj = collapse_to_subject_level(df_task2_clean)

    print("正在合并数据集（被试级）...")
    df_merged = pd.merge(df_task2_subj, df_task1_agg, on="Name", how="inner")

    followup_col = None
    for candidate in ["PCL_T3", "PCL_T2"]:
        if candidate in df_merged.columns:
            followup_col = candidate
            break
    if followup_col is None:
        raise KeyError("缺少随访 PCL 列（PCL_T2/PCL_T3）")
    if "PCL_T1" not in df_merged.columns:
        raise KeyError("缺少基线 PCL_T1 列")

    threshold = -10
    df_merged["PCL_delta"] = df_merged[followup_col] - df_merged["PCL_T1"]
    df_merged["Is_Responder"] = (df_merged["PCL_delta"] <= threshold).astype(int)
    df_merged["Is_NonResponder"] = (1 - df_merged["Is_Responder"]).astype(int)

    tri_available = ("PCL_T2" in df_merged.columns) and ("PCL_T3" in df_merged.columns)
    if tri_available:
        df_merged["PCL_delta_T2"] = df_merged["PCL_T2"] - df_merged["PCL_T1"]
        df_merged["PCL_delta_T3"] = df_merged["PCL_T3"] - df_merged["PCL_T1"]
        df_merged["Response_Class"] = np.select(
            condlist=[
                df_merged["PCL_delta_T2"] <= threshold,
                (df_merged["PCL_delta_T2"] > threshold) & (df_merged["PCL_delta_T3"] <= threshold),
            ],
            choicelist=[2, 1],  # 2=快, 1=慢
            default=0,  # 0=无
        ).astype(int)

    df_final = df_merged.dropna(subset=["PCL_delta", "Is_NonResponder"]).copy()
    print(f"最终样本量(被试级): {len(df_final)}  |  唯一 Name: {df_final['Name'].nunique()}")

    baseline_scale_cols = pick_baseline_scale_columns(df_final)
    feature_cols = list(dict.fromkeys(text_feature_cols + baseline_scale_cols))
    X_bin, feature_names = build_X(df_final, feature_cols)
    y_bin = df_final["Is_NonResponder"].astype(int)
    y_reg = df_final["PCL_delta"].astype(float)

    metrics_payload: dict = {
        "n_samples_subject_level": int(len(df_final)),
        "followup_col": followup_col,
        "threshold_delta": float(threshold),
        "class_distribution(Is_NonResponder)": y_bin.value_counts().to_dict(),
        "n_text_features": int(len(text_feature_cols)),
        "n_baseline_scale_features": int(len(baseline_scale_cols)),
        "n_model_features_after_dummies": int(X_bin.shape[1]),
    }

    print("\n--- [A] 二分类：预测无响应 (Is_NonResponder=1) ---")
    cv_bin = pick_stratified_cv(y_bin)
    pos = int(y_bin.sum())
    neg = int(len(y_bin) - pos)
    scale_pos_weight = float(neg / max(pos, 1))

    binary_models: dict[str, tuple[object, dict]] = {
        "logreg": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "xgb": (
            XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
            ),
            {
                "n_estimators": [100, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
    }

    binary_results: dict[str, dict] = {}
    best_model_name = None
    best_auprc = -1.0

    for name, (estimator, param_grid) in binary_models.items():
        print(f"\n[A/{name}] 正在训练与交叉验证...")
        grid = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring="average_precision",
            cv=cv_bin if cv_bin is not None else 3,
            n_jobs=-1,
            refit=True,
        )

        if cv_bin is not None:
            y_prob_oof = cross_val_predict(grid, X_bin, y_bin, cv=cv_bin, method="predict_proba")[:, 1]
        else:
            grid.fit(X_bin, y_bin)
            y_prob_oof = grid.predict_proba(X_bin)[:, 1]

        thr = choose_threshold_by_f1(y_bin.to_numpy(), y_prob_oof)
        y_pred_oof = (y_prob_oof >= thr).astype(int)

        roc_auc = float(roc_auc_score(y_bin, y_prob_oof))
        auprc = float(average_precision_score(y_bin, y_prob_oof))
        brier = float(brier_score_loss(y_bin, y_prob_oof))
        bal_acc = float(balanced_accuracy_score(y_bin, y_pred_oof))
        f1 = float(f1_score(y_bin, y_pred_oof))
        cm = confusion_matrix(y_bin, y_pred_oof)
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]
        sensitivity = float(tp / max(tp + fn, 1))
        specificity = float(tn / max(tn + fp, 1))

        print(f"[A/{name}] OOF: AUC={roc_auc:.3f}  AUPRC={auprc:.3f}  Brier={brier:.3f}  thr={thr:.3f}")
        print(classification_report(y_bin, y_pred_oof, digits=3))

        plot_binary_diagnostics(
            y_true=y_bin.to_numpy(),
            y_prob=y_prob_oof,
            y_pred=y_pred_oof,
            title_suffix=f"{name} (Is_NonResponder)",
            out_prefix=f"binary_{name}",
        )

        binary_results[name] = {
            "oof": {
                "roc_auc": roc_auc,
                "auprc": auprc,
                "brier": brier,
                "balanced_accuracy": bal_acc,
                "f1": f1,
                "threshold": thr,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            }
        }

        if auprc > best_auprc:
            best_auprc = auprc
            best_model_name = name

    metrics_payload["binary_models"] = binary_results
    metrics_payload["binary_best_by_auprc"] = best_model_name

    if best_model_name is not None:
        best_estimator, best_param_grid = binary_models[best_model_name]
        best_search = GridSearchCV(
            best_estimator,
            param_grid=best_param_grid,
            scoring="average_precision",
            cv=cv_bin if cv_bin is not None else 3,
            n_jobs=-1,
            refit=True,
        )
        best_search.fit(X_bin, y_bin)

        if best_model_name == "logreg":
            clf = best_search.best_estimator_.named_steps["clf"]
            coefs = np.abs(clf.coef_).ravel()
            imp = pd.Series(coefs, index=feature_names).sort_values(ascending=False).head(20)
            title = "Top 20 Feature Importance (LogReg | abs(coef))"
        else:
            clf = best_search.best_estimator_
            importances = clf.feature_importances_
            imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
            title = "Top 20 Feature Importance (XGBoost)"

        plt.figure(figsize=(10, 8))
        imp.sort_values().plot(kind="barh", color="#4c72b0")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "feature_importance.png")
        plt.close()

    if tri_available:
        print("\n--- [C] 三分类：快/慢/无响应 ---")
        df_tri = df_final.dropna(subset=["Response_Class"]).copy()
        X_tri, _ = build_X(df_tri, feature_cols)
        y_tri = df_tri["Response_Class"].astype(int)

        cv_tri = pick_stratified_cv(y_tri)
        tri_pipe = Pipeline(
            steps=[
                ("var", VarianceThreshold(threshold=0.0)),
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )
        tri_grid = GridSearchCV(
            tri_pipe,
            param_grid={"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.01, 0.1]},
            scoring="f1_macro",
            cv=cv_tri if cv_tri is not None else 3,
            n_jobs=-1,
            refit=True,
        )
        if cv_tri is not None:
            y_tri_pred = cross_val_predict(tri_grid, X_tri, y_tri, cv=cv_tri, method="predict")
        else:
            tri_grid.fit(X_tri, y_tri)
            y_tri_pred = tri_grid.predict(X_tri)

        print(classification_report(y_tri, y_tri_pred, digits=3))
        plt.figure(figsize=(6, 5))
        cm3 = confusion_matrix(y_tri, y_tri_pred, labels=[0, 1, 2])
        sns.heatmap(
            cm3,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["无响应", "慢响应", "快响应"],
            yticklabels=["无响应", "慢响应", "快响应"],
        )
        plt.title("混淆矩阵 (三分类)")
        plt.savefig(RESULTS_DIR / "confusion_matrix_3class.png")
        plt.close()

    print("\n--- [B] 回归：预测 PCL_delta ---")
    cv_reg = pick_kfold(len(df_final))
    rf_model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE)
    if cv_reg is not None:
        y_reg_pred = cross_val_predict(rf_model, X_bin, y_reg, cv=cv_reg, method="predict")
        metrics_payload["regression_oof_r2"] = float(r2_score(y_reg, y_reg_pred))
        print(f"[B] OOF R2={metrics_payload['regression_oof_r2']:.3f}")

    rf_model.fit(X_bin, y_reg)
    importances = rf_model.feature_importances_
    top_idx = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_idx)), importances[top_idx], color="#4c72b0", align="center")
    plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
    plt.xlabel("Relative Importance")
    plt.title("Top 20 特征重要性 (Random Forest Regressor)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance_regression.png")
    plt.close()

    save_json(RESULTS_DIR / "metrics.json", metrics_payload)
    print("\n分析全部完成！结果已保存在 results 文件夹（含 metrics.json）。")


if __name__ == "__main__":
    main()

