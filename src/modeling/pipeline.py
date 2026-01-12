# src/modeling/pipeline.py
"""
Model training and evaluation pipeline.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


from .core_utils import (
    build_X,
    calculate_slope,
    choose_threshold_by_f1,
    collapse_to_subject_level,
    drop_task1_columns_from_task2,
    ensure_dir,
    pick_baseline_scale_columns,
    pick_kfold,
    pick_stratified_cv,
    safe_std0,
    save_json,
)
from .plots import plot_binary_diagnostics, plot_top_feature_importance, plot_triclass_confusion
from .feature_mining import mine_advanced_features
from .time_series import extract_suds_dynamics_features
from .unsupervised import run_spectral_clustering

_N_JOBS = 1

def _select_followup_pcl_column(df: pd.DataFrame) -> str:
    for candidate in ["PCL_T3", "PCL_T2"]:
        if candidate in df.columns:
            return candidate
    raise KeyError("Missing follow-up PCL column: expected `PCL_T2` or `PCL_T3`.")


def _aggregate_text_features(df_task1: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Aggregate text features to subject level.
    Standard aggregation (Mean, Std, Slope) + Advanced Mining (Dissociation, Breakthrough).
    """
    nlp_features = ["VAM_index", "SAM_index", "temporal", "coherence", "reflection", "sensory", "arousal"]
    
    # Ensure numeric types
    for col in nlp_features:
        if col in df_task1.columns:
            df_task1[col] = pd.to_numeric(df_task1[col], errors="coerce")
    
    # 1. Basic aggregation
    agg_funcs = {feat: ["mean", safe_std0, calculate_slope] for feat in nlp_features}
    df_agg = df_task1.groupby("name").agg(agg_funcs)
    df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
    df_agg.reset_index(inplace=True)
    df_agg.rename(columns={"name": "Name"}, inplace=True)

    text_feature_cols = [c for c in df_agg.columns if c != "Name"]
    df_agg[text_feature_cols] = df_agg[text_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # 2. Advanced feature mining (Dissociation & Breakthrough)
    print("  Running advanced feature mining (Dissociation & Breakthrough)...")
    df_task1_with_name = df_task1.copy()
    if "name" in df_task1_with_name.columns:
        df_task1_with_name.rename(columns={"name": "Name"}, inplace=True)
    
    df_advanced = mine_advanced_features(df_task1_with_name)
    
    # Merge advanced features if available
    if not df_advanced.empty:
        df_agg = pd.merge(df_agg, df_advanced, on="Name", how="left")
        # Update feature list to include advanced features
        text_feature_cols = [c for c in df_agg.columns if c != "Name"]
        # Fill NA with 0
        df_agg[text_feature_cols] = df_agg[text_feature_cols].fillna(0.0)

    # 3. SUDS time-series analysis features (ARIMA dynamics + GARCH volatility)
    print("  Extracting SUDS time-series features (ARIMA + GARCH)...")
    df_suds_ts = extract_suds_dynamics_features(df_task1_with_name)
    if not df_suds_ts.empty:
        df_agg = pd.merge(df_agg, df_suds_ts, on="Name", how="left")
        text_feature_cols = [c for c in df_agg.columns if c != "Name"]
        df_agg[text_feature_cols] = df_agg[text_feature_cols].fillna(0.0)
    
    return df_agg, text_feature_cols


def main(
    *,
    task1_path: str = "data/task1_scored.xlsx",
    task2_path: str = "data/task2_processing_data.xlsx",
    results_dir: Path = Path("results"),
    random_state: int = 42,
) -> None:
    ensure_dir(results_dir)

    print("Loading input data...")
    df_task1 = pd.read_excel(task1_path)
    df_task2 = pd.read_excel(task2_path)

    if "name" not in df_task1.columns and "Name" in df_task1.columns:
        df_task1.rename(columns={"Name": "name"}, inplace=True)
    if "Name" not in df_task2.columns and "name" in df_task2.columns:
        df_task2.rename(columns={"name": "Name"}, inplace=True)

    df_task1["name"] = df_task1["name"].astype(str)
    df_task2["Name"] = df_task2["Name"].astype(str)

    print("Aggregating text features (subject-level)...")
    df_task1_agg, text_feature_cols = _aggregate_text_features(df_task1)

    print("Aggregating task2 to subject-level (avoid CV leakage)...")
    df_task1_for_drop = df_task1.rename(columns={"name": "Name"})
    df_task2_clean = drop_task1_columns_from_task2(df_task2, df_task1_for_drop)
    df_task2_subj = collapse_to_subject_level(df_task2_clean)

    print("Merging datasets (subject-level)...")
    df_merged = pd.merge(df_task2_subj, df_task1_agg, on="Name", how="inner")

    followup_col = _select_followup_pcl_column(df_merged)
    if "PCL_T1" not in df_merged.columns:
        raise KeyError("Missing baseline PCL column: expected `PCL_T1`.")

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
            choicelist=[2, 1],  # 2=fast responder, 1=slow responder
            default=0,  # 0=non-responder
        ).astype(int)

    df_final = df_merged.dropna(subset=["PCL_delta", "Is_NonResponder"]).copy()
    print(f"Final sample size (subject-level): {len(df_final)} | Unique Name: {df_final['Name'].nunique()}")

    # --- Unsupervised Analysis: Spectral Clustering ---
    # Use advanced features + key dynamic features for clustering
    print("\n[Unsupervised] Spectral Clustering Analysis...")
    clustering_feats = []
    
    # Check for advanced features
    potential_feats = [
        'Advanced_dissociation_index_mean',
        'Advanced_dissociation_index_max', 
        'Advanced_reflection_max',
        'VAM_index_calculate_slope',
        'reflection_calculate_slope',
        'coherence_mean',
        'SUDS_ARIMA_ar1',
        'SUDS_GARCH_sigma2_mean',
        'SUDS_GARCH_sigma2_max',
    ]
    
    # Only use features that exist in the dataframe
    clustering_feats = [c for c in potential_feats if c in df_final.columns]
    
    if len(clustering_feats) >= 2:
        df_final = run_spectral_clustering(
            df_final, 
            clustering_feats, 
            results_dir,
            n_clusters=3
        )
    else:
        print(f"  [Warning] Not enough features for clustering (found {len(clustering_feats)}), skipping unsupervised analysis.")

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

    print("[A] Binary classification: predict non-response (Is_NonResponder=1)")
    cv_bin = pick_stratified_cv(y_bin, max_splits=5, random_state=random_state)
    pos = int(y_bin.sum())
    neg = int(len(y_bin) - pos)
    scale_pos_weight = float(neg / max(pos, 1))

    binary_models: dict[str, tuple[object, dict]] = {
        "logreg": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
                ]
            ),
            {"clf__C": [0.1, 1.0, 10.0]},
        )
    }

    if XGBClassifier is not None:
        binary_models["xgb"] = (
            XGBClassifier(
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=_N_JOBS,
                scale_pos_weight=scale_pos_weight,
            ),
            {
                "n_estimators": [100, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        )
    else:
        print("  [Warning] `xgboost` not installed; skipping XGBClassifier baseline.")

    binary_results: dict[str, dict] = {}
    best_model_name = None
    best_auprc = -1.0

    for name, (estimator, param_grid) in binary_models.items():
        print(f"\n[A/{name}] Training + cross-validation...")
        grid = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring="average_precision",
            cv=cv_bin if cv_bin is not None else 3,
            n_jobs=_N_JOBS,
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

        print(f"[A/{name}] OOF: AUC={roc_auc:.3f} AUPRC={auprc:.3f} Brier={brier:.3f} thr={thr:.3f}")
        print(classification_report(y_bin, y_pred_oof, digits=3))

        plot_binary_diagnostics(
            results_dir=results_dir,
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
            n_jobs=_N_JOBS,
            refit=True,
        )
        best_search.fit(X_bin, y_bin)

        if best_model_name == "logreg":
            clf = best_search.best_estimator_.named_steps["clf"]
            importances = np.abs(clf.coef_).ravel()
            title = "Top 20 Feature Importance (LogReg | abs(coef))"
        else:
            clf = best_search.best_estimator_
            importances = clf.feature_importances_
            title = "Top 20 Feature Importance (XGBoost)"

        plot_top_feature_importance(
            results_dir=results_dir,
            importances=importances,
            feature_names=feature_names,
            title=title,
            out_name="feature_importance.png",
            top_k=20,
        )

    if tri_available:
        print("[C] 3-class classification: fast/slow/non-responder")
        df_tri = df_final.dropna(subset=["Response_Class"]).copy()
        X_tri, _ = build_X(df_tri, feature_cols)
        y_tri = df_tri["Response_Class"].astype(int)

        cv_tri = pick_stratified_cv(y_tri, max_splits=5, random_state=random_state)
        tri_pipe = Pipeline(
            steps=[
                ("var", VarianceThreshold(threshold=0.0)),
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
            ]
        )
        tri_grid = GridSearchCV(
            tri_pipe,
            param_grid={"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.01, 0.1]},
            scoring="f1_macro",
            cv=cv_tri if cv_tri is not None else 3,
            n_jobs=_N_JOBS,
            refit=True,
        )
        if cv_tri is not None:
            y_tri_pred = cross_val_predict(tri_grid, X_tri, y_tri, cv=cv_tri, method="predict")
        else:
            tri_grid.fit(X_tri, y_tri)
            y_tri_pred = tri_grid.predict(X_tri)

        print(classification_report(y_tri, y_tri_pred, digits=3))
        plot_triclass_confusion(
            results_dir=results_dir,
            y_true=y_tri.to_numpy(),
            y_pred=y_tri_pred,
            labels=[0, 1, 2],
            class_names=["Non-responder", "Slow responder", "Fast responder"],
            out_name="confusion_matrix_3class.png",
        )

    print("[B] Regression: predict PCL_delta")
    cv_reg = pick_kfold(len(df_final), max_splits=5, random_state=random_state)
    rf_model = RandomForestRegressor(n_estimators=500, random_state=random_state)
    if cv_reg is not None:
        y_reg_pred = cross_val_predict(rf_model, X_bin, y_reg, cv=cv_reg, method="predict")
        metrics_payload["regression_oof_r2"] = float(r2_score(y_reg, y_reg_pred))
        print(f"[B] OOF R2={metrics_payload['regression_oof_r2']:.3f}")

    rf_model.fit(X_bin, y_reg)
    plot_top_feature_importance(
        results_dir=results_dir,
        importances=rf_model.feature_importances_,
        feature_names=feature_names,
        title="Top 20 Feature Importance (Random Forest Regressor)",
        out_name="feature_importance_regression.png",
        top_k=20,
    )

    save_json(results_dir / "metrics.json", metrics_payload)
    print("\nDone. Results saved to results/ (including metrics.json).")
