"""
Corrected BIS Hazard Classification Pipeline (Spatial LOGO Cross-Validation)

Hard Constraint: Evaluated strictly on the 755 real observed monthly rows
from `src/data/monthly_wqi_parameter_dataset.csv`. Never uses spline/interpolated labels.

Features: pH_mean, TDS_mean, Hardness_mean, Chloride_mean (4 cheap field-measurable parameters)
Target Schemas:
  - Binary Task: Good (<100) vs Not-Good (>=100) [Target Labels: 'Good', 'Not-Good']
  - 3-Class Task: Good (<100), Poor (100-200), Very Poor (>=200) [Target Labels: 'Good', 'Poor', 'Very Poor']

Validation: LeaveOneGroupOut (LOGO) by Location (98 locations).
Baseline: Majority-Class Baseline ("Always Predict Good") explicitly included in every table.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "src" / "data" / "monthly_wqi_parameter_dataset.csv"
OUT_DIR = ROOT / "reproducibility_results" / "table3_bis_classification_corrected"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ["pH_mean", "TDS_mean", "Hardness_mean", "Chloride_mean"]
TARGET_LABELS_BINARY = ["Good", "Not-Good"]
TARGET_LABELS_3CLASS = ["Good", "Poor", "Very Poor"]


def assign_bis_3class(wqi: float) -> str:
    if wqi < 100.0:
        return "Good"
    elif wqi < 200.0:
        return "Poor"
    else:
        return "Very Poor"


def assign_bis_binary(wqi: float) -> str:
    return "Good" if wqi < 100.0 else "Not-Good"


def load_and_preprocess_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Real observed dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_cols = ["Block", "Location", "Date", "WQI_mean"] + FEATURE_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df.dropna(subset=required_cols).copy()
    df["BIS_3Class"] = df["WQI_mean"].apply(assign_bis_3class)
    df["BIS_Binary"] = df["WQI_mean"].apply(assign_bis_binary)
    return df


def run_logo_evaluation(
    df: pd.DataFrame, target_col: str, target_labels: list[str], task_name: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    X = df[FEATURE_COLS].copy()
    y = df[target_col].copy()
    groups = df["Location"].copy()

    majority_class = "Good"
    class_counts = {c: int((y == c).sum()) for c in target_labels}
    total_samples = len(y)
    class_dist_str = ", ".join([f"{k}: {v} ({v/total_samples*100:.2f}%)" for k, v in class_counts.items()])

    print(f"\n========================================================")
    print(f"Task: {task_name}")
    print(f"Total Real Rows: {total_samples}")
    print(f"Target Labels Schema: {target_labels}")
    print(f"Class Distribution: {class_dist_str}")
    print(f"Majority Baseline Class: '{majority_class}' ({class_counts[majority_class]/total_samples*100:.2f}%)")
    print(f"========================================================")

    models: dict[str, Any] = {
        "Majority Baseline": "baseline",
        "Random Forest (Default)": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=50, class_weight="balanced", min_samples_leaf=5, random_state=42, n_jobs=1),
        "Gradient Boosting (Default)": HistGradientBoostingClassifier(max_iter=50, random_state=42),
        "Gradient Boosting (Balanced)": HistGradientBoostingClassifier(max_iter=50, class_weight="balanced", random_state=42),
        "KNN (k=5, Uniform)": KNeighborsClassifier(n_neighbors=5, weights="uniform"),
        "KNN (k=5, Distance)": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Logistic Regression (Balanced)": LogisticRegression(class_weight="balanced", max_iter=500, random_state=42),
    }

    logo = LeaveOneGroupOut()

    summary_rows = []
    fold_detail_rows = []
    ci_summary_rows = []
    rng = np.random.default_rng(42)

    for model_name, model in models.items():
        y_true_all = []
        y_pred_all = []
        fold_true = []
        fold_pred = []

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            heldout_loc = groups.iloc[test_idx[0]]

            if model == "baseline":
                preds = np.array([majority_class] * len(test_idx))
            else:
                model.fit(X_train, y_train)
                preds = np.array(model.predict(X_test))

            yt_arr = np.array(y_test)
            fold_true.append(yt_arr)
            fold_pred.append(preds)

            y_true_all.extend(yt_arr)
            y_pred_all.extend(preds)

            # Record per-fold metrics
            fold_acc = accuracy_score(y_test, preds)
            fold_f1 = f1_score(y_test, preds, labels=target_labels, average="macro", zero_division=0)
            fold_detail_rows.append({
                "Task": task_name,
                "Model": model_name,
                "Heldout_Location": heldout_loc,
                "N_Test": len(test_idx),
                "Accuracy": round(fold_acc, 4),
                "Macro_F1": round(fold_f1, 4),
            })

        acc = accuracy_score(y_true_all, y_pred_all)
        macro_f1 = f1_score(y_true_all, y_pred_all, labels=target_labels, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true_all, y_pred_all, labels=target_labels, average="weighted", zero_division=0)
        macro_prec = precision_score(y_true_all, y_pred_all, labels=target_labels, average="macro", zero_division=0)
        macro_rec = recall_score(y_true_all, y_pred_all, labels=target_labels, average="macro", zero_division=0)

        # Per-class recall, precision, and F1 with explicit labels parameter
        report = classification_report(y_true_all, y_pred_all, labels=target_labels, output_dict=True, zero_division=0)

        # Fold-level bootstrap over 98 LOGO folds (1000 resamples)
        n_folds = len(fold_true)
        boot_recalls = {c: [] for c in target_labels}
        for _ in range(1000):
            sample_idx = rng.choice(n_folds, size=n_folds, replace=True)
            yt = np.concatenate([fold_true[i] for i in sample_idx])
            yp = np.concatenate([fold_pred[i] for i in sample_idx])
            rep = classification_report(yt, yp, labels=target_labels, output_dict=True, zero_division=0)
            for c in target_labels:
                boot_recalls[c].append(rep[c]['recall'])

        row_dict = {
            "Task": task_name,
            "Model": model_name,
            "Accuracy (%)": round(acc * 100, 2),
            "Macro F1": round(macro_f1, 4),
            "Macro Precision": round(macro_prec, 4),
            "Macro Recall": round(macro_rec, 4),
            "Weighted F1": round(weighted_f1, 4),
        }

        for c in target_labels:
            c_metrics = report.get(c, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})
            row_dict[f"Recall ({c})"] = round(c_metrics["recall"], 4)
            row_dict[f"Recall_CI_2.5 ({c})"] = round(float(np.percentile(boot_recalls[c], 2.5)), 4)
            row_dict[f"Recall_CI_97.5 ({c})"] = round(float(np.percentile(boot_recalls[c], 97.5)), 4)
            row_dict[f"Precision ({c})"] = round(c_metrics["precision"], 4)
            row_dict[f"F1 ({c})"] = round(c_metrics["f1-score"], 4)

            ci_summary_rows.append({
                "Task": task_name,
                "Model": model_name,
                "Class": c,
                "Point_Recall": round(c_metrics["recall"], 4),
                "Recall_CI_2.5": round(float(np.percentile(boot_recalls[c], 2.5)), 4),
                "Recall_CI_97.5": round(float(np.percentile(boot_recalls[c], 97.5)), 4),
            })

        summary_rows.append(row_dict)

    df_summary = pd.DataFrame(summary_rows)
    df_fold = pd.DataFrame(fold_detail_rows)
    df_ci = pd.DataFrame(ci_summary_rows)

    metadata = {
        "Task": task_name,
        "Total_Observed_Rows": total_samples,
        "Num_Locations": len(df["Location"].unique()),
        "Class_Distribution": class_counts,
        "Majority_Baseline_Class": majority_class,
        "Majority_Baseline_Accuracy_Pct": round(class_counts[majority_class] / total_samples * 100, 2),
    }

    return df_summary, df_fold, df_ci, metadata


def main() -> None:
    print("Loading real observed dataset (755 rows)...")
    df = load_and_preprocess_data()

    # 1. Primary Target: Binary BIS Classification (Good vs Not-Good)
    summary_binary, fold_binary, ci_binary, meta_binary = run_logo_evaluation(
        df, target_col="BIS_Binary", target_labels=TARGET_LABELS_BINARY, task_name="Binary_BIS_Hazard_Classification"
    )

    # 2. Secondary Target: 3-Class BIS Classification (Good, Poor, Very Poor)
    summary_3class, fold_3class, ci_3class, meta_3class = run_logo_evaluation(
        df, target_col="BIS_3Class", target_labels=TARGET_LABELS_3CLASS, task_name="3Class_BIS_Hazard_Classification"
    )

    # Save distinct CSV files for binary and 3-class tasks
    binary_csv_path = OUT_DIR / "logo_classification_binary.csv"
    summary_binary.to_csv(binary_csv_path, index=False)

    threeclass_csv_path = OUT_DIR / "logo_classification_3class.csv"
    summary_3class.to_csv(threeclass_csv_path, index=False)

    # Bootstrap CIs CSV
    ci_csv_path = OUT_DIR / "recall_bootstrap_ci.csv"
    df_combined_ci = pd.concat([ci_binary, ci_3class], ignore_index=True)
    df_combined_ci.to_csv(ci_csv_path, index=False)

    # Combined summary CSV
    summary_csv_path = OUT_DIR / "logo_classification_comparison.csv"
    df_combined_summary = pd.concat([summary_binary, summary_3class], ignore_index=True)
    df_combined_summary.to_csv(summary_csv_path, index=False)

    df_combined_folds = pd.concat([fold_binary, fold_3class], ignore_index=True)
    fold_csv_path = OUT_DIR / "logo_classification_fold_details.csv"
    df_combined_folds.to_csv(fold_csv_path, index=False)

    metadata_path = OUT_DIR / "logo_classification_summary.json"
    with open(metadata_path, "w") as f:
        json.dump({"Binary": meta_binary, "3Class": meta_3class}, f, indent=2)

    print(f"\n[Results Saved Successfully]")
    print(f"Binary CSV     : {binary_csv_path}")
    print(f"3-Class CSV    : {threeclass_csv_path}")
    print(f"Combined CSV   : {summary_csv_path}")
    print(f"Fold CSV       : {fold_csv_path}")
    print(f"JSON Meta      : {metadata_path}")

    print("\n--- Corrected BIS Classification Summary: Binary Task (Good vs Not-Good) ---")
    print(summary_binary[["Model", "Accuracy (%)", "Macro F1", "Macro Recall", "Recall (Good)", "Recall (Not-Good)"]].to_string(index=False))

    print("\n--- Corrected BIS Classification Summary: 3-Class Task (Good, Poor, Very Poor) ---")
    print(summary_3class[["Model", "Accuracy (%)", "Macro F1", "Macro Recall", "Recall (Good)", "Recall (Poor)", "Recall (Very Poor)"]].to_string(index=False))


if __name__ == "__main__":
    main()
