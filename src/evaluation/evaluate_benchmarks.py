"""Evaluation of operational forecasting baselines and expanding-window cross-validation.

Distinguishes:
1. Fixed Chronological Holdout (Cycles 465–580, N=116) model comparison.
2. Operational Forecasting Baselines (Train-Mean, Persistence, Linear Trend) vs. Oracle Test-Mean.
3. Expanding-Window Time-Series Cross-Validation across 4 temporal partitions.

Documents the exact information available to each baseline at prediction time.
"""

import json
import os
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.models.first_order_anfis import FirstOrderSugenoANFIS

FEATURE_COLUMNS = ["tv_ch", "tv_d", "tI_ch", "Tavg", "tTpeak_ch", "Eavg_ch", "Eavg_d"]
TARGET_COLUMN = "SOH"


def evaluate_operational_baselines(project_root: str) -> pd.DataFrame:
    """Evaluate operational forecasting baselines against held-out test cycles."""
    data_path = os.path.join(project_root, "data", "processed", "battery1_features.csv")
    df = pd.read_csv(data_path)

    train_df = df.iloc[:371]
    val_df = df.iloc[371:464]
    test_df = df.iloc[464:]

    y_train = train_df[TARGET_COLUMN].values
    cn_train = train_df["CN"].values

    y_val = val_df[TARGET_COLUMN].values
    cn_val = val_df["CN"].values

    y_test = test_df[TARGET_COLUMN].values
    cn_test = test_df["CN"].values

    # 1. Train-Mean Baseline
    p_train_mean = np.full_like(y_test, np.mean(y_train))

    # 2. Persistence from Last Training Cycle
    p_last_train = np.full_like(y_test, y_train[-1])

    # 3. Persistence from Last Validation Cycle
    p_last_val = np.full_like(y_test, y_val[-1])

    # 4. Linear Aging Trend fit on training cycle number
    poly_cn = np.polyfit(cn_train, y_train, 1)
    p_linear_trend = np.polyval(poly_cn, cn_test)

    # 5. Theoretical Oracle Constant Test-Mean (Non-deployable)
    p_test_mean_oracle = np.full_like(y_test, np.mean(y_test))

    baseline_definitions = [
        {
            "baseline_name": "Train-Mean Baseline",
            "deployable": True,
            "information_available": "Training set mean SOH only (Cycles 1–371; 90.9634%)",
            "parameters": 1,
            "predictions": p_train_mean,
        },
        {
            "baseline_name": "Persistence (Last Known Train SOH)",
            "deployable": True,
            "information_available": "SOH at final training calibration cycle (Cycle 371; 86.7921%)",
            "parameters": 1,
            "predictions": p_last_train,
        },
        {
            "baseline_name": "Persistence (Last Known Val SOH)",
            "deployable": True,
            "information_available": "SOH at final pre-deployment validation cycle (Cycle 464; 85.0732%)",
            "parameters": 1,
            "predictions": p_last_val,
        },
        {
            "baseline_name": "Linear Aging Trend on CN",
            "deployable": True,
            "information_available": "OLS linear degradation slope & intercept fit on training cycles (1–371)",
            "parameters": 2,
            "predictions": p_linear_trend,
        },
        {
            "baseline_name": "Oracle Test-Mean (Non-Deployable Reference)",
            "deployable": False,
            "information_available": "Future held-out test labels mean (84.2670%; theoretical oracle lower bound)",
            "parameters": 1,
            "predictions": p_test_mean_oracle,
        },
    ]

    records = []
    for b in baseline_definitions:
        preds = b["predictions"]
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(root_mean_squared_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        bias = float(np.mean(preds - y_test))
        max_err = float(np.max(np.abs(preds - y_test)))

        records.append({
            "baseline_name": b["baseline_name"],
            "deployable": b["deployable"],
            "parameters": b["parameters"],
            "test_rmse_percent": round(rmse, 4),
            "test_mae_percent": round(mae, 4),
            "test_r2_score": round(r2, 4),
            "test_mean_bias_percent": round(bias, 4),
            "test_max_error_percent": round(max_err, 4),
            "information_available": b["information_available"],
        })

    out_df = pd.DataFrame(records)
    out_csv = os.path.join(project_root, "results", "metrics", "operational_baselines.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] Operational baselines exported to: {out_csv}")
    return out_df


def evaluate_expanding_window(project_root: str) -> pd.DataFrame:
    """Run expanding-window chronological cross-validation across 4 progressive folds."""
    data_path = os.path.join(project_root, "data", "processed", "battery1_features.csv")
    df = pd.read_csv(data_path)

    X_all = df[FEATURE_COLUMNS].values
    y_all = df[TARGET_COLUMN].values

    folds = [
        ("Fold 1: CN 201–300", 200, 300),
        ("Fold 2: CN 301–400", 300, 400),
        ("Fold 3: CN 401–500", 400, 500),
        ("Fold 4: CN 501–580", 500, 580),
    ]

    fold_records = []
    for name, train_end, test_end in folds:
        X_tr, y_tr = X_all[:train_end], y_all[:train_end]
        X_te, y_te = X_all[train_end:test_end], y_all[train_end:test_end]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Train a 2-rule First-Order Sugeno ANFIS with lambda=5.0
        model = FirstOrderSugenoANFIS(
            n_rules=2, lr=0.005, lambda_reg=5.0, n_epochs=50, random_state=42
        )
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_te_s)

        mae = float(mean_absolute_error(y_te, preds))
        rmse = float(root_mean_squared_error(y_te, preds))
        r2 = float(r2_score(y_te, preds))
        bias = float(np.mean(preds - y_te))
        max_err = float(np.max(np.abs(preds - y_te)))

        fold_records.append({
            "fold_name": name,
            "train_samples": train_end,
            "test_samples": len(y_te),
            "test_rmse_percent": round(rmse, 4),
            "test_mae_percent": round(mae, 4),
            "test_r2_score": round(r2, 4),
            "test_mean_bias_percent": round(bias, 4),
            "test_max_error_percent": round(max_err, 4),
        })

    exp_df = pd.DataFrame(fold_records)
    out_csv = os.path.join(
        project_root, "results", "metrics", "expanding_window_evaluation.csv"
    )
    exp_df.to_csv(out_csv, index=False)
    print(f"[OK] Expanding-window metrics exported to: {out_csv}")
    return exp_df


def main() -> None:
    """Execute complete operational baseline and expanding-window evaluation."""
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    print("=================================================================")
    print("STEP 1: OPERATIONAL DEPLOYABLE BASELINES ON HELD-OUT TEST CYCLES")
    print("=================================================================")
    b_df = evaluate_operational_baselines(project_root)
    print(b_df[["baseline_name", "deployable", "test_mae_percent", "test_rmse_percent", "test_r2_score"]])

    print("\n=================================================================")
    print("STEP 2: EXPANDING-WINDOW TIME-SERIES EVALUATION")
    print("=================================================================")
    e_df = evaluate_expanding_window(project_root)
    print(e_df[["fold_name", "train_samples", "test_samples", "test_mae_percent", "test_r2_score"]])


if __name__ == "__main__":
    main()
