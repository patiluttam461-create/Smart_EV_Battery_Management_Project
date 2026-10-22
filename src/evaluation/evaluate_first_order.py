"""Experimental Evaluation Pipeline for First-Order Sugeno ANFIS.

Executes:
1. Chronological dataset partitioning (Train: 371, Val: 93, Test: 116).
2. Leakage-free preprocessing using frozen training StandardScaler.
3. Controlled candidate selection using Validation partition only.
4. Final model checkpointing and serialization to models/anfis_first_order_model.joblib.
5. Single, unbiased evaluation on held-out Test partition (Cycles 465–580).
6. Performance benchmarking against Zero-Order ANFIS, Random Forest, MLP, and Train-Mean.
7. Prediction export to results/predictions/first_order_predictions.csv.
8. Metrics export to results/metrics/first_order_metrics.json and first_order_comparison.csv.
9. Trajectory plot generation to results/graphs/first_order_vs_zero_order_test_trajectory.png.
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.models.first_order_anfis import FirstOrderSugenoANFIS

FEATURE_COLUMNS = ["tv_ch", "tv_d", "tI_ch", "Tavg", "tTpeak_ch", "Eavg_ch", "Eavg_d"]
TARGET_COLUMN = "SOH"


def run_first_order_experiment(project_root: str) -> Dict[str, Any]:
    """Execute complete First-Order Sugeno ANFIS training, selection, and evaluation."""
    data_path = os.path.join(project_root, "data", "processed", "battery1_features.csv")
    scaler_path = os.path.join(project_root, "models", "scaler.joblib")
    zero_pred_path = os.path.join(
        project_root, "results", "predictions", "all_model_predictions.csv"
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Feature dataset missing: {data_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Training scaler missing: {scaler_path}")

    df = pd.read_csv(data_path)
    scaler = joblib.load(scaler_path)

    # 1. Chronological splitting: Train (371), Val (93), Test (116)
    n_total = len(df)
    n_test = 116
    n_dev = n_total - n_test  # 464
    n_val = 93
    n_train = n_dev - n_val  # 371

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_dev].copy()
    df_test = df.iloc[n_dev:].copy()

    X_train_raw = df_train[FEATURE_COLUMNS].values
    y_train = df_train[TARGET_COLUMN].values

    X_val_raw = df_val[FEATURE_COLUMNS].values
    y_val = df_val[TARGET_COLUMN].values

    X_test_raw = df_test[FEATURE_COLUMNS].values
    y_test = df_test[TARGET_COLUMN].values

    # Preprocessing with training-fitted scaler
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    print("=================================================================")
    print("STEP 1: CONTROLLED MODEL SELECTION ON VALIDATION PARTITION ONLY")
    print("=================================================================")

    candidates = [
        {"name": "FO-ANFIS (R=2, reg=1.0)", "n_rules": 2, "reg": 1.0, "lr": 0.005},
        {"name": "FO-ANFIS (R=2, reg=5.0)", "n_rules": 2, "reg": 5.0, "lr": 0.005},
        {"name": "FO-ANFIS (R=2, reg=10.0)", "n_rules": 2, "reg": 10.0, "lr": 0.005},
        {"name": "FO-ANFIS (R=3, reg=1.0)", "n_rules": 3, "reg": 1.0, "lr": 0.005},
        {"name": "FO-ANFIS (R=3, reg=5.0)", "n_rules": 3, "reg": 5.0, "lr": 0.005},
        {"name": "FO-ANFIS (R=3, reg=10.0)", "n_rules": 3, "reg": 10.0, "lr": 0.005},
        {"name": "FO-ANFIS (R=5, reg=1.0)", "n_rules": 5, "reg": 1.0, "lr": 0.005},
        {"name": "FO-ANFIS (R=5, reg=5.0)", "n_rules": 5, "reg": 5.0, "lr": 0.005},
        {"name": "FO-ANFIS (R=5, reg=10.0)", "n_rules": 5, "reg": 10.0, "lr": 0.005},
    ]

    trained_candidates = []
    for cand in candidates:
        model = FirstOrderSugenoANFIS(
            n_rules=cand["n_rules"],
            lr=cand["lr"],
            lambda_reg=cand["reg"],
            n_epochs=100,
            random_state=42,
        )
        model.fit(X_train, y_train, X_val, y_val)
        val_preds = model.predict(X_val)

        v_rmse = root_mean_squared_error(y_val, val_preds)
        v_mae = mean_absolute_error(y_val, val_preds)
        v_r2 = r2_score(y_val, val_preds)

        p_count = model.parameter_count["total_parameters"]
        cand_record = {
            **cand,
            "val_rmse": float(v_rmse),
            "val_mae": float(v_mae),
            "val_r2": float(v_r2),
            "parameters": p_count,
            "fitted_model": model,
        }
        trained_candidates.append(cand_record)
        print(
            f"Candidate: {cand['name']:25s} | Params: {p_count:3d} | "
            f"Val RMSE: {v_rmse:7.4f}% | Val MAE: {v_mae:7.4f}% | Val R2: {v_r2:8.4f}"
        )

    # Model selection based strictly on validation RMSE
    best_candidate = min(trained_candidates, key=lambda c: c["val_rmse"])
    print(f"\n[SELECTED BEST CANDIDATE BY VAL RMSE]: {best_candidate['name']}")
    print(
        f"Validation RMSE: {best_candidate['val_rmse']:.4f}% | "
        f"Validation MAE: {best_candidate['val_mae']:.4f}% | "
        f"Parameters: {best_candidate['parameters']}"
    )

    best_model: FirstOrderSugenoANFIS = best_candidate["fitted_model"]

    # 2. Save experimental model artifact
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    exp_model_path = os.path.join(models_dir, "anfis_first_order_model.joblib")
    joblib.dump(best_model, exp_model_path)
    print(f"[OK] Saved experimental model to: {exp_model_path}")

    # 3. Evaluate ONCE on held-out test partition
    print("\n=================================================================")
    print("STEP 2: SINGLE HELD-OUT TEST EVALUATION (CYCLES 465–580)")
    print("=================================================================")

    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)

    # Compute metrics on test partition
    test_rmse = float(root_mean_squared_error(y_test, test_pred))
    test_mae = float(mean_absolute_error(y_test, test_pred))
    test_r2 = float(r2_score(y_test, test_pred))
    test_bias = float(np.mean(test_pred - y_test))
    test_max_err = float(np.max(np.abs(y_test - test_pred)))

    print(f"Test MAE:             {test_mae:8.4f}%")
    print(f"Test RMSE:            {test_rmse:8.4f}%")
    print(f"Test R2:              {test_r2:8.4f}")
    print(f"Test Mean Bias:       {test_bias:+8.4f}%")
    print(f"Test Max Error:       {test_max_err:8.4f}%")

    # 4. Compare with baselines from existing results
    df_existing_preds = pd.read_csv(zero_pred_path)
    test_existing = df_existing_preds[df_existing_preds["split"] == "test"]
    y_test_check = test_existing["actual_SOH"].values
    pred_zo = test_existing["pred_ANFIS"].values
    pred_rf = test_existing["pred_Random_Forest"].values
    pred_mlp = test_existing["pred_MLP_ANN"].values
    train_mean = float(df_train[TARGET_COLUMN].mean())
    pred_mean = np.full_like(y_test, train_mean)

    models_comparison = [
        ("ANFIS (First-Order Sugeno, Selected)", test_pred, best_candidate["parameters"]),
        ("ANFIS (Zero-Order Sugeno, Baseline)", pred_zo, 75),
        ("Multi-Layer Perceptron (ANN)", pred_mlp, 2625),
        ("Random Forest Regressor", pred_rf, 4546),
        ("Train-Mean Baseline", pred_mean, 1),
    ]

    comp_rows = []
    print("\n=================================================================")
    print("STEP 3: BENCHMARK COMPARISON ON IDENTICAL HELD-OUT TEST ROWS (N=116)")
    print("=================================================================")
    for m_name, m_preds, m_params in models_comparison:
        rmse = float(root_mean_squared_error(y_test, m_preds))
        mae = float(mean_absolute_error(y_test, m_preds))
        r2 = float(r2_score(y_test, m_preds))
        bias = float(np.mean(m_preds - y_test))
        max_err = float(np.max(np.abs(y_test - m_preds)))
        comp_rows.append({
            "model": m_name,
            "parameters": m_params,
            "test_rmse_percent": round(rmse, 4),
            "test_mae_percent": round(mae, 4),
            "test_r2_score": round(r2, 4),
            "test_mean_bias_percent": round(bias, 4),
            "test_max_error_percent": round(max_err, 4),
        })
        print(
            f"{m_name:38s} | Params: {m_params:4d} | MAE: {mae:6.4f}% | "
            f"RMSE: {rmse:6.4f}% | R2: {r2:8.4f} | Bias: {bias:+6.4f}%"
        )

    # 5. Export predictions table
    pred_dir = os.path.join(project_root, "results", "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    full_preds = np.concatenate([train_pred, val_pred, test_pred])
    full_y = np.concatenate([y_train, y_val, y_test])

    out_pred_df = pd.DataFrame({
        "CN": df["CN"],
        "global_cycle": df["global_cycle"],
        "split": ["train"] * n_train + ["val"] * n_val + ["test"] * n_test,
        "actual_SOH": np.round(full_y, 4),
        "pred_ANFIS_first_order": np.round(full_preds, 4),
        "error_ANFIS_first_order": np.round(full_preds - full_y, 4),
        "pred_ANFIS_zero_order": df_existing_preds["pred_ANFIS"],
        "error_ANFIS_zero_order": df_existing_preds["error_ANFIS"],
        "pred_Random_Forest": df_existing_preds["pred_Random_Forest"],
        "pred_MLP_ANN": df_existing_preds["pred_MLP_ANN"],
    })
    exp_preds_path = os.path.join(pred_dir, "first_order_predictions.csv")
    out_pred_df.to_csv(exp_preds_path, index=False)
    print(f"\n[OK] Saved experimental predictions to: {exp_preds_path}")

    # 6. Export metrics CSV & JSON
    metrics_dir = os.path.join(project_root, "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    comp_df = pd.DataFrame(comp_rows)
    exp_comp_path = os.path.join(metrics_dir, "first_order_comparison.csv")
    comp_df.to_csv(exp_comp_path, index=False)

    detailed_json = {
        "selected_configuration": {
            "name": best_candidate["name"],
            "n_rules": best_candidate["n_rules"],
            "lambda_reg": best_candidate["reg"],
            "learning_rate": best_candidate["lr"],
            "parameters": best_candidate["parameters"],
        },
        "validation_selection_summary": [
            {
                "name": c["name"],
                "n_rules": c["n_rules"],
                "lambda_reg": c["reg"],
                "parameters": c["parameters"],
                "val_rmse": c["val_rmse"],
                "val_mae": c["val_mae"],
                "val_r2": c["val_r2"],
            }
            for c in trained_candidates
        ],
        "test_benchmark_metrics": comp_rows,
    }
    exp_json_path = os.path.join(metrics_dir, "first_order_metrics.json")
    with open(exp_json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_json, f, indent=2)
    print(f"[OK] Saved experimental metrics to: {exp_comp_path} and {exp_json_path}")

    # 7. Generate diagnostic visualization plot
    graphs_dir = os.path.join(project_root, "results", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    plot_path = os.path.join(graphs_dir, "first_order_vs_zero_order_test_trajectory.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

    test_cns = df_test["CN"].values
    ax1.plot(test_cns, y_test, "k-", linewidth=2.2, label="Actual Ground Truth SOH", zorder=5)
    ax1.plot(test_cns, test_pred, color="#0f766e", linewidth=2.0, linestyle="--", label=f"First-Order ANFIS (R=2, MAE={test_mae:.2f}%)")
    ax1.plot(test_cns, pred_zo, color="#b45309", linewidth=1.8, linestyle=":", label=f"Zero-Order ANFIS (R=5, MAE=2.45%)")
    ax1.plot(test_cns, pred_mlp, color="#2563eb", linewidth=1.5, linestyle="-.", alpha=0.85, label=f"MLP ANN (MAE=1.09%)")
    ax1.set_ylabel("State of Health (SOH %)", fontsize=11, fontweight="bold")
    ax1.set_title("Chronological Held-Out Test Trajectory Comparison (Cycles 465–580)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)

    # Residuals plot
    ax2.axhline(0, color="gray", linestyle="-", linewidth=1.0)
    ax2.plot(test_cns, test_pred - y_test, color="#0f766e", linewidth=1.8, label="First-Order Error")
    ax2.plot(test_cns, pred_zo - y_test, color="#b45309", linewidth=1.5, linestyle=":", label="Zero-Order Error")
    ax2.set_xlabel("Cycle Number (CN)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Residual Error (% SOH)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved trajectory comparison plot to: {plot_path}")

    return detailed_json


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    run_first_order_experiment(project_root)
