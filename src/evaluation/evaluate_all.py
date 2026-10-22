"""Complete training, evaluation, benchmarking, and artifact generation pipeline.

Executes:
1. Chronological dataset partitioning (Train: 371, Val: 93, Test: 116).
2. Leakage-free preprocessing (StandardScaler fit strictly on Train).
3. Training of Zero-Order Sugeno ANFIS and baselines (Random Forest, MLP).
4. Performance evaluation (RMSE, MAE, R², Max Error) across Train, Val, and Test.
5. Model serialization to models/.
6. Prediction exports to results/predictions/.
7. Metrics export to results/metrics/.
8. Publication-ready visualization plots to results/graphs/.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import compute_regression_metrics, format_metrics_table
from src.models.anfis import SugenoANFIS
from src.models.baselines import (
    get_mlp_baseline,
    get_model_parameter_count,
    get_random_forest_baseline,
)

FEATURE_COLUMNS = ["tv_ch", "tv_d", "tI_ch", "Tavg", "tTpeak_ch", "Eavg_ch", "Eavg_d"]
TARGET_COLUMN = "SOH"


def load_and_split_data(
    csv_path: str,
    n_test: int = 116,
    val_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]]]:
    """Load dataset and perform strictly chronological train/validation/test split.

    Args:
        csv_path: Path to battery1_features.csv.
        n_test: Fixed number of final rows reserved as untouched test set (116).
        val_ratio: Fraction of development set allocated to validation (0.2).

    Returns:
        Tuple of (full_df, splits_dict) containing data arrays and tracking metadata.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Feature dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if len(df) != 580:
        raise ValueError(f"Expected 580 rows in {csv_path}, found {len(df)}")

    # Verify input feature completeness and absence of leakage columns
    for feat in FEATURE_COLUMNS:
        if feat not in df.columns:
            raise KeyError(f"Required feature '{feat}' missing from dataset")
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' missing from dataset")

    n_total = len(df)
    n_dev = n_total - n_test
    n_val = int(round(n_dev * val_ratio))
    n_train = n_dev - n_val

    # Chronological partition
    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_dev].copy()
    df_test = df.iloc[n_dev:].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    X_train = df_train[FEATURE_COLUMNS].values
    y_train = df_train[TARGET_COLUMN].values

    X_val = df_val[FEATURE_COLUMNS].values
    y_val = df_val[TARGET_COLUMN].values

    X_test = df_test[FEATURE_COLUMNS].values
    y_test = df_test[TARGET_COLUMN].values

    splits = {
        "train": (X_train, y_train, df_train),
        "val": (X_val, y_val, df_val),
        "test": (X_test, y_test, df_test),
    }
    return df, splits


def train_and_benchmark_models(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    project_root: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], pd.DataFrame, StandardScaler]:
    """Train ANFIS, Random Forest, and MLP, and compute standardized metrics."""
    X_train, y_train, df_train = splits["train"]
    X_val, y_val, df_val = splits["val"]
    X_test, y_test, df_test = splits["test"]

    # 1. Preprocessing: Fit scaler STRICTLY on training data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("\n" + "=" * 70)
    print("MODEL TRAINING & EVALUATION BENCHMARK")
    print("=" * 70)
    print(f"Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"Target: {TARGET_COLUMN} (SOH %)")
    print(f"Train split : {len(y_train)} rows (CN {df_train['CN'].min()}..{df_train['CN'].max()})")
    print(f"Val split   : {len(y_val)} rows (CN {df_val['CN'].min()}..{df_val['CN'].max()})")
    print(f"Test split  : {len(y_test)} rows (CN {df_test['CN'].min()}..{df_test['CN'].max()}) [UNTOUCHED]")

    # 2. Instantiate models
    # Zero-Order Sugeno ANFIS (5 rules, Gaussian MFs, L2 lambda=0.05, Adam lr=0.01)
    anfis_model = SugenoANFIS(
        n_rules=5,
        order=0,
        lr=0.01,
        lambda_reg=0.05,
        n_epochs=100,
        random_state=42,
    )

    # Random Forest Regressor Baseline
    rf_model = get_random_forest_baseline(
        n_estimators=50,
        max_depth=12,
        min_samples_split=10,
        random_state=42,
    )

    # Multi-layer Perceptron (ANN) Baseline
    mlp_model = get_mlp_baseline(
        hidden_layer_sizes=(64, 32),
        alpha=1e-4,
        learning_rate_init=0.01,
        max_iter=1000,
        random_state=42,
    )

    models = {
        "ANFIS": anfis_model,
        "Random_Forest": rf_model,
        "MLP_ANN": mlp_model,
    }

    all_metrics: Dict[str, Dict[str, float]] = {}
    preds_dict: Dict[str, Dict[str, np.ndarray]] = {"ANFIS": {}, "Random_Forest": {}, "MLP_ANN": {}}

    for name, model in models.items():
        print(f"\nFitting {name}...")
        if name == "ANFIS":
            model.fit(X_train_s, y_train, X_val_s, y_val)
        else:
            model.fit(X_train_s, y_train)

        param_count = get_model_parameter_count(model)
        print(f"  {name} parameter count: {param_count}")

        # Compute predictions across splits
        p_train = model.predict(X_train_s)
        p_val = model.predict(X_val_s)
        p_test = model.predict(X_test_s)

        preds_dict[name]["train"] = p_train
        preds_dict[name]["val"] = p_val
        preds_dict[name]["test"] = p_test

        m_train = compute_regression_metrics(y_train, p_train, split_name="train")
        m_val = compute_regression_metrics(y_val, p_val, split_name="val")
        m_test = compute_regression_metrics(y_test, p_test, split_name="test")

        combined = {**m_train, **m_val, **m_test, "parameters": param_count}
        all_metrics[name] = combined

        print(f"  Train RMSE: {m_train['train_rmse_percent']:.4f}% | MAE: {m_train['train_mae_percent']:.4f}% | R2: {m_train['train_r2_score']:.4f}")
        print(f"  Val   RMSE: {m_val['val_rmse_percent']:.4f}% | MAE: {m_val['val_mae_percent']:.4f}% | R2: {m_val['val_r2_score']:.4f}")
        print(f"  Test  RMSE: {m_test['test_rmse_percent']:.4f}% | MAE: {m_test['test_mae_percent']:.4f}% | R2: {m_test['test_r2_score']:.4f}")

    # Build master predictions DataFrame
    df_all = pd.concat([df_train, df_val, df_test]).sort_values("global_cycle").reset_index(drop=True)
    all_y = df_all[TARGET_COLUMN].values

    # Concatenate predictions across splits for each model
    pred_df = pd.DataFrame({
        "CN": df_all["CN"],
        "global_cycle": df_all["global_cycle"],
        "source_file": df_all["source_file"],
        "source_cycle": df_all["source_cycle"],
        "split": df_all["split"],
        "actual_SOH": all_y,
    })

    for name in ["ANFIS", "Random_Forest", "MLP_ANN"]:
        full_pred = np.concatenate([
            preds_dict[name]["train"],
            preds_dict[name]["val"],
            preds_dict[name]["test"],
        ])
        pred_df[f"pred_{name}"] = np.round(full_pred, 4)
        pred_df[f"error_{name}"] = np.round(full_pred - all_y, 4)

    return models, all_metrics, pred_df, scaler


def save_artifacts(
    models: Dict[str, Any],
    all_metrics: Dict[str, Dict[str, float]],
    pred_df: pd.DataFrame,
    scaler: StandardScaler,
    project_root: str,
) -> None:
    """Save models, predictions, metrics tables, and plots to workspace directories."""
    models_dir = os.path.join(project_root, "models")
    metrics_dir = os.path.join(project_root, "results", "metrics")
    pred_dir = os.path.join(project_root, "results", "predictions")
    graphs_dir = os.path.join(project_root, "results", "graphs")

    for d in [models_dir, metrics_dir, pred_dir, graphs_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Serialize models & scaler
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(models["ANFIS"], os.path.join(models_dir, "anfis_model.joblib"))
    joblib.dump(models["Random_Forest"], os.path.join(models_dir, "rf_model.joblib"))
    joblib.dump(models["MLP_ANN"], os.path.join(models_dir, "mlp_model.joblib"))
    print(f"\n[OK] Serialized models and scaler to: {models_dir}")

    # 2. Save predictions CSV
    pred_csv_path = os.path.join(pred_dir, "all_model_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"[OK] Saved predictions table to: {pred_csv_path}")

    # 3. Save metrics JSON & CSV
    json_path = os.path.join(metrics_dir, "detailed_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Create tabular summary DataFrame for metrics
    rows = []
    for model_name, m in all_metrics.items():
        for split in ["train", "val", "test"]:
            rows.append({
                "model": model_name,
                "split": split,
                "rmse_percent": m[f"{split}_rmse_percent"],
                "mae_percent": m[f"{split}_mae_percent"],
                "max_error_percent": m[f"{split}_max_error_percent"],
                "r2_score": m[f"{split}_r2_score"],
                "parameters": m["parameters"],
            })
    metrics_summary_df = pd.DataFrame(rows)
    metrics_csv_path = os.path.join(metrics_dir, "model_comparison_metrics.csv")
    metrics_summary_df.to_csv(metrics_csv_path, index=False)
    print(f"[OK] Saved metrics artifacts to: {metrics_dir}")

    # 4. Generate Publication-Quality Figures
    generate_comparison_plots(pred_df, metrics_summary_df, models["ANFIS"], graphs_dir)


def generate_comparison_plots(
    pred_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    anfis_model: SugenoANFIS,
    graphs_dir: str,
) -> None:
    """Generate high-resolution comparative diagnostic plots."""
    cycles = pred_df["CN"].values
    y_actual = pred_df["actual_SOH"].values

    train_end_cn = pred_df[pred_df["split"] == "train"]["CN"].max()
    val_end_cn = pred_df[pred_df["split"] == "val"]["CN"].max()

    # --- Plot 1: Actual vs. Predicted Trajectory Across Chronological Splits ---
    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(cycles, y_actual, label="Actual Ground Truth SOH", color="black", linewidth=2.0, alpha=0.9)
    plt.plot(cycles, pred_df["pred_ANFIS"], label="ANFIS (Zero-Order Sugeno)", color="#1f77b4", linestyle="--", linewidth=1.8)
    plt.plot(cycles, pred_df["pred_MLP_ANN"], label="MLP (ANN Baseline)", color="#2ca02c", linestyle="-.", linewidth=1.5)
    plt.plot(cycles, pred_df["pred_Random_Forest"], label="Random Forest Baseline", color="#d62728", linestyle=":", linewidth=1.5)

    plt.axvline(train_end_cn, color="gray", linestyle="--", alpha=0.7, label=f"Train/Val Boundary (CN={train_end_cn})")
    plt.axvline(val_end_cn, color="blue", linestyle="--", alpha=0.7, label=f"Val/Test Boundary (CN={val_end_cn})")

    plt.title("State of Health (SOH) Estimation: ANFIS vs. Baselines Across 580 Cycles", fontsize=14, pad=12)
    plt.xlabel("Cycle Number (CN)", fontsize=12)
    plt.ylabel("Battery State of Health (SOH %)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    p1_path = os.path.join(graphs_dir, "actual_vs_predicted_all_models.png")
    plt.savefig(p1_path)
    plt.close()

    # --- Plot 2: Residuals and Error Distributions on Test Set ---
    test_df = pred_df[pred_df["split"] == "test"]
    plt.figure(figsize=(10, 5), dpi=300)
    plt.subplot(1, 2, 1)
    plt.scatter(test_df["CN"], test_df["error_ANFIS"], color="#1f77b4", label="ANFIS", alpha=0.7, s=25)
    plt.scatter(test_df["CN"], test_df["error_MLP_ANN"], color="#2ca02c", label="MLP", alpha=0.7, s=25)
    plt.scatter(test_df["CN"], test_df["error_Random_Forest"], color="#d62728", label="RF", alpha=0.7, s=25)
    plt.axhline(0, color="black", linestyle="--", alpha=0.6)
    plt.title("Test Set Residuals vs. Cycle Number", fontsize=11)
    plt.xlabel("Cycle Number (CN)", fontsize=10)
    plt.ylabel("Residual Error (% SOH)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(test_df["error_ANFIS"], bins=15, alpha=0.5, color="#1f77b4", label="ANFIS", density=True)
    plt.hist(test_df["error_MLP_ANN"], bins=15, alpha=0.5, color="#2ca02c", label="MLP", density=True)
    plt.hist(test_df["error_Random_Forest"], bins=15, alpha=0.5, color="#d62728", label="RF", density=True)
    plt.title("Test Error Distribution", fontsize=11)
    plt.xlabel("Residual Error (% SOH)", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    p2_path = os.path.join(graphs_dir, "residuals_distribution_comparison.png")
    plt.savefig(p2_path)
    plt.close()

    # --- Plot 3: Metrics Comparison Bar Chart ---
    plt.figure(figsize=(10, 5), dpi=300)
    models = ["ANFIS", "MLP_ANN", "Random_Forest"]
    splits_list = ["train", "val", "test"]
    x = np.arange(len(models))
    width = 0.25

    plt.subplot(1, 2, 1)
    for i, s in enumerate(splits_list):
        vals = [metrics_df[(metrics_df["model"] == m) & (metrics_df["split"] == s)]["rmse_percent"].values[0] for m in models]
        plt.bar(x + (i - 1) * width, vals, width, label=s.upper())
    plt.xticks(x, ["ANFIS", "MLP (ANN)", "Random Forest"])
    plt.ylabel("RMSE (% SOH)", fontsize=11)
    plt.title("RMSE Comparison by Split", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, s in enumerate(splits_list):
        vals = [metrics_df[(metrics_df["model"] == m) & (metrics_df["split"] == s)]["mae_percent"].values[0] for m in models]
        plt.bar(x + (i - 1) * width, vals, width, label=s.upper())
    plt.xticks(x, ["ANFIS", "MLP (ANN)", "Random Forest"])
    plt.ylabel("MAE (% SOH)", fontsize=11)
    plt.title("MAE Comparison by Split", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.legend()

    plt.tight_layout()
    p3_path = os.path.join(graphs_dir, "model_metrics_comparison_bar.png")
    plt.savefig(p3_path)
    plt.close()

    # --- Plot 4: Learned Gaussian Membership Functions (ANFIS) ---
    plt.figure(figsize=(12, 6), dpi=300)
    x_range = np.linspace(-3.0, 3.0, 300)
    # Plot MFs for top 4 correlated features (indices: 0=tv_ch, 1=tv_d, 2=tI_ch, 5=Eavg_ch)
    top_feats = [(2, "tI_ch"), (0, "tv_ch"), (1, "tv_d"), (5, "Eavg_ch")]

    for idx, (feat_idx, feat_name) in enumerate(top_feats, start=1):
        plt.subplot(2, 2, idx)
        for r in range(anfis_model.n_rules):
            c = anfis_model.centers_[r, feat_idx]
            s = anfis_model.sigmas_[r, feat_idx]
            mf = np.exp(-0.5 * ((x_range - c) / s) ** 2)
            plt.plot(x_range, mf, label=f"Rule {r+1}")
        plt.title(f"Gaussian MFs for {feat_name} (Normalized)", fontsize=10)
        plt.xlabel("Standardized Feature Value", fontsize=9)
        plt.ylabel("Degree of Membership", fontsize=9)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle=":", alpha=0.5)
        if idx == 1:
            plt.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    p4_path = os.path.join(graphs_dir, "anfis_membership_functions.png")
    plt.savefig(p4_path)
    plt.close()

    print(f"[OK] Generated publication-quality figures in: {graphs_dir}")


def run_pipeline() -> None:
    """Execute complete benchmarking pipeline from project root."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(project_root, "data", "processed", "battery1_features.csv")

    df, splits = load_and_split_data(csv_path, n_test=116, val_ratio=0.2)
    models, all_metrics, pred_df, scaler = train_and_benchmark_models(splits, project_root)
    save_artifacts(models, all_metrics, pred_df, scaler, project_root)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (TABULAR)")
    print("=" * 70)
    print(format_metrics_table(all_metrics))
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
