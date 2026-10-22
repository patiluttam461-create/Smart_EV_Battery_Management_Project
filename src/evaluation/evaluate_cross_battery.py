"""Cross-Battery Generalization Evaluation Pipeline.

Evaluates generalization capability of Adaptive Neuro-Fuzzy Inference Systems (ANFIS)
trained exclusively on Samsung INR18650-35E Battery 1 when deployed on unseen Battery 2 cycles.

Key Scientific Principles:
1. Strict Battery Separation: Preprocessing scalers and model parameters are fit
   strictly on Battery 1 training partition (Cycles 1–371). Zero data leakage.
2. Canonical Schema Parity: Enforces identical 7 health indicators, feature ordering,
   and units across both batteries.
3. Unbiased Validation-Only Bias Calibration: Evaluates whether systematic bias
   calibrated strictly on Battery 1 validation split (Cycles 372–464) transfers to Battery 2,
   without ever accessing Battery 1 test labels or Battery 2 data.
4. Naive Baseline Benchmarks: Evaluates performance against Train-Mean, Persistence
   (Final B1 Cycle and Initial B2 State), and Linear Trend aging extrapolation.
5. Strict Academic Integrity: Halts with an informative error if Battery 2 raw or processed
   data is missing. Never fabricates measurements or reference labels.
6. Isolated Artifacts: Exports results to dedicated cross_battery_* files without
   overwriting any production model checkpoints or existing evaluation metrics.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path when executed directly
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.models.first_order_anfis import FirstOrderSugenoANFIS

# Canonical 7 health indicators and target
FEATURE_COLUMNS: List[str] = [
    "tv_ch",
    "tv_d",
    "tI_ch",
    "Tavg",
    "tTpeak_ch",
    "Eavg_ch",
    "Eavg_d",
]
TARGET_COLUMN: str = "SOH"


def load_battery1_splits(
    battery1_csv_path: str,
    n_train: int = 371,
    n_val: int = 93,
    n_test: int = 116,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """Load Battery 1 dataset and partition chronologically into Train, Val, and Test.

    Args:
        battery1_csv_path: Path to data/processed/battery1_features.csv.
        n_train: Number of chronological training cycles (default: 371).
        n_val: Number of chronological validation cycles (default: 93).
        n_test: Number of chronological test cycles (default: 116).

    Returns:
        Dict mapping split names ('train', 'val', 'test') to (X_raw, y, df_split).
    """
    if not os.path.exists(battery1_csv_path):
        raise FileNotFoundError(
            f"Battery 1 dataset missing at: {battery1_csv_path}\n"
            f"Please verify data/processed/battery1_features.csv is present."
        )

    df = pd.read_csv(battery1_csv_path)
    total_expected = n_train + n_val + n_test
    if len(df) != total_expected:
        raise ValueError(
            f"Expected {total_expected} rows in Battery 1 dataset, found {len(df)}."
        )

    n_dev = n_train + n_val
    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_dev].copy()
    df_test = df.iloc[n_dev:].copy()

    splits = {
        "train": (df_train[FEATURE_COLUMNS].values, df_train[TARGET_COLUMN].values, df_train),
        "val": (df_val[FEATURE_COLUMNS].values, df_val[TARGET_COLUMN].values, df_val),
        "test": (df_test[FEATURE_COLUMNS].values, df_test[TARGET_COLUMN].values, df_test),
    }
    return splits


def fit_training_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit StandardScaler strictly on Battery 1 training features.

    Args:
        X_train: Raw training feature array of shape (N_train, 7).

    Returns:
        Fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def train_experimental_first_order_anfis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_rules: int = 2,
    lambda_reg: float = 5.0,
    lr: float = 0.005,
    n_epochs: int = 100,
    random_state: int = 42,
) -> FirstOrderSugenoANFIS:
    """Train an experimental First-Order Sugeno ANFIS on Battery 1 data.

    Uses the verified optimal regularization hyperparameters (R=2, lambda=5.0).

    Args:
        X_train: Scaled training features.
        y_train: Training SOH targets.
        X_val: Scaled validation features.
        y_val: Validation SOH targets.
        n_rules: Number of fuzzy rules (default: 2).
        lambda_reg: L2 regularization parameter for consequent least squares (default: 5.0).
        lr: Learning rate for premise parameters (default: 0.005).
        n_epochs: Training epochs (default: 100).
        random_state: Random seed for initialization reproducibility (default: 42).

    Returns:
        Fitted FirstOrderSugenoANFIS instance.
    """
    model = FirstOrderSugenoANFIS(
        n_rules=n_rules,
        lr=lr,
        lambda_reg=lambda_reg,
        n_epochs=n_epochs,
        random_state=random_state,
    )
    model.fit(X_train, y_train, X_val, y_val)
    return model


def compute_validation_bias(
    model: FirstOrderSugenoANFIS,
    scaler: StandardScaler,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Calculate mean signed error (bias) strictly on Battery 1 validation split.

    Formula: mean(y_pred_val - y_val).
    Positive value indicates the model systematically overpredicts SOH.
    Never uses Battery 1 test data or Battery 2 data.

    Args:
        model: Fitted ANFIS model.
        scaler: Scaler fitted on training split.
        X_val_raw: Raw validation feature array.
        y_val: Ground-truth validation SOH array.

    Returns:
        Validation mean signed error in percentage points.
    """
    X_val_scaled = scaler.transform(X_val_raw)
    val_pred = model.predict(X_val_scaled)
    val_bias = float(np.mean(val_pred - y_val))
    return val_bias


def compute_naive_baselines(
    y_train_b1: np.ndarray,
    cn_train_b1: np.ndarray,
    y_b2: np.ndarray,
    cn_b2: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute naive operational baselines for Battery 2.

    Baselines:
    1. Train-Mean: Scalar mean of Battery 1 training SOH.
    2. Persistence (Final B1 Cycle): Last observed SOH in Battery 1 training split.
    3. Persistence (Initial B2 State): First cycle SOH of Battery 2 (nominal fresh state).
    4. Linear Trend: OLS linear degradation line fit on (CN, SOH) of Battery 1 train,
       extrapolated to Battery 2 cycle numbers.

    Args:
        y_train_b1: Battery 1 training SOH array.
        cn_train_b1: Battery 1 training cycle numbers (CN).
        y_b2: Battery 2 actual SOH array.
        cn_b2: Battery 2 cycle numbers (CN).

    Returns:
        Dict mapping baseline name to prediction vector for Battery 2.
    """
    n_b2 = len(y_b2)

    # 1. Train-Mean
    train_mean = float(np.mean(y_train_b1))
    pred_mean = np.full(n_b2, train_mean, dtype=np.float64)

    # 2. Persistence (Final B1 Cycle)
    pred_persist_b1 = np.full(n_b2, float(y_train_b1[-1]), dtype=np.float64)

    # 3. Persistence (Initial B2 State)
    pred_persist_b2 = np.full(n_b2, float(y_b2[0]), dtype=np.float64)

    # 4. Linear Trend Extrapolation
    ols = LinearRegression()
    ols.fit(cn_train_b1.reshape(-1, 1), y_train_b1)
    pred_linear_trend = ols.predict(cn_b2.reshape(-1, 1)).astype(np.float64)

    return {
        "Train-Mean Baseline": pred_mean,
        "Persistence Baseline (Final B1 Train Cycle)": pred_persist_b1,
        "Persistence Baseline (Initial B2 State)": pred_persist_b2,
        "Linear Trend Extrapolation Baseline": pred_linear_trend,
    }


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard regression metrics between ground truth and predicted SOH.

    Metrics:
    - MAE: Mean Absolute Error (% SOH)
    - RMSE: Root Mean Squared Error (% SOH)
    - R2: Coefficient of determination
    - Mean Bias: Mean signed error (% SOH, pred - true)
    - Max Error: Maximum absolute error (% SOH)

    Args:
        y_true: Ground truth SOH array in percentage points.
        y_pred: Predicted SOH array in percentage points.

    Returns:
        Dict of computed metrics rounded to 4 decimal places.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true_arr)} != len(y_pred)={len(y_pred_arr)}"
        )
    if len(y_true_arr) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")
    if np.any(~np.isfinite(y_true_arr)) or np.any(~np.isfinite(y_pred_arr)):
        raise ValueError("Non-finite values (NaN / Inf) detected in evaluation arrays.")

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    max_err = float(max_error(y_true_arr, y_pred_arr))
    bias = float(np.mean(y_pred_arr - y_true_arr))

    var_y = float(np.var(y_true_arr))
    if var_y < 1e-12:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true_arr, y_pred_arr))

    return {
        "rmse_percent": round(rmse, 4),
        "mae_percent": round(mae, 4),
        "r2_score": round(r2, 4),
        "mean_bias_percent": round(bias, 4),
        "max_error_percent": round(max_err, 4),
    }


def run_cross_battery_evaluation(
    project_root: Optional[str] = None,
    battery1_csv_path: Optional[str] = None,
    battery2_csv_path: Optional[str] = None,
    save_artifacts: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute complete cross-battery evaluation pipeline.

    1. Loads Battery 1 data and fits scaler strictly on training split (Cycles 1–371).
    2. Trains experimental First-Order Sugeno ANFIS (R=2, lambda=5.0).
    3. Calculates validation bias strictly on Battery 1 validation split (Cycles 372–464).
    4. Loads Battery 2 dataset (or raises descriptive FileNotFoundError if missing).
    5. Validates identical 7 canonical features, order, and schema parity.
    6. Predicts Battery 2 SOH using uncalibrated and validation-calibrated ANFIS.
    7. Computes naive baseline benchmarks (Train-Mean, Persistence, Linear Trend).
    8. Exports comparison CSV, detailed JSON, predictions CSV, and experimental models.

    Args:
        project_root: Path to repository root. Defaults to auto-detection.
        battery1_csv_path: Optional explicit path to Battery 1 CSV.
        battery2_csv_path: Optional explicit path to Battery 2 CSV.
        save_artifacts: Whether to write result files to results/ and models/.
        verbose: Whether to print progress and evaluation tables.

    Returns:
        Dict containing metrics, comparisons, and paths to saved artifacts.

    Raises:
        FileNotFoundError: If Battery 1 or Battery 2 data is unavailable.
        ValueError: If schema, dimensions, or inputs are invalid.
    """
    if project_root is None:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

    # 1. Resolve Battery 1 dataset path
    if battery1_csv_path is None:
        battery1_csv_path = os.path.join(
            project_root, "data", "processed", "battery1_features.csv"
        )

    if not os.path.exists(battery1_csv_path):
        raise FileNotFoundError(
            f"Battery 1 dataset missing at: {battery1_csv_path}\n"
            f"Please run feature extraction for Battery 1 first."
        )

    # 2. Resolve Battery 2 dataset path
    if battery2_csv_path is None:
        battery2_csv_path = os.path.join(
            project_root, "data", "processed", "battery2_features.csv"
        )

    # Check Battery 2 availability
    if not os.path.exists(battery2_csv_path):
        # Check if raw files exist to inform the user
        b2_raw_dir = os.path.join(project_root, "data", "raw", "battery2")
        ref_path = os.path.join(
            project_root, "data", "raw", "reference", "VZ5_008_021_VUT_D_0043_v1.txt"
        )
        has_ref = os.path.exists(ref_path)

        err_msg = (
            f"\n"
            f"========================================================================\n"
            f"ERROR: BATTERY 2 DATASET UNAVAILABLE FOR CROSS-BATTERY EVALUATION\n"
            f"========================================================================\n"
            f"Processed Battery 2 dataset was not found at:\n"
            f"  {battery2_csv_path}\n\n"
            f"Battery 2 Status Diagnostic:\n"
            f"  - Ground-truth reference SOH: {'VERIFIED' if has_ref else 'MISSING'} ({ref_path})\n"
            f"  - Raw telemetry directory:    {b2_raw_dir} (missing or empty)\n\n"
            f"Required Actions to Enable Cross-Battery Evaluation:\n"
            f"  1. Download the Battery 2 raw cycling files from Zenodo Record 15657829:\n"
            f"     Samsung INR18650-35E Battery 2 cycling measurements (0032_v1 to 0038_v1).\n"
            f"  2. Place raw files into: {b2_raw_dir}\n"
            f"  3. Run: python -m src.data.prepare_battery2\n"
            f"  4. Re-run: python -m src.evaluation.evaluate_cross_battery\n\n"
            f"ACADEMIC INTEGRITY NOTE:\n"
            f"Cross-battery evaluation requires genuine cycling telemetry from an independent\n"
            f"physical cell. In accordance with capstone project standards, synthetic data\n"
            f"or fabricated measurements will not be generated.\n"
            f"========================================================================\n"
        )
        raise FileNotFoundError(err_msg)

    # 3. Load Battery 1 splits
    if verbose:
        print("========================================================================")
        print("STAGE 1: LOADING BATTERY 1 DATA & PRESERVING SCIENTIFIC ISOLATION")
        print("========================================================================")

    b1_splits = load_battery1_splits(battery1_csv_path)
    X_train_raw, y_train, df_train = b1_splits["train"]
    X_val_raw, y_val, df_val = b1_splits["val"]
    X_test_raw, y_test, df_test = b1_splits["test"]

    if verbose:
        print(f"  Battery 1 Training Samples:   {len(df_train)} (Cycles 1–371)")
        print(f"  Battery 1 Validation Samples: {len(df_val)} (Cycles 372–464)")
        print(f"  Battery 1 Test Samples:       {len(df_test)} (Cycles 465–580)")

    # 4. Fit Scaler strictly on Battery 1 training partition
    scaler = fit_training_scaler(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    if verbose:
        print("\n[OK] StandardScaler fit strictly on Battery 1 training split.")
        print(f"     Feature means: {np.round(scaler.mean_, 3)}")
        print(f"     Feature stds:  {np.round(scaler.scale_, 3)}")

    # 5. Train experimental First-Order Sugeno ANFIS
    if verbose:
        print("\n========================================================================")
        print("STAGE 2: TRAINING EXPERIMENTAL FIRST-ORDER SUGENO ANFIS (BATTERY 1)")
        print("========================================================================")

    model = train_experimental_first_order_anfis(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        n_rules=2,
        lambda_reg=5.0,
        lr=0.005,
        n_epochs=100,
        random_state=42,
    )

    # Evaluate internal validation performance
    val_pred = model.predict(X_val_scaled)
    val_metrics = compute_evaluation_metrics(y_val, val_pred)
    val_bias = compute_validation_bias(model, scaler, X_val_raw, y_val)

    if verbose:
        print(f"  Validation RMSE:      {val_metrics['rmse_percent']:.4f}% SOH")
        print(f"  Validation MAE:       {val_metrics['mae_percent']:.4f}% SOH")
        print(f"  Validation R²:        {val_metrics['r2_score']:.4f}")
        print(f"  Validation Mean Bias: {val_bias:+.4f}% SOH")
        print(f"  Trainable Parameters: {model.parameter_count['total_parameters']}")

    # 6. Load Battery 2 dataset
    if verbose:
        print("\n========================================================================")
        print("STAGE 3: LOADING UNSEEN BATTERY 2 DATASET & VALIDATING PARITY")
        print("========================================================================")

    df_b2 = pd.read_csv(battery2_csv_path)

    # Verify canonical feature schema
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df_b2.columns]
    if missing_cols:
        raise ValueError(
            f"Battery 2 dataset is missing required canonical feature columns: {missing_cols}"
        )
    if TARGET_COLUMN not in df_b2.columns:
        raise ValueError(
            f"Battery 2 dataset is missing target column '{TARGET_COLUMN}'."
        )

    # Enforce identical feature order
    X_b2_raw = df_b2[FEATURE_COLUMNS].values
    y_b2 = df_b2[TARGET_COLUMN].values
    cn_b2 = df_b2["CN"].values if "CN" in df_b2.columns else np.arange(1, len(df_b2) + 1)
    cn_train_b1 = df_train["CN"].values if "CN" in df_train.columns else np.arange(1, len(df_train) + 1)

    if verbose:
        print(f"  Battery 2 Samples Loaded: {len(df_b2)}")
        print(f"  Battery 2 SOH Range:      {y_b2.min():.4f}% to {y_b2.max():.4f}%")
        print(f"  Canonical Feature Order:  {FEATURE_COLUMNS}")

    # Scale Battery 2 using frozen Battery 1 scaler
    X_b2_scaled = scaler.transform(X_b2_raw)

    # 7. Model Predictions on Battery 2
    pred_fo_uncal = model.predict(X_b2_scaled)
    pred_fo_val_cal = pred_fo_uncal - val_bias  # Validation-calibrated prediction

    # 8. Compute Naive Baselines
    baselines = compute_naive_baselines(y_train, cn_train_b1, y_b2, cn_b2)

    # 9. Evaluate all models on Battery 2
    models_to_evaluate: List[Tuple[str, np.ndarray, int]] = [
        ("ANFIS (First-Order Sugeno, Uncalibrated)", pred_fo_uncal, model.parameter_count["total_parameters"]),
        ("ANFIS (First-Order Sugeno, Val-Calibrated)", pred_fo_val_cal, model.parameter_count["total_parameters"]),
        ("Train-Mean Baseline (Battery 1 Mean)", baselines["Train-Mean Baseline"], 1),
        ("Persistence Baseline (Final B1 Train Cycle)", baselines["Persistence Baseline (Final B1 Train Cycle)"], 1),
        ("Persistence Baseline (Initial B2 State)", baselines["Persistence Baseline (Initial B2 State)"], 1),
        ("Linear Trend Extrapolation Baseline", baselines["Linear Trend Extrapolation Baseline"], 2),
    ]

    comparison_records: List[Dict[str, Any]] = []
    for model_name, preds, n_params in models_to_evaluate:
        m = compute_evaluation_metrics(y_b2, preds)
        record = {
            "model": model_name,
            "parameters": n_params,
            "b2_rmse_percent": m["rmse_percent"],
            "b2_mae_percent": m["mae_percent"],
            "b2_r2_score": m["r2_score"],
            "b2_mean_bias_percent": m["mean_bias_percent"],
            "b2_max_error_percent": m["max_error_percent"],
        }
        comparison_records.append(record)

    df_comparison = pd.DataFrame(comparison_records)

    # Display results
    if verbose:
        print("\n========================================================================")
        print("STAGE 4: CROSS-BATTERY EVALUATION RESULTS (BATTERY 1 -> BATTERY 2)")
        print("========================================================================")
        print(f"{'Model Name':46s} | {'RMSE (%)':8s} | {'MAE (%)':8s} | {'R²':8s} | {'Bias (%)':8s} | {'MaxErr (%)':10s}")
        print("-" * 102)
        for _, r in df_comparison.iterrows():
            print(
                f"{r['model']:46s} | "
                f"{r['b2_rmse_percent']:8.4f} | "
                f"{r['b2_mae_percent']:8.4f} | "
                f"{r['b2_r2_score']:8.4f} | "
                f"{r['b2_mean_bias_percent']:+8.4f} | "
                f"{r['b2_max_error_percent']:10.4f}"
            )
        print("========================================================================")

    # 10. Save Artifacts Separately
    saved_paths: Dict[str, str] = {}
    if save_artifacts:
        metrics_dir = os.path.join(project_root, "results", "metrics")
        preds_dir = os.path.join(project_root, "results", "predictions")
        models_dir = os.path.join(project_root, "models")

        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(preds_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # 1. Comparison CSV
        comp_csv_path = os.path.join(metrics_dir, "cross_battery_comparison.csv")
        df_comparison.to_csv(comp_csv_path, index=False)
        saved_paths["comparison_csv"] = comp_csv_path

        # 2. Detailed JSON
        detailed_json = {
            "evaluation_type": "Cross-Battery Generalization (Battery 1 -> Battery 2)",
            "training_source": "Samsung INR18650-35E Battery 1 (Cycles 1-371)",
            "validation_source": "Samsung INR18650-35E Battery 1 (Cycles 372-464)",
            "test_target": "Samsung INR18650-35E Battery 2",
            "battery2_sample_count": len(df_b2),
            "battery1_validation_bias_percent": round(val_bias, 4),
            "models_comparison": comparison_records,
        }
        json_path = os.path.join(metrics_dir, "cross_battery_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detailed_json, f, indent=2)
        saved_paths["metrics_json"] = json_path

        # 3. Predictions CSV
        df_preds = pd.DataFrame({
            "CN": cn_b2,
            "actual_SOH": y_b2,
            "pred_ANFIS_First_Order_Uncalibrated": pred_fo_uncal,
            "pred_ANFIS_First_Order_ValCal": pred_fo_val_cal,
            "pred_Train_Mean": baselines["Train-Mean Baseline"],
            "pred_Persistence_B1_Last": baselines["Persistence Baseline (Final B1 Train Cycle)"],
            "pred_Persistence_B2_Initial": baselines["Persistence Baseline (Initial B2 State)"],
            "pred_Linear_Trend": baselines["Linear Trend Extrapolation Baseline"],
            "resid_ANFIS_Uncalibrated": pred_fo_uncal - y_b2,
            "resid_ANFIS_ValCal": pred_fo_val_cal - y_b2,
        })
        preds_csv_path = os.path.join(preds_dir, "cross_battery_predictions.csv")
        df_preds.to_csv(preds_csv_path, index=False)
        saved_paths["predictions_csv"] = preds_csv_path

        # 4. Experimental Model & Scaler checkpoints (Isolated from production)
        exp_model_path = os.path.join(models_dir, "anfis_first_order_cross_battery_exp.joblib")
        exp_scaler_path = os.path.join(models_dir, "scaler_cross_battery_exp.joblib")
        joblib.dump(model, exp_model_path)
        joblib.dump(scaler, exp_scaler_path)
        saved_paths["experimental_model"] = exp_model_path
        saved_paths["experimental_scaler"] = exp_scaler_path

        if verbose:
            print("\n[OK] Experimental artifacts saved to:")
            for k, p in saved_paths.items():
                print(f"  - {k}: {p}")

    return {
        "status": "success",
        "battery2_samples": len(df_b2),
        "validation_bias": val_bias,
        "comparison_dataframe": df_comparison,
        "comparison_records": comparison_records,
        "saved_paths": saved_paths,
    }


def main():
    """CLI entry point for cross-battery evaluation."""
    parser = argparse.ArgumentParser(
        description="Cross-Battery Generalization Pipeline (Battery 1 -> Battery 2)"
    )
    parser.add_argument(
        "--b2-csv",
        dest="b2_csv",
        type=str,
        default=None,
        help="Optional path to processed Battery 2 CSV (data/processed/battery2_features.csv)",
    )
    parser.add_argument(
        "--no-save",
        dest="no_save",
        action="store_true",
        help="Run evaluation without saving artifacts to disk",
    )

    args = parser.parse_args()

    try:
        run_cross_battery_evaluation(
            battery2_csv_path=args.b2_csv,
            save_artifacts=not args.no_save,
            verbose=True,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR]: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
