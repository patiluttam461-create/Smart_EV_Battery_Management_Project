"""Performance evaluation metrics for EV Battery SOH regression models.

Calculates:
- Root Mean Squared Error (RMSE) in percentage points (% SOH)
- Mean Absolute Error (MAE) in percentage points (% SOH)
- Coefficient of Determination (R²)
- Maximum Absolute Error (Max Error) in percentage points (% SOH)
"""

from typing import Dict, Union
import numpy as np
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    split_name: str = "",
) -> Dict[str, float]:
    """Compute regression performance metrics between actual and predicted SOH.

    Args:
        y_true: True SOH values (in percent).
        y_pred: Predicted SOH values (in percent).
        split_name: Optional split descriptor (e.g., 'Train', 'Val', 'Test').

    Returns:
        Dictionary of computed metrics with clear unit labels:
        - rmse_percent: RMSE in % SOH points
        - mae_percent: MAE in % SOH points
        - max_error_percent: Maximum error in % SOH points
        - r2_score: Coefficient of determination
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_true_arr.shape} != y_pred shape {y_pred_arr.shape}"
        )
    if len(y_true_arr) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")
    if np.any(np.isnan(y_true_arr)) or np.any(np.isnan(y_pred_arr)):
        raise ValueError("NaN values detected in inputs to metric calculation.")

    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    max_err = float(max_error(y_true_arr, y_pred_arr))

    # Guard R2 calculation if variance is zero
    y_var = float(np.var(y_true_arr))
    if y_var < 1e-12:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true_arr, y_pred_arr))

    prefix = f"{split_name}_" if split_name else ""
    return {
        f"{prefix}rmse_percent": round(rmse, 4),
        f"{prefix}mae_percent": round(mae, 4),
        f"{prefix}max_error_percent": round(max_err, 4),
        f"{prefix}r2_score": round(r2, 4),
    }


def format_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> str:
    """Format multiple model metrics into a markdown-compatible comparison table."""
    headers = ["Model", "Split", "RMSE (% SOH)", "MAE (% SOH)", "Max Error (%)", "R² Score"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for model_name, splits in metrics_dict.items():
        for split_label in ["train", "val", "test"]:
            if f"{split_label}_rmse_percent" in splits:
                rmse = splits[f"{split_label}_rmse_percent"]
                mae = splits[f"{split_label}_mae_percent"]
                m_err = splits[f"{split_label}_max_error_percent"]
                r2 = splits[f"{split_label}_r2_score"]
                lines.append(
                    f"| {model_name} | {split_label.upper()} | {rmse:.4f} | {mae:.4f} | {m_err:.4f} | {r2:.4f} |"
                )

    return "\n".join(lines)
