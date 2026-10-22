"""Artifact data loader for EV Battery SOH Dashboard.

Loads:
- Benchmark comparison metrics (CSV/JSON)
- Historical cycle predictions (CSV)
- Saved graph image paths
"""

import json
import os
from typing import Any, Dict, List, Optional
import pandas as pd


class DashboardDataLoader:
    """Manages reading and parsing of project results and diagnostic artifacts."""

    def __init__(self, project_root: Optional[str] = None) -> None:
        if project_root is None:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
        self.project_root = project_root
        self.metrics_csv_path = os.path.join(
            project_root, "results", "metrics", "model_comparison_metrics.csv"
        )
        self.metrics_json_path = os.path.join(
            project_root, "results", "metrics", "detailed_metrics.json"
        )
        self.predictions_csv_path = os.path.join(
            project_root, "results", "predictions", "all_model_predictions.csv"
        )
        self.graphs_dir = os.path.join(project_root, "results", "graphs")

    def load_metrics(self) -> pd.DataFrame:
        """Load benchmark comparison metrics table."""
        if not os.path.exists(self.metrics_csv_path):
            raise FileNotFoundError(
                f"Metrics file missing: {self.metrics_csv_path}. Run evaluate_all first."
            )
        return pd.read_csv(self.metrics_csv_path)

    def load_detailed_metrics(self) -> Dict[str, Any]:
        """Load detailed metrics JSON."""
        if not os.path.exists(self.metrics_json_path):
            return {}
        with open(self.metrics_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_predictions(self) -> pd.DataFrame:
        """Load historical predictions table for all 580 cycles."""
        if not os.path.exists(self.predictions_csv_path):
            raise FileNotFoundError(
                f"Predictions file missing: {self.predictions_csv_path}."
            )
        return pd.read_csv(self.predictions_csv_path)

    def get_graph_paths(self) -> Dict[str, str]:
        """Return absolute paths for all generated diagnostic plots."""
        plots = {
            "actual_vs_predicted": "actual_vs_predicted_all_models.png",
            "residuals": "residuals_distribution_comparison.png",
            "metrics_bar": "model_metrics_comparison_bar.png",
            "membership_functions": "anfis_membership_functions.png",
        }
        found_paths = {}
        for key, fname in plots.items():
            full_path = os.path.join(self.graphs_dir, fname)
            if os.path.exists(full_path):
                found_paths[key] = full_path
            else:
                found_paths[key] = ""
        return found_paths
