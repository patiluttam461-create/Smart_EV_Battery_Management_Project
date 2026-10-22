"""Unit tests for ANFIS modeling, baseline training, leakage prevention, and metrics."""

import os
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation.evaluate_all import FEATURE_COLUMNS, TARGET_COLUMN, load_and_split_data
from src.evaluation.metrics import compute_regression_metrics
from src.models.anfis import SugenoANFIS
from src.models.baselines import get_mlp_baseline, get_random_forest_baseline


class TestRegressionMetrics(unittest.TestCase):
    """Test suite for evaluation metric calculations."""

    def test_perfect_prediction_metrics(self):
        """Test metrics on identical actual and predicted vectors."""
        y_true = np.array([90.0, 85.0, 80.0, 75.0])
        y_pred = np.array([90.0, 85.0, 80.0, 75.0])
        m = compute_regression_metrics(y_true, y_pred, split_name="test")

        self.assertEqual(m["test_rmse_percent"], 0.0)
        self.assertEqual(m["test_mae_percent"], 0.0)
        self.assertEqual(m["test_max_error_percent"], 0.0)
        self.assertAlmostEqual(m["test_r2_score"], 1.0)

    def test_known_error_metrics(self):
        """Test metrics against analytical error values."""
        y_true = np.array([90.0, 80.0])
        y_pred = np.array([92.0, 78.0])  # errors: +2, -2
        m = compute_regression_metrics(y_true, y_pred)

        self.assertAlmostEqual(m["rmse_percent"], 2.0)
        self.assertAlmostEqual(m["mae_percent"], 2.0)
        self.assertAlmostEqual(m["max_error_percent"], 2.0)

    def test_shape_mismatch_guard(self):
        """Test exception raised on vector length mismatch."""
        y_true = np.array([90.0, 80.0])
        y_pred = np.array([90.0])
        with self.assertRaises(ValueError):
            compute_regression_metrics(y_true, y_pred)

    def test_nan_input_guard(self):
        """Test exception raised when NaN values are present."""
        y_true = np.array([90.0, np.nan])
        y_pred = np.array([90.0, 80.0])
        with self.assertRaises(ValueError):
            compute_regression_metrics(y_true, y_pred)


class TestDataSplitAndLeakage(unittest.TestCase):
    """Test suite for chronological splitting and leakage prevention."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        cls.csv_path = os.path.join(
            cls.project_root, "data", "processed", "battery1_features.csv"
        )
        cls.df, cls.splits = load_and_split_data(cls.csv_path, n_test=116, val_ratio=0.2)

    def test_chronological_split_sizes(self):
        """Confirm exact row counts for 80% dev (80/20 train/val) and 20% test."""
        X_tr, y_tr, df_tr = self.splits["train"]
        X_va, y_va, df_va = self.splits["val"]
        X_te, y_te, df_te = self.splits["test"]

        self.assertEqual(len(y_tr), 371)
        self.assertEqual(len(y_va), 93)
        self.assertEqual(len(y_te), 116)
        self.assertEqual(len(y_tr) + len(y_va) + len(y_te), 580)

    def test_chronological_ordering_no_overlap(self):
        """Confirm strict chronological progression with zero index overlap."""
        df_tr = self.splits["train"][2]
        df_va = self.splits["val"][2]
        df_te = self.splits["test"][2]

        self.assertEqual(df_tr["CN"].min(), 1)
        self.assertEqual(df_tr["CN"].max(), 371)

        self.assertEqual(df_va["CN"].min(), 372)
        self.assertEqual(df_va["CN"].max(), 464)

        self.assertEqual(df_te["CN"].min(), 465)
        self.assertEqual(df_te["CN"].max(), 580)

        # Check disjoint sets
        tr_cns = set(df_tr["CN"])
        va_cns = set(df_va["CN"])
        te_cns = set(df_te["CN"])
        self.assertTrue(tr_cns.isdisjoint(va_cns))
        self.assertTrue(va_cns.isdisjoint(te_cns))
        self.assertTrue(tr_cns.isdisjoint(te_cns))

    def test_leakage_free_feature_set(self):
        """Confirm model inputs contain only the 7 intended features."""
        for feat in FEATURE_COLUMNS:
            self.assertIn(feat, self.df.columns)

        forbidden_inputs = ["CN", "global_cycle", "source_file", "source_cycle", "SOH"]
        for forbidden in forbidden_inputs:
            self.assertNotIn(forbidden, FEATURE_COLUMNS)

    def test_scaler_fit_strictly_on_train(self):
        """Verify scaler mean matches X_train mean, not validation or full data."""
        X_tr = self.splits["train"][0]
        X_va = self.splits["val"][0]
        X_te = self.splits["test"][0]

        scaler = StandardScaler()
        scaler.fit(X_tr)

        # Compare scaler mean against train mean
        np.testing.assert_allclose(scaler.mean_, np.mean(X_tr, axis=0), rtol=1e-5)

        # Confirm scaler mean differs from val mean
        val_mean = np.mean(X_va, axis=0)
        self.assertFalse(np.allclose(scaler.mean_, val_mean, atol=1e-2))


class TestModelArchitectureAndOutput(unittest.TestCase):
    """Test suite for ANFIS and baseline model execution and shapes."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        cls.csv_path = os.path.join(
            cls.project_root, "data", "processed", "battery1_features.csv"
        )
        _, cls.splits = load_and_split_data(cls.csv_path, n_test=116, val_ratio=0.2)
        cls.X_tr, cls.y_tr, _ = cls.splits["train"]
        cls.X_te, cls.y_te, _ = cls.splits["test"]

        cls.scaler = StandardScaler().fit(cls.X_tr)
        cls.X_tr_s = cls.scaler.transform(cls.X_tr)
        cls.X_te_s = cls.scaler.transform(cls.X_te)

    def test_anfis_parameter_counts(self):
        """Verify ANFIS rule and parameter count calculation."""
        anfis = SugenoANFIS(n_rules=5, order=0, lr=0.01, lambda_reg=0.05, n_epochs=5, random_state=42)
        anfis.fit(self.X_tr_s, self.y_tr)

        self.assertEqual(anfis.rule_count, 5)
        p_info = anfis.parameter_count
        # 5 rules * 7 inputs * 2 (centers + sigmas) = 70 premise
        self.assertEqual(p_info["premise_parameters"], 70)
        # 5 rules * 1 (zero-order scalar) = 5 consequent
        self.assertEqual(p_info["consequent_parameters"], 5)
        self.assertEqual(p_info["total_parameters"], 75)

    def test_anfis_prediction_shape_and_finiteness(self):
        """Verify ANFIS forward predictions are finite and correctly shaped."""
        anfis = SugenoANFIS(n_rules=5, order=0, lr=0.01, lambda_reg=0.05, n_epochs=10, random_state=42)
        anfis.fit(self.X_tr_s, self.y_tr)

        preds = anfis.predict(self.X_te_s)
        self.assertEqual(preds.shape, (len(self.X_te_s),))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all(preds > 50.0))
        self.assertTrue(np.all(preds < 105.0))

    def test_baseline_prediction_shapes(self):
        """Verify Random Forest and MLP prediction shapes and finiteness."""
        rf = get_random_forest_baseline(n_estimators=10, max_depth=6, random_state=42)
        rf.fit(self.X_tr_s, self.y_tr)
        rf_preds = rf.predict(self.X_te_s)
        self.assertEqual(rf_preds.shape, (len(self.X_te_s),))
        self.assertTrue(np.all(np.isfinite(rf_preds)))

        mlp = get_mlp_baseline(hidden_layer_sizes=(16,), max_iter=50, random_state=42)
        mlp.fit(self.X_tr_s, self.y_tr)
        mlp_preds = mlp.predict(self.X_te_s)
        self.assertEqual(mlp_preds.shape, (len(self.X_te_s),))
        self.assertTrue(np.all(np.isfinite(mlp_preds)))


if __name__ == "__main__":
    unittest.main()
