"""Unit and integration test suite for Cross-Battery Generalization Pipeline.

Verifies:
1. Battery 2 reference SOH parsing and integrity from reference file 0043.
2. Canonical feature schema parity and strict 7-feature ordering.
3. Strict scaler fitting on Battery 1 training split only (zero leakage).
4. Complete isolation between Battery 1 and Battery 2.
5. Missing raw/processed Battery 2 data handling (descriptive FileNotFoundError).
6. Validation-only bias calibration (estimating bias strictly on B1 validation).
7. Naive baselines mathematical correctness (Train-Mean, Persistence, Linear Trend).
8. Regression metric computation accuracy and guards against non-finite inputs.
9. End-to-end execution of the cross-battery evaluation pipeline with mock data.
10. Preservation of production model checkpoints and scalers.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.prepare_battery2 import (
    FEATURE_BOUNDS,
    FEATURE_NAMES,
    REF_FILENAME,
    check_battery2_raw_availability,
    load_battery2_reference_soh,
    process_battery2_dataset,
)
from src.evaluation.evaluate_cross_battery import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    compute_evaluation_metrics,
    compute_naive_baselines,
    compute_validation_bias,
    fit_training_scaler,
    load_battery1_splits,
    run_cross_battery_evaluation,
    train_experimental_first_order_anfis,
)
from src.models.first_order_anfis import FirstOrderSugenoANFIS


class TestBattery2DataPreparation(unittest.TestCase):
    """Test suite for Battery 2 raw inspection and reference SOH parsing."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        cls.ref_path = os.path.join(
            cls.project_root, "data", "raw", "reference", REF_FILENAME
        )

    def test_reference_soh_loading_exact_properties(self):
        """Verify ground-truth SOH for Battery 2 loads exactly 580 valid cycles."""
        self.assertTrue(
            os.path.exists(self.ref_path),
            f"Reference file missing at: {self.ref_path}",
        )
        soh_dict = load_battery2_reference_soh(self.ref_path)

        # 1. Exactly 580 cycles
        self.assertEqual(len(soh_dict), 580)
        # 2. Cycle numbers 1 to 580
        self.assertEqual(min(soh_dict.keys()), 1)
        self.assertEqual(max(soh_dict.keys()), 580)
        # 3. Plausible SOH % range [80.0%, 100.0%]
        vals = list(soh_dict.values())
        min_soh = min(vals)
        max_soh = max(vals)
        self.assertAlmostEqual(min_soh, 82.1332, places=3)
        self.assertAlmostEqual(max_soh, 97.0550, places=3)
        self.assertTrue(all(np.isfinite(vals)))

    def test_missing_reference_file_error(self):
        """Verify FileNotFoundError is raised when reference file does not exist."""
        with self.assertRaises(FileNotFoundError):
            load_battery2_reference_soh("non_existent_reference_file.txt")

    def test_raw_availability_reporting(self):
        """Verify check_battery2_raw_availability returns False when directory is missing."""
        non_existent_dir = os.path.join(self.project_root, "data", "raw", "non_existent_battery")
        is_avail, files, msg = check_battery2_raw_availability(non_existent_dir)
        self.assertFalse(is_avail)
        self.assertEqual(files, [])
        self.assertIn("does not exist", msg)

    def test_process_battery2_missing_raw_fails_gracefully(self):
        """Verify process_battery2_dataset raises FileNotFoundError with diagnostic message."""
        non_existent_dir = os.path.join(self.project_root, "data", "raw", "battery2_test_empty")
        with self.assertRaises(FileNotFoundError) as ctx:
            process_battery2_dataset(
                raw_dir=non_existent_dir,
                ref_file=self.ref_path,
                output_path="tmp.csv",
                cycle_mapping=[],
            )
        self.assertIn("raw measurement files are unavailable", str(ctx.exception))


class TestCrossBatteryPipelineIntegrity(unittest.TestCase):
    """Test suite for feature ordering, scaler fitting, and battery isolation."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        cls.b1_path = os.path.join(
            cls.project_root, "data", "processed", "battery1_features.csv"
        )
        cls.splits = load_battery1_splits(cls.b1_path)

    def test_canonical_feature_ordering_enforced(self):
        """Verify identical 7 features and exact ordering across both modules."""
        self.assertEqual(FEATURE_COLUMNS, FEATURE_NAMES)
        expected = [
            "tv_ch",
            "tv_d",
            "tI_ch",
            "Tavg",
            "tTpeak_ch",
            "Eavg_ch",
            "Eavg_d",
        ]
        self.assertEqual(FEATURE_COLUMNS, expected)

    def test_scaler_fitted_strictly_on_b1_train(self):
        """Verify scaler mean matches Battery 1 training partition only."""
        X_train, y_train, df_train = self.splits["train"]
        X_val, y_val, df_val = self.splits["val"]
        X_test, y_test, df_test = self.splits["test"]

        scaler = fit_training_scaler(X_train)

        # Scaler mean must exactly match X_train mean
        expected_mean = np.mean(X_train, axis=0)
        expected_std = np.std(X_train, axis=0)
        np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-5)
        np.testing.assert_allclose(scaler.scale_, expected_std, rtol=1e-5)

        # Scaler mean must NOT match validation or test mean
        val_mean = np.mean(X_val, axis=0)
        test_mean = np.mean(X_test, axis=0)
        self.assertFalse(np.allclose(scaler.mean_, val_mean))
        self.assertFalse(np.allclose(scaler.mean_, test_mean))

    def test_validation_bias_calculation_isolation(self):
        """Verify validation bias is computed strictly on validation partition without test data."""
        X_train, y_train, _ = self.splits["train"]
        X_val, y_val, _ = self.splits["val"]

        scaler = fit_training_scaler(X_train)
        model = train_experimental_first_order_anfis(
            scaler.transform(X_train),
            y_train,
            scaler.transform(X_val),
            y_val,
            n_epochs=10,  # Fast test
            random_state=42,
        )

        bias = compute_validation_bias(model, scaler, X_val, y_val)
        self.assertTrue(np.isfinite(bias))

        # Manually compute validation bias to verify exact arithmetic
        val_pred = model.predict(scaler.transform(X_val))
        manual_bias = float(np.mean(val_pred - y_val))
        self.assertAlmostEqual(bias, manual_bias, places=6)

    def test_missing_battery2_raises_descriptive_error(self):
        """Verify run_cross_battery_evaluation raises FileNotFoundError when B2 is absent."""
        fake_b2_path = os.path.join(self.project_root, "data", "processed", "non_existent_b2.csv")
        with self.assertRaises(FileNotFoundError) as ctx:
            run_cross_battery_evaluation(
                project_root=self.project_root,
                battery2_csv_path=fake_b2_path,
                save_artifacts=False,
                verbose=False,
            )
        err_str = str(ctx.exception)
        self.assertIn("BATTERY 2 DATASET UNAVAILABLE", err_str)
        self.assertIn("Zenodo Record 15657829", err_str)
        self.assertIn("ACADEMIC INTEGRITY NOTE", err_str)


class TestCrossBatteryBaselinesAndMetrics(unittest.TestCase):
    """Test suite for naive operational baselines and evaluation metric formulas."""

    def test_naive_baselines_mathematical_properties(self):
        """Verify Train-Mean, Persistence (B1 and B2), and Linear Trend formulas."""
        y_train_b1 = np.array([98.0, 96.0, 94.0, 92.0])
        cn_train_b1 = np.array([1, 2, 3, 4])
        y_b2 = np.array([95.0, 93.0, 90.0])
        cn_b2 = np.array([1, 2, 3])

        baselines = compute_naive_baselines(y_train_b1, cn_train_b1, y_b2, cn_b2)

        # 1. Train-Mean
        expected_mean = 95.0
        np.testing.assert_allclose(baselines["Train-Mean Baseline"], [95.0, 95.0, 95.0])

        # 2. Persistence (Final B1 Cycle)
        np.testing.assert_allclose(
            baselines["Persistence Baseline (Final B1 Train Cycle)"], [92.0, 92.0, 92.0]
        )

        # 3. Persistence (Initial B2 State)
        np.testing.assert_allclose(
            baselines["Persistence Baseline (Initial B2 State)"], [95.0, 95.0, 95.0]
        )

        # 4. Linear Trend: slope is -2.0, intercept is 100.0 => predictions: 98, 96, 94
        np.testing.assert_allclose(
            baselines["Linear Trend Extrapolation Baseline"], [98.0, 96.0, 94.0]
        )

    def test_evaluation_metrics_exact_values(self):
        """Test compute_evaluation_metrics against analytical hand-calculated values."""
        y_true = np.array([90.0, 85.0, 80.0])
        # Errors: +1.0, -2.0, +3.0
        y_pred = np.array([91.0, 83.0, 83.0])

        m = compute_evaluation_metrics(y_true, y_pred)

        # MAE = (1 + 2 + 3) / 3 = 2.0
        self.assertAlmostEqual(m["mae_percent"], 2.0, places=4)
        # MSE = (1 + 4 + 9) / 3 = 14/3 = 4.6667 => RMSE = sqrt(4.6667) = 2.1602
        self.assertAlmostEqual(m["rmse_percent"], 2.1602, places=4)
        # Max error = 3.0
        self.assertAlmostEqual(m["max_error_percent"], 3.0, places=4)
        # Mean bias = (+1 - 2 + 3) / 3 = +2/3 = +0.6667
        self.assertAlmostEqual(m["mean_bias_percent"], 0.6667, places=4)

    def test_evaluation_metrics_guards(self):
        """Test error handling on empty arrays, mismatched lengths, and NaNs."""
        with self.assertRaises(ValueError):
            compute_evaluation_metrics(np.array([1.0, 2.0]), np.array([1.0]))
        with self.assertRaises(ValueError):
            compute_evaluation_metrics(np.array([]), np.array([]))
        with self.assertRaises(ValueError):
            compute_evaluation_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


class TestEndToEndCrossBatteryExecution(unittest.TestCase):
    """Test suite for complete end-to-end execution of the pipeline using mock data."""

    def setUp(self):
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.temp_dir = tempfile.mkdtemp()

        # Create synthetic Battery 2 test CSV matching schema
        n_samples = 30
        mock_data = {
            "CN": np.arange(1, n_samples + 1),
            "global_cycle": np.arange(n_samples),
            "source_file": ["mock_file.txt"] * n_samples,
            "source_cycle": np.arange(1, n_samples + 1),
            "tv_ch": np.linspace(4800.0, 4600.0, n_samples),
            "tv_d": np.linspace(3500.0, 3200.0, n_samples),
            "tI_ch": np.linspace(7200.0, 7000.0, n_samples),
            "Tavg": np.linspace(24.0, 25.0, n_samples),
            "tTpeak_ch": np.linspace(4500.0, 4700.0, n_samples),
            "Eavg_ch": np.linspace(4.01, 4.02, n_samples),
            "Eavg_d": np.linspace(3.52, 3.50, n_samples),
            "SOH": np.linspace(96.0, 88.0, n_samples),
        }
        self.mock_b2_df = pd.DataFrame(mock_data)
        self.mock_b2_path = os.path.join(self.temp_dir, "mock_battery2_features.csv")
        self.mock_b2_df.to_csv(self.mock_b2_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_evaluation_with_mock_data(self):
        """Verify run_cross_battery_evaluation executes completely with mock data."""
        # Snapshot production model file timestamps to ensure they are untouched
        prod_model_path = os.path.join(
            self.project_root, "models", "anfis_first_order_model.joblib"
        )
        prod_mtime_before = os.path.getmtime(prod_model_path) if os.path.exists(prod_model_path) else None

        results = run_cross_battery_evaluation(
            project_root=self.project_root,
            battery2_csv_path=self.mock_b2_path,
            save_artifacts=False,  # Do not pollute disk during unit test
            verbose=False,
        )

        self.assertEqual(results["status"], "success")
        self.assertEqual(results["battery2_samples"], 30)
        self.assertIn("comparison_dataframe", results)
        df_comp = results["comparison_dataframe"]

        # Ensure all 6 models/baselines are present
        self.assertEqual(len(df_comp), 6)
        models_found = set(df_comp["model"].values)
        self.assertIn("ANFIS (First-Order Sugeno, Uncalibrated)", models_found)
        self.assertIn("ANFIS (First-Order Sugeno, Val-Calibrated)", models_found)
        self.assertIn("Train-Mean Baseline (Battery 1 Mean)", models_found)
        self.assertIn("Persistence Baseline (Final B1 Train Cycle)", models_found)
        self.assertIn("Persistence Baseline (Initial B2 State)", models_found)
        self.assertIn("Linear Trend Extrapolation Baseline", models_found)

        # Ensure all metrics are finite numbers
        for col in ["b2_rmse_percent", "b2_mae_percent", "b2_r2_score", "b2_mean_bias_percent", "b2_max_error_percent"]:
            self.assertTrue(np.all(np.isfinite(df_comp[col].values)))

        # Verify production model artifact was NOT modified
        if prod_mtime_before is not None:
            prod_mtime_after = os.path.getmtime(prod_model_path)
            self.assertEqual(
                prod_mtime_before,
                prod_mtime_after,
                "CRITICAL: Production model checkpoint was modified during cross-battery evaluation!",
            )


if __name__ == "__main__":
    unittest.main()
