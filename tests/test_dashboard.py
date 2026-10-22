"""Automated unit and integration tests for dashboard components and inference engine."""

import os
import unittest
import numpy as np
import pandas as pd

from src.dashboard.data_loader import DashboardDataLoader
from src.dashboard.inference_engine import (
    FEATURE_BOUNDS,
    FEATURE_NAMES,
    PRESET_EXAMPLES,
    BatteryInferenceEngine,
)
from src.dashboard.app import build_app


class TestDashboardInferenceEngine(unittest.TestCase):
    """Test suite for dashboard input validation and inference engine."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        cls.models_dir = os.path.join(cls.project_root, "models")
        cls.engine = BatteryInferenceEngine(models_dir=cls.models_dir)

        cls.valid_sample = {
            "tv_ch": "5400.3957",
            "tv_d": "3604.0192",
            "tI_ch": "7283.0749",
            "Tavg": "30.7537",
            "tTpeak_ch": "5887.0143",
            "Eavg_ch": "4.095285",
            "Eavg_d": "3.578466",
        }

    def test_engine_artifacts_loaded(self):
        """Confirm scaler and models (including First-Order and Zero-Order) are loaded into memory."""
        self.assertIsNotNone(self.engine.scaler)
        self.assertIn("ANFIS", self.engine.models)
        self.assertIn("First_Order_ANFIS", self.engine.models)
        self.assertIn("Zero_Order_ANFIS", self.engine.models)
        self.assertIn("Random_Forest", self.engine.models)
        self.assertIn("MLP_ANN", self.engine.models)

    def test_valid_input_validation(self):
        """Confirm valid numeric inputs within bounds pass validation."""
        is_valid, arr, errors = self.engine.validate_inputs(self.valid_sample)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertIsNotNone(arr)
        self.assertEqual(arr.shape, (1, 7))

    def test_missing_input_rejection(self):
        """Confirm missing or blank input is rejected with clear error."""
        sample = self.valid_sample.copy()
        sample["tv_ch"] = ""
        is_valid, arr, errors = self.engine.validate_inputs(sample)
        self.assertFalse(is_valid)
        self.assertIn("tv_ch", errors[0])

    def test_non_numeric_input_rejection(self):
        """Confirm non-numeric string is rejected."""
        sample = self.valid_sample.copy()
        sample["Tavg"] = "twenty-five"
        is_valid, arr, errors = self.engine.validate_inputs(sample)
        self.assertFalse(is_valid)
        self.assertIn("Tavg", errors[0])

    def test_out_of_bounds_input_rejection(self):
        """Confirm physically implausible values trigger bounds rejection."""
        # Unphysical charging voltage 9.0 V
        sample = self.valid_sample.copy()
        sample["Eavg_ch"] = "9.50"
        is_valid, arr, errors = self.engine.validate_inputs(sample)
        self.assertFalse(is_valid)
        self.assertTrue(any("outside plausible range" in e for e in errors))

        # Unphysical temperature 120 °C
        sample2 = self.valid_sample.copy()
        sample2["Tavg"] = "120.0"
        is_valid2, arr2, errors2 = self.engine.validate_inputs(sample2)
        self.assertFalse(is_valid2)
        self.assertTrue(any("Tavg" in e for e in errors2))

    def test_forbidden_input_rejection(self):
        """Confirm forbidden leakage keys are rejected immediately."""
        sample = self.valid_sample.copy()
        sample["CN"] = 10
        sample["SOH"] = 90.0
        is_valid, arr, errors = self.engine.validate_inputs(sample)
        self.assertFalse(is_valid)
        self.assertTrue(any("Forbidden feature" in e for e in errors))

    def test_model_selection_and_predictions(self):
        """Confirm predictions and health classification for all models."""
        _, arr, _ = self.engine.validate_inputs(self.valid_sample)

        for model_name in ["ANFIS", "Random_Forest", "MLP_ANN"]:
            res = self.engine.predict(arr, model_name=model_name)
            self.assertIn("predicted_soh", res)
            self.assertIn("health_status", res)
            self.assertIn("disclaimer", res)
            self.assertTrue(res["is_estimate"])
            self.assertGreater(res["predicted_soh"], 80.0)
            self.assertLess(res["predicted_soh"], 100.0)

    def test_numerical_parity_with_predictions_csv(self):
        """Verify dashboard inference engine exactly reproduces saved predictions."""
        pred_csv = os.path.join(
            self.project_root, "results", "predictions", "all_model_predictions.csv"
        )
        fo_pred_csv = os.path.join(
            self.project_root, "results", "predictions", "first_order_predictions.csv"
        )
        feat_csv = os.path.join(
            self.project_root, "data", "processed", "battery1_features.csv"
        )

        pred_df = pd.read_csv(pred_csv)
        fo_pred_df = pd.read_csv(fo_pred_csv)
        feat_df = pd.read_csv(feat_csv)

        # Audit rows across train, val, and test splits
        test_rows = [0, 150, 371, 464, 579]
        for idx in test_rows:
            raw_dict = {feat: str(feat_df.loc[idx, feat]) for feat in FEATURE_NAMES}
            is_valid, arr, errors = self.engine.validate_inputs(raw_dict)
            self.assertTrue(is_valid, f"Validation failed for row {idx}: {errors}")

            # Check Zero-Order ANFIS, RF, MLP
            for model_key, pred_col in [
                ("Zero_Order_ANFIS", "pred_ANFIS"),
                ("Random_Forest", "pred_Random_Forest"),
                ("MLP_ANN", "pred_MLP_ANN"),
            ]:
                res = self.engine.predict(arr, model_name=model_key)
                pred_val = res["predicted_soh"]
                expected_val = pred_df.loc[idx, pred_col]
                diff = abs(pred_val - expected_val)
                self.assertLess(
                    diff,
                    1e-3,
                    f"Mismatch at row {idx} for model {model_key}: pred={pred_val}, expected={expected_val}",
                )

            # Check First-Order ANFIS (active default)
            res_fo = self.engine.predict(arr, model_name="ANFIS")
            fo_pred_val = res_fo["predicted_soh"]
            fo_expected_val = fo_pred_df.loc[idx, "pred_ANFIS_first_order"]
            fo_diff = abs(fo_pred_val - fo_expected_val)
            self.assertLess(
                fo_diff,
                1e-3,
                f"Mismatch at row {idx} for First-Order ANFIS: pred={fo_pred_val}, expected={fo_expected_val}",
            )

    def test_preset_examples_academic_integrity(self):
        """Verify preset examples match real recorded cycles with no synthetic data."""
        feat_csv = os.path.join(
            self.project_root, "data", "processed", "battery1_features.csv"
        )
        feat_df = pd.read_csv(feat_csv)

        expected_categories = [
            "Typical measured battery cycle",
            "Higher-SOH example",
            "Degraded-battery example",
            "Custom input mode",
            "Healthy Battery",
            "Moderately Degraded Battery",
            "More Degraded Battery",
        ]
        for cat in expected_categories:
            self.assertIn(cat, PRESET_EXAMPLES)

        # Check real measured presets
        for cat in [
            "Typical measured battery cycle",
            "Higher-SOH example",
            "Degraded-battery example",
            "Healthy Battery",
            "Moderately Degraded Battery",
            "More Degraded Battery",
        ]:
            preset = PRESET_EXAMPLES[cat]
            self.assertFalse(preset["is_synthetic"], f"{cat} must not be synthetic")
            cn = preset["CN"]
            self.assertIsNotNone(cn)

            # Match against ground-truth processed dataset
            matched_row = feat_df[feat_df["CN"] == cn].iloc[0]
            self.assertAlmostEqual(preset["reference_soh"], matched_row["SOH"], places=4)
            for feat in FEATURE_NAMES:
                self.assertAlmostEqual(
                    preset["features"][feat], matched_row[feat], places=4
                )

    def test_feature_ordering_enforced(self):
        """Verify exact 7-feature sequence is preserved regardless of input dict ordering."""
        expected_order = [
            "tv_ch",
            "tv_d",
            "tI_ch",
            "Tavg",
            "tTpeak_ch",
            "Eavg_ch",
            "Eavg_d",
        ]
        self.assertEqual(FEATURE_NAMES, expected_order)

        # Pass in reverse order
        reversed_dict = {feat: self.valid_sample[feat] for feat in reversed(expected_order)}
        is_valid, arr, errors = self.engine.validate_inputs(reversed_dict)
        self.assertTrue(is_valid)
        self.assertEqual(arr.shape, (1, 7))

        # Check that parsed array matches the exact canonical order
        for idx, feat in enumerate(expected_order):
            self.assertAlmostEqual(arr[0, idx], float(self.valid_sample[feat]))

    def test_anfis_inference_no_silent_fallback(self):
        """Verify that requesting ANFIS explicitly uses the First-Order ANFIS model."""
        _, arr, _ = self.engine.validate_inputs(self.valid_sample)
        res = self.engine.predict(arr, model_name="ANFIS")
        self.assertEqual(res["model_used"], "First-Order Sugeno ANFIS")
        self.assertIn("predicted_soh", res)
        self.assertTrue(res["is_estimate"])

        # Unknown model name must raise KeyError, never fallback silently
        with self.assertRaises(KeyError):
            self.engine.predict(arr, model_name="Unknown_Model")

    def test_applicability_domain_initialization(self):
        """Verify applicability domain distance model is calibrated and invertible."""
        self.assertIsNotNone(self.engine.ad_inv_cov)
        self.assertEqual(self.engine.ad_inv_cov.shape, (7, 7))
        self.assertIsNotNone(self.engine.ad_mean)
        self.assertEqual(self.engine.ad_mean.shape, (7,))
        self.assertGreater(self.engine.ad_cov_condition_number, 0.0)
        self.assertLess(self.engine.ad_threshold_in_domain, self.engine.ad_threshold_extrapolation)
        self.assertLess(self.engine.ad_threshold_extrapolation, self.engine.ad_threshold_outlier)
        # Check empirical training thresholds (~3.68, ~6.52, ~10.43)
        self.assertAlmostEqual(self.engine.ad_threshold_in_domain, 3.6838, places=2)
        self.assertAlmostEqual(self.engine.ad_threshold_extrapolation, 6.5223, places=2)

    def test_applicability_domain_in_domain_and_extrapolation(self):
        """Verify in-domain classification for mid-life cycle and extrapolation for held-out cycle."""
        # 1. Typical cycle (Cycle 313) should be IN_DOMAIN
        ex_typical = PRESET_EXAMPLES["Typical measured battery cycle"]
        raw_typical = {f: ex_typical["features"][f] for f in FEATURE_NAMES}
        is_val, arr_typ, _ = self.engine.validate_inputs(raw_typical)
        self.assertTrue(is_val)
        res_typ = self.engine.predict(arr_typ)
        self.assertEqual(res_typ["ad_status"], "IN_DOMAIN")
        self.assertFalse(res_typ["applicability_domain"]["is_extrapolating"])
        self.assertIsNone(res_typ["ad_warning"])
        self.assertLessEqual(res_typ["mahalanobis_distance"], self.engine.ad_threshold_in_domain)

        # 2. Degraded cycle (Cycle 580) should trigger EXTRAPOLATION
        ex_degraded = PRESET_EXAMPLES["Degraded-battery example"]
        raw_degraded = {f: ex_degraded["features"][f] for f in FEATURE_NAMES}
        is_val2, arr_deg, _ = self.engine.validate_inputs(raw_degraded)
        self.assertTrue(is_val2)
        res_deg = self.engine.predict(arr_deg)
        self.assertEqual(res_deg["ad_status"], "EXTRAPOLATION")
        self.assertTrue(res_deg["applicability_domain"]["is_extrapolating"])
        self.assertIsNotNone(res_deg["ad_warning"])
        self.assertIn("Extrapolation Warning", res_deg["ad_warning"])
        self.assertGreater(res_deg["mahalanobis_distance"], self.engine.ad_threshold_extrapolation)

    def test_applicability_domain_non_finite_inputs(self):
        """Verify non-finite inputs (NaN / Inf) are rejected and return INVALID status."""
        # Direct non-finite array to check_applicability_domain
        nan_arr = np.full((1, 7), np.nan)
        ad_nan = self.engine.check_applicability_domain(nan_arr)
        self.assertEqual(ad_nan["status"], "INVALID")
        self.assertTrue(np.isnan(ad_nan["distance"]))
        self.assertIn("non-finite", ad_nan["warning"])

        # String 'nan' in validate_inputs
        sample = self.valid_sample.copy()
        sample["tv_ch"] = "nan"
        is_val, _, errs = self.engine.validate_inputs(sample)
        self.assertFalse(is_val)
        self.assertTrue(any("infinite or NaN" in e for e in errs))

        # String 'inf' in validate_inputs
        sample_inf = self.valid_sample.copy()
        sample_inf["Tavg"] = "inf"
        is_val_inf, _, errs_inf = self.engine.validate_inputs(sample_inf)
        self.assertFalse(is_val_inf)
        self.assertTrue(any("infinite or NaN" in e for e in errs_inf))

    def test_raw_prediction_preservation_and_degradation(self):
        """Verify raw predicted SOH is unclipped and degradation percentage is correctly calculated."""
        _, arr, _ = self.engine.validate_inputs(self.valid_sample)
        res = self.engine.predict(arr, model_name="ANFIS")

        self.assertIn("raw_predicted_soh", res)
        self.assertIsInstance(res["raw_predicted_soh"], float)
        self.assertAlmostEqual(res["raw_predicted_soh"], res["predicted_soh"], places=3)

        # Check degradation calculation: 100.0 - raw_pred
        expected_degradation = round(100.0 - res["raw_predicted_soh"], 4)
        self.assertEqual(res["degradation_percent"], expected_degradation)
        self.assertIn("100.0% - Predicted SOH", res["degradation_formula"])

    def test_test_set_residuals_all_positive_and_bias_identity(self):
        """Confirm that all 116 held-out test residuals are strictly positive, proving Mean Bias == MAE."""
        pred_csv = os.path.join(
            self.project_root, "results", "predictions", "first_order_predictions.csv"
        )
        self.assertTrue(os.path.exists(pred_csv), f"Missing predictions CSV: {pred_csv}")

        df = pd.read_csv(pred_csv)
        test_df = df[df["split"] == "test"]
        self.assertEqual(len(test_df), 116)

        y_true = test_df["actual_SOH"].values
        y_pred = test_df["pred_ANFIS_first_order"].values
        residuals = y_pred - y_true

        # Mathematical property: every single residual is positive
        self.assertTrue(np.all(residuals > 0), "All test set residuals must be strictly positive")
        mean_bias = float(np.mean(residuals))
        mae = float(np.mean(np.abs(residuals)))

        # Therefore signed bias equals MAE to machine precision
        self.assertAlmostEqual(mean_bias, mae, places=6)
        self.assertAlmostEqual(mae, 0.824628, places=4)

    def test_operational_baselines_integrity(self):
        """Verify operational baselines CSV exists and clearly identifies deployable vs non-deployable models."""
        base_csv = os.path.join(
            self.project_root, "results", "metrics", "operational_baselines.csv"
        )
        self.assertTrue(os.path.exists(base_csv), f"Missing operational baselines CSV: {base_csv}")

        df = pd.read_csv(base_csv)
        self.assertEqual(len(df), 5)

        names = df["baseline_name"].tolist()
        self.assertIn("Train-Mean Baseline", names)
        self.assertIn("Persistence (Last Known Train SOH)", names)
        self.assertIn("Persistence (Last Known Val SOH)", names)
        self.assertIn("Linear Aging Trend on CN", names)
        self.assertIn("Oracle Test-Mean (Non-Deployable Reference)", names)

        # Oracle test mean must not be marked deployable
        oracle_row = df[df["baseline_name"].str.contains("Oracle")].iloc[0]
        self.assertFalse(bool(oracle_row["deployable"]))

        # Operational baselines must be marked deployable
        train_mean_row = df[df["baseline_name"] == "Train-Mean Baseline"].iloc[0]
        self.assertTrue(bool(train_mean_row["deployable"]))

    def test_expanding_window_evaluation_metrics(self):
        """Verify expanding-window cross-validation metrics are computed across 4 folds."""
        exp_csv = os.path.join(
            self.project_root, "results", "metrics", "expanding_window_evaluation.csv"
        )
        self.assertTrue(os.path.exists(exp_csv), f"Missing expanding window CSV: {exp_csv}")

        df = pd.read_csv(exp_csv)
        self.assertEqual(len(df), 4)

        # Every fold should exhibit MAE < 1.0%
        for _, row in df.iterrows():
            self.assertLess(row["test_mae_percent"], 1.0)

        # Fold 4 (late life) should achieve low MAE and positive R2
        fold4 = df.iloc[3]
        self.assertLess(fold4["test_mae_percent"], 0.3)
        self.assertGreater(fold4["test_r2_score"], 0.5)


class TestDashboardDataLoader(unittest.TestCase):
    """Test suite for data loading utilities."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        cls.loader = DashboardDataLoader(project_root=cls.project_root)

    def test_metrics_loading(self):
        """Confirm model_comparison_metrics.csv loads with required columns."""
        df = self.loader.load_metrics()
        self.assertGreaterEqual(len(df), 9)
        self.assertIn("model", df.columns)
        self.assertIn("rmse_percent", df.columns)
        self.assertIn("mae_percent", df.columns)

    def test_predictions_loading(self):
        """Confirm all_model_predictions.csv loads exactly 580 rows."""
        df = self.loader.load_predictions()
        self.assertEqual(len(df), 580)
        self.assertIn("actual_SOH", df.columns)
        self.assertIn("pred_ANFIS", df.columns)

    def test_graphs_presence(self):
        """Confirm generated diagnostic graphs exist on disk."""
        paths = self.loader.get_graph_paths()
        self.assertTrue(os.path.exists(paths["actual_vs_predicted"]))
        self.assertTrue(os.path.exists(paths["residuals"]))
        self.assertTrue(os.path.exists(paths["membership_functions"]))


class TestDashboardAppStructure(unittest.TestCase):
    """Test suite for professional dark-mode single-window application structure."""

    def setUp(self):
        import flet as ft
        self.ft = ft

        class MockPage:
            def __init__(self):
                self.title = ""
                self.theme_mode = None
                self.bgcolor = None
                self.padding = 0
                self.scroll = None
                self.horizontal_alignment = None
                self.controls = []

            def add(self, *c):
                self.controls.extend(c)

            def update(self):
                pass

        self.MockPage = MockPage

    def test_build_app_window_title_and_controls(self):
        """Verify build_app creates single window titled 'SMART EV BATTERY MANAGEMENT SYSTEM' in dark mode."""
        page = self.MockPage()
        build_app(page)
        self.assertEqual(page.title, "SMART EV BATTERY MANAGEMENT SYSTEM")
        self.assertEqual(page.theme_mode, self.ft.ThemeMode.DARK)
        self.assertEqual(page.bgcolor, "#0B1220")
        self.assertEqual(len(page.controls), 1)  # Single main layout container

        # Verify main layout sections
        main_column = page.controls[0].content
        self.assertEqual(len(main_column.controls), 4)  # Header, Top Row, Process Flow, Disclaimer

        top_row = main_column.controls[1]
        self.assertEqual(len(top_row.controls), 2)  # Left column (inputs) and Right column (predictions & info)

    def test_example_buttons_populate_inputs_only(self):
        """Verify the three preset buttons populate features without inserting predictions."""
        page = self.MockPage()
        build_app(page)
        ctrls = page.app_controls

        healthy_btn = ctrls["healthy_btn"]
        warning_btn = ctrls["warning_btn"]
        critical_btn = ctrls["critical_btn"]
        result_soh_text = ctrls["soh_text"]
        status_text = ctrls["status_text"]

        # 1. Click Healthy Example
        healthy_btn.on_click(None)
        self.assertEqual(result_soh_text.value, "-- %")
        self.assertIn("Healthy Example", status_text.value)
        self.assertIn("97.50%", status_text.value)
        self.assertAlmostEqual(float(ctrls["input_fields"]["tv_ch"].value), 5400.3957, places=2)

        # 2. Click Warning Example
        warning_btn.on_click(None)
        self.assertEqual(result_soh_text.value, "-- %")
        self.assertIn("Warning Example", status_text.value)
        self.assertIn("87.94%", status_text.value)
        self.assertAlmostEqual(float(ctrls["input_fields"]["tv_ch"].value), 4733.8045, places=2)

        # 3. Click Critical Example
        critical_btn.on_click(None)
        self.assertEqual(result_soh_text.value, "-- %")
        self.assertIn("Critical Example", status_text.value)
        self.assertIn("83.10%", status_text.value)
        self.assertAlmostEqual(float(ctrls["input_fields"]["tv_ch"].value), 4360.1016, places=2)

    def test_predict_and_reset_button_functionality(self):
        """Verify PREDICT SOH runs ANFIS inference and CLEAR clears all inputs and resets gauge."""
        page = self.MockPage()
        build_app(page)
        ctrls = page.app_controls

        healthy_btn = ctrls["healthy_btn"]
        predict_btn = ctrls["predict_btn"]
        clear_btn = ctrls["clear_btn"]
        result_soh_text = ctrls["soh_text"]
        result_badge = ctrls["health_badge"]
        progress_bar = ctrls["progress_bar"]
        status_text = ctrls["status_text"]

        # Load healthy battery
        healthy_btn.on_click(None)
        self.assertEqual(result_soh_text.value, "-- %")
        self.assertEqual(progress_bar.value, 0.0)

        # Run prediction
        predict_btn.on_click(None)
        self.assertNotEqual(result_soh_text.value, "-- %")
        self.assertNotEqual(result_soh_text.value, "Error")
        self.assertIn("%", result_soh_text.value)
        pred_val = float(result_soh_text.value.replace("%", "").strip())
        self.assertGreater(pred_val, 90.0)
        self.assertEqual(result_badge.content.value, "HEALTHY")
        self.assertGreater(progress_bar.value, 0.90)

        # Now test CLEAR (Reset)
        clear_btn.on_click(None)
        self.assertEqual(result_soh_text.value, "-- %")
        self.assertEqual(progress_bar.value, 0.0)
        self.assertEqual(result_badge.content.value.upper(), "AWAITING INPUT")
        self.assertIn("Form reset", status_text.value)
        for feat in FEATURE_NAMES:
            self.assertEqual(ctrls["input_fields"][feat].value, "")

    def test_model_evaluation_metrics_section(self):
        """Verify the model information card displays verified First-Order ANFIS test MAE, RMSE, and R2 values."""
        page = self.MockPage()
        build_app(page)
        model_card = page.app_controls["model_info_card"]

        # Extract metric cards row inside model_info_card (controls[2].content is the benchmark metrics row)
        benchmark_row = model_card.content.controls[2].content
        metric_cards = benchmark_row.controls
        mae_text = metric_cards[0].content.controls[1].value
        rmse_text = metric_cards[1].content.controls[1].value
        r2_text = metric_cards[2].content.controls[1].value

        self.assertEqual(mae_text, "0.8246 %")
        self.assertEqual(rmse_text, "0.8574 %")
        self.assertEqual(r2_text, "-1.4560")

        # Verify specifications mention 2 Rules and First-Order Sugeno ANFIS
        specs_row = model_card.content.controls[1]
        all_spec_texts = [t.value for col in specs_row.controls for t in col.controls]
        self.assertTrue(any("2 Rules" in t for t in all_spec_texts))
        self.assertTrue(any("First-Order" in t for t in all_spec_texts))

    def test_gui_degradation_and_ad_pill_rendering(self):
        """Verify UI updates SOH, degradation percentage, and AD status badge upon prediction."""
        page = self.MockPage()
        build_app(page)
        ctrls = page.app_controls

        warning_btn = ctrls["warning_btn"]
        critical_btn = ctrls["critical_btn"]
        predict_btn = ctrls["predict_btn"]
        result_soh = ctrls["soh_text"]
        degradation_text = ctrls["degradation_text"]
        ad_badge = ctrls["ad_badge"]
        warning_banner = ctrls["warning_banner"]

        # 1. Warning Example (Cycle 313, In-Domain)
        warning_btn.on_click(None)
        predict_btn.on_click(None)
        self.assertNotEqual(result_soh.value, "-- %")
        self.assertIn("Degradation:", degradation_text.value)
        self.assertIn("In-Domain", ad_badge.content.value)
        self.assertFalse(warning_banner.visible)

        # 2. Critical Example (Cycle 580, Extrapolation)
        critical_btn.on_click(None)
        predict_btn.on_click(None)
        self.assertIn("Extrapolation", ad_badge.content.value)
        self.assertTrue(warning_banner.visible)
        self.assertIn("Extrapolation Warning", warning_banner.content.controls[1].value)


if __name__ == "__main__":
    unittest.main()
