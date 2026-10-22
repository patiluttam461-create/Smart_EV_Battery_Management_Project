"""Automated unit tests for First-Order Sugeno ANFIS model."""

import unittest
import numpy as np
from src.models.first_order_anfis import FirstOrderSugenoANFIS


class TestFirstOrderANFIS(unittest.TestCase):
    """Test suite for First-Order Sugeno ANFIS implementation."""

    def setUp(self):
        np.random.seed(42)
        # Synthetic feature matrix: 100 samples, 7 features
        self.X_train = np.random.randn(100, 7)
        # Synthetic linear target with slight noise
        true_weights = np.array([0.5, -0.3, 0.8, -0.1, 0.4, -0.6, 0.2])
        self.y_train = 90.0 + self.X_train @ true_weights + 0.05 * np.random.randn(100)

        self.X_val = np.random.randn(25, 7)
        self.y_val = 90.0 + self.X_val @ true_weights + 0.05 * np.random.randn(25)

        self.model = FirstOrderSugenoANFIS(
            n_rules=2,
            lr=0.01,
            lambda_reg=1.0,
            n_epochs=20,
            random_state=42,
        )

    def test_parameter_count_formula(self):
        """Verify parameter count matches 2*R*D premise + R*(D+1) consequent."""
        self.model.fit(self.X_train, self.y_train)
        R = self.model.n_rules
        D = self.model.n_features_in_
        self.assertEqual(D, 7)

        expected_premise = 2 * R * D  # 2 * 2 * 7 = 28
        expected_consequent = R * (D + 1)  # 2 * 8 = 16
        expected_total = expected_premise + expected_consequent  # 44

        p_info = self.model.parameter_count
        self.assertEqual(p_info["premise_parameters"], expected_premise)
        self.assertEqual(p_info["consequent_parameters"], expected_consequent)
        self.assertEqual(p_info["total_parameters"], expected_total)

    def test_output_shape_and_types(self):
        """Verify predict returns 1D float array with exact sample length."""
        self.model.fit(self.X_train, self.y_train)
        preds = self.model.predict(self.X_val)

        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.ndim, 1)
        self.assertEqual(len(preds), len(self.X_val))

    def test_predictions_are_finite(self):
        """Verify predictions contain no NaN or infinite values."""
        self.model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        preds = self.model.predict(self.X_val)

        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertFalse(np.any(np.isnan(preds)))

    def test_reproducibility(self):
        """Verify identical random state produces identical weights and predictions."""
        m1 = FirstOrderSugenoANFIS(n_rules=2, lr=0.01, lambda_reg=1.0, n_epochs=20, random_state=42)
        m1.fit(self.X_train, self.y_train)
        p1 = m1.predict(self.X_val)

        m2 = FirstOrderSugenoANFIS(n_rules=2, lr=0.01, lambda_reg=1.0, n_epochs=20, random_state=42)
        m2.fit(self.X_train, self.y_train)
        p2 = m2.predict(self.X_val)

        np.testing.assert_allclose(m1.centers_, m2.centers_, rtol=1e-6)
        np.testing.assert_allclose(m1.consequents_, m2.consequents_, rtol=1e-6)
        np.testing.assert_allclose(p1, p2, rtol=1e-6)

    def test_input_validation_and_dimensions(self):
        """Verify model handles 1D vector and rejects mismatched feature counts."""
        self.model.fit(self.X_train, self.y_train)

        # 1D single-sample vector of shape (7,) should be accepted
        single_sample = self.X_val[0]
        p_single = self.model.predict(single_sample)
        self.assertEqual(p_single.shape, (1,))
        self.assertTrue(np.isfinite(p_single[0]))

        # Mismatched dimension (e.g. 5 features instead of 7) should raise ValueError
        bad_dim = np.random.randn(10, 5)
        with self.assertRaises(ValueError):
            self.model.predict(bad_dim)

        # Calling predict before fit should raise ValueError
        unfitted = FirstOrderSugenoANFIS()
        with self.assertRaises(ValueError):
            unfitted.predict(self.X_val)

    def test_feature_order_sensitivity(self):
        """Verify that permuting feature order produces different predictions (order matters)."""
        self.model.fit(self.X_train, self.y_train)
        orig_preds = self.model.predict(self.X_val)

        # Permute columns (e.g., reverse order)
        reversed_X = self.X_val[:, ::-1]
        rev_preds = self.model.predict(reversed_X)

        # The predictions must differ because feature positions represent different physical variables
        diff = np.max(np.abs(orig_preds - rev_preds))
        self.assertGreater(diff, 1e-2, "Model must be sensitive to feature order.")

    def test_consequent_linearity(self):
        """Verify consequents shape is (R, D+1) containing linear slopes and intercept."""
        self.model.fit(self.X_train, self.y_train)
        R = self.model.n_rules
        D = self.model.n_features_in_
        self.assertEqual(self.model.consequents_.shape, (R, D + 1))

    def test_saved_first_order_model_artifact_loading_and_test_metrics(self):
        """Verify saved Joblib artifact reproduces verified test MAE, RMSE, and R2 on held-out cycles."""
        import os
        import joblib
        import pandas as pd
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_path = os.path.join(project_root, "models", "anfis_first_order_model.joblib")
        scaler_path = os.path.join(project_root, "models", "scaler.joblib")
        data_path = os.path.join(project_root, "data", "processed", "battery1_features.csv")

        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(scaler_path))
        self.assertTrue(os.path.exists(data_path))

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        df = pd.read_csv(data_path)

        # Test partition: rows 464 to 580 (116 cycles)
        features = ["tv_ch", "tv_d", "tI_ch", "Tavg", "tTpeak_ch", "Eavg_ch", "Eavg_d"]
        test_df = df.iloc[464:]
        X_test_scaled = scaler.transform(test_df[features].values)
        y_test = test_df["SOH"].values

        preds = model.predict(X_test_scaled)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(root_mean_squared_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        bias = float(np.mean(preds - y_test))
        max_err = float(np.max(np.abs(preds - y_test)))

        self.assertAlmostEqual(mae, 0.8246, places=3)
        self.assertAlmostEqual(rmse, 0.8574, places=3)
        self.assertAlmostEqual(r2, -1.4560, places=3)
        self.assertAlmostEqual(bias, 0.8246, places=3)
        self.assertAlmostEqual(max_err, 1.4240, places=3)


if __name__ == "__main__":
    unittest.main()
