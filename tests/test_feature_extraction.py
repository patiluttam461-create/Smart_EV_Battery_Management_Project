"""Automated unit and regression tests for feature extraction and dataset outputs."""

import csv
import os
import unittest

from src.data.feature_extractor import (
    extract_all_cycle_features,
    extract_tI_ch,
    extract_temperature_features,
    extract_tv_ch,
    extract_tv_d,
    extract_voltage_features,
    linear_interpolate_crossing,
    segment_cycle_phases,
)
from src.data.prepare_dataset import (
    CYCLE_MAPPING,
    UNLABELED_RANGE,
    load_reference_soh,
)


class TestFeatureExtractor(unittest.TestCase):
    """Unit tests for feature extractor algorithms."""

    def test_linear_interpolation(self):
        """Test linear interpolation arithmetic."""
        t = linear_interpolate_crossing(t1=10.0, v1=3.0, t2=20.0, v2=4.0, target_v=3.5)
        self.assertAlmostEqual(t, 15.0)

        t_edge = linear_interpolate_crossing(t1=10.0, v1=3.2, t2=10.0, v2=3.2, target_v=3.2)
        self.assertAlmostEqual(t_edge, 10.0)

    def test_synthetic_cycle_extraction(self):
        """Test feature extraction on a clean synthetic CCCV cycle."""
        records = []
        t = 0.0

        # 1. Rest 10s: I = 0, V = 3.10 V
        for _ in range(2):
            records.append({"t": t, "qd": 0.0, "qc": 0.0, "v": 3.10, "i": 0.0, "temp": 25.0})
            t += 5.0

        # 2. CC Charge at 1700 mA from 3.10 V to 4.20 V in 100s
        for step in range(11):
            v = 3.10 + (4.20 - 3.10) * (step / 10.0)
            records.append({"t": t, "qd": 0.0, "qc": 100.0 * step, "v": v, "i": 1700.0, "temp": 25.0 + step * 0.5})
            t += 10.0

        # 3. CV Charge held at 4.20 V, current tapers from 1700 mA to 100 mA in 50s
        for step in range(1, 6):
            i = 1700.0 - (1600.0) * (step / 5.0)
            records.append({"t": t, "qd": 0.0, "qc": 1000.0 + 50.0 * step, "v": 4.20, "i": i, "temp": 30.0 - step * 0.2})
            t += 10.0

        # 4. Rest 20s
        for _ in range(2):
            records.append({"t": t, "qd": 0.0, "qc": 1250.0, "v": 4.19, "i": 0.0, "temp": 28.0})
            t += 10.0

        # 5. CC Discharge at -1700 mA from 4.14 V down to 2.65 V in 100s
        for step in range(11):
            v = 4.14 - (4.14 - 2.65) * (step / 10.0)
            records.append({"t": t, "qd": 100.0 * step, "qc": 1250.0, "v": v, "i": -1700.0, "temp": 28.0 + step * 0.3})
            t += 10.0

        feats = extract_all_cycle_features(records, interpolate=True)

        self.assertGreater(feats["tv_ch"], 0.0)
        self.assertGreater(feats["tv_d"], 0.0)
        self.assertGreater(feats["tI_ch"], 0.0)
        self.assertGreater(feats["Tavg"], 20.0)
        self.assertGreater(feats["tTpeak_ch"], 0.0)
        self.assertGreater(feats["Eavg_ch"], 3.2)
        self.assertGreater(feats["Eavg_d"], 2.6)

    def test_missing_phase_failure(self):
        """Test extractor throws meaningful error when charging phase is missing."""
        records = [
            {"t": 0.0, "qd": 100.0, "qc": 0.0, "v": 3.8, "i": -1700.0, "temp": 25.0},
            {"t": 10.0, "qd": 200.0, "qc": 0.0, "v": 3.5, "i": -1700.0, "temp": 26.0},
        ]
        with self.assertRaises(ValueError) as ctx:
            extract_all_cycle_features(records)
        self.assertIn("charging phase", str(ctx.exception))


class TestDatasetOutputs(unittest.TestCase):
    """Validation tests for generated CSV artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        cls.labeled_csv = os.path.join(
            cls.project_root, "data", "processed", "battery1_features.csv"
        )
        cls.unlabeled_csv = os.path.join(
            cls.project_root, "data", "processed", "unlabeled_cycles.csv"
        )

    def test_cycle_mapping_continuity(self):
        """Test continuity of the cycle mapping definition."""
        expected_cn = 1
        total_mapped = 0
        for fname, start_c, end_c, s_cn, e_cn in CYCLE_MAPPING:
            self.assertEqual(s_cn, expected_cn)
            cycle_span = (end_c - start_c + 1)
            cn_span = (e_cn - s_cn + 1)
            self.assertEqual(cycle_span, cn_span)
            total_mapped += cycle_span
            expected_cn = e_cn + 1
        self.assertEqual(total_mapped, 580)
        self.assertEqual(expected_cn, 581)

    def test_labeled_csv_properties(self):
        """Verify battery1_features.csv schema, completeness, and values."""
        self.assertTrue(os.path.exists(self.labeled_csv), "battery1_features.csv missing")

        with open(self.labeled_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 580, "Expected exactly 580 labeled rows")

        cns = []
        for r in rows:
            cn = int(r["CN"])
            cns.append(cn)

            # Check that all 7 features are positive finite floats
            for feat in ["tv_ch", "tv_d", "tI_ch", "Tavg", "tTpeak_ch", "Eavg_ch", "Eavg_d"]:
                val = float(r[feat])
                self.assertFalse(val is None)
                self.assertGreater(val, 0.0)

            # Check SOH percentage bounds (80% to 100%)
            soh = float(r["SOH"])
            self.assertGreaterEqual(soh, 80.0)
            self.assertLessEqual(soh, 100.0)

        self.assertEqual(len(set(cns)), 580, "Duplicate CN values detected")
        self.assertEqual(min(cns), 1)
        self.assertEqual(max(cns), 580)

    def test_unlabeled_csv_properties(self):
        """Verify unlabeled_cycles.csv contains 20 rows with unassigned SOH."""
        self.assertTrue(os.path.exists(self.unlabeled_csv), "unlabeled_cycles.csv missing")

        with open(self.unlabeled_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 20, "Expected exactly 20 unlabeled rows")
        for r in rows:
            self.assertEqual(r["CN"], "", "Unlabeled row must have empty CN")
            self.assertEqual(r["SOH"], "", "Unlabeled row must have empty SOH")
            for feat in ["tv_ch", "tv_d", "tI_ch", "Tavg", "tTpeak_ch", "Eavg_ch", "Eavg_d"]:
                val = float(r[feat])
                self.assertGreater(val, 0.0)


if __name__ == "__main__":
    unittest.main()
