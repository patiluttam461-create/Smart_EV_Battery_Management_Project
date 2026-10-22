"""Dataset preparation script for Smart EV Battery Management ANFIS.

Extracts the 7 health indicator features for the 580 labeled cycles and
the 20 unlabeled cycles, producing:
- data/processed/battery1_features.csv (580 labeled cycles with SOH in percent)
- data/processed/unlabeled_cycles.csv (20 unlabeled cycles from file 0031)
"""

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

from src.data.feature_extractor import extract_all_cycle_features

# Approved mapping between raw measurement files and reference SOH CN
CYCLE_MAPPING: List[Tuple[str, int, int, int, int]] = [
    ("VZ5_008_021_VUT_D_0025_v1.txt", 0, 99, 1, 100),
    ("VZ5_008_021_VUT_D_0026_v1.txt", 0, 29, 101, 130),
    ("VZ5_008_021_VUT_D_0027_v1.txt", 0, 69, 131, 200),
    ("VZ5_008_021_VUT_D_0028_v1.txt", 0, 99, 201, 300),
    ("VZ5_008_021_VUT_D_0029_v1.txt", 0, 99, 301, 400),
    ("VZ5_008_021_VUT_D_0030_v1.txt", 0, 99, 401, 500),
    ("VZ5_008_021_VUT_D_0031_v1.txt", 0, 79, 501, 580),
]

UNLABELED_RANGE: Tuple[str, int, int] = ("VZ5_008_021_VUT_D_0031_v1.txt", 80, 99)

RAW_COLUMNS = [
    "cycle number",
    "time/s",
    "Q discharge/mA.h",
    "Q charge/mA.h",
    "Ecell/V",
    "<I>/mA",
    "Temperature/°C",
]


def load_reference_soh(ref_filepath: str) -> Dict[int, float]:
    """Load ground-truth SOH for Battery 1 from reference file.

    Args:
        ref_filepath: Path to VZ5_008_021_VUT_D_0043_v1.txt.

    Returns:
        Dict mapping cycle number (CN) -> SOH fraction.
    """
    if not os.path.exists(ref_filepath):
        raise FileNotFoundError(f"Reference SOH file not found at: {ref_filepath}")

    soh_dict: Dict[int, float] = {}
    with open(ref_filepath, "r", encoding="latin-1") as f:
        header = f.readline().strip("\r\n").split("\t")
        if len(header) < 2 or header[0] != "CN" or "Battery1" not in header[1]:
            raise ValueError(f"Unexpected header in reference file: {header}")

        for line_num, line in enumerate(f, start=2):
            parts = line.strip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            try:
                cn = int(parts[0])
                soh = float(parts[1])
                soh_dict[cn] = soh
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Error parsing reference line {line_num}: {line.strip()} - {e}"
                )

    if len(soh_dict) != 580:
        raise ValueError(
            f"Expected exactly 580 reference SOH entries, but found {len(soh_dict)}"
        )

    return soh_dict


def read_raw_cycles(filepath: str) -> Dict[int, List[Dict[str, float]]]:
    """Read all cycles from a raw battery measurement file.

    Args:
        filepath: Path to raw txt file.

    Returns:
        Dict mapping integer cycle index -> list of record dictionaries.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw measurement file not found at: {filepath}")

    cycles_dict: Dict[int, List[Dict[str, float]]] = {}

    with open(filepath, "r", encoding="latin-1") as f:
        header_line = f.readline().strip("\r\n").split("\t")
        if header_line and header_line[-1] == "":
            header_line = header_line[:-1]

        for line_num, line in enumerate(f, start=2):
            parts = line.strip("\r\n").split("\t")
            if not parts or parts[0] == "":
                continue

            try:
                cycle_num = int(float(parts[0]))
                record = {
                    "t": float(parts[1]),
                    "qd": float(parts[2]),
                    "qc": float(parts[3]),
                    "v": float(parts[4]),
                    "i": float(parts[5]),
                    "temp": float(parts[6]),
                }
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Corrupt telemetry at {filepath}:{line_num}: {line.strip()} - {e}"
                )

            if cycle_num not in cycles_dict:
                cycles_dict[cycle_num] = []
            cycles_dict[cycle_num].append(record)

    return cycles_dict


def process_dataset(
    raw_dir: str,
    ref_file: str,
    output_labeled: str,
    output_unlabeled: str,
    interpolate: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """Execute feature extraction and dataset preparation.

    Args:
        raw_dir: Directory containing battery1 raw files.
        ref_file: Path to reference SOH file.
        output_labeled: Target path for battery1_features.csv.
        output_unlabeled: Target path for unlabeled_cycles.csv.
        interpolate: If True, uses linear interpolation for threshold crossings.

    Returns:
        Tuple of (labeled_rows, unlabeled_rows).
    """
    soh_dict = load_reference_soh(ref_file)
    labeled_rows: List[Dict] = []
    unlabeled_rows: List[Dict] = []
    global_cycle_idx = 0

    print("=" * 70)
    print("STEP 3: FEATURE EXTRACTION PIPELINE")
    print(f"Source raw directory : {raw_dir}")
    print(f"Reference SOH file   : {ref_file}")
    print(f"Output labeled file  : {output_labeled}")
    print(f"Output unlabeled file: {output_unlabeled}")
    print("=" * 70)

    # 1. Process Labeled Cycles (580 cycles)
    for fname, start_c, end_c, expected_start_cn, expected_end_cn in CYCLE_MAPPING:
        fpath = os.path.join(raw_dir, fname)
        print(
            f"Parsing {fname}: cycles {start_c}..{end_c} -> CN {expected_start_cn}..{expected_end_cn}..."
        )
        file_cycles = read_raw_cycles(fpath)

        for c in range(start_c, end_c + 1):
            if c not in file_cycles:
                raise KeyError(f"Cycle {c} missing in raw file {fname}")

            pts = file_cycles[c]
            try:
                feats = extract_all_cycle_features(pts, interpolate=interpolate)
            except Exception as e:
                raise RuntimeError(
                    f"Extraction failed at {fname} cycle {c} (global {global_cycle_idx}): {e}"
                ) from e

            expected_cn = expected_start_cn + (c - start_c)
            if expected_cn not in soh_dict:
                raise KeyError(
                    f"Reference SOH missing for expected CN {expected_cn} ({fname} c={c})"
                )

            # Express SOH as a percentage, not a fraction
            soh_percent = round(soh_dict[expected_cn] * 100.0, 6)

            labeled_rows.append(
                {
                    "CN": expected_cn,
                    "global_cycle": global_cycle_idx,
                    "source_file": fname,
                    "source_cycle": c,
                    "tv_ch": round(feats["tv_ch"], 4),
                    "tv_d": round(feats["tv_d"], 4),
                    "tI_ch": round(feats["tI_ch"], 4),
                    "Tavg": round(feats["Tavg"], 4),
                    "tTpeak_ch": round(feats["tTpeak_ch"], 4),
                    "Eavg_ch": round(feats["Eavg_ch"], 6),
                    "Eavg_d": round(feats["Eavg_d"], 6),
                    "SOH": soh_percent,
                }
            )
            global_cycle_idx += 1

    # 2. Process Unlabeled Cycles (20 cycles from 0031)
    unlab_fname, unlab_start, unlab_end = UNLABELED_RANGE
    unlab_fpath = os.path.join(raw_dir, unlab_fname)
    print(
        f"Parsing unlabeled cycles from {unlab_fname}: cycles {unlab_start}..{unlab_end}..."
    )
    unlab_cycles = read_raw_cycles(unlab_fpath)

    for c in range(unlab_start, unlab_end + 1):
        if c not in unlab_cycles:
            raise KeyError(f"Cycle {c} missing in raw file {unlab_fname}")

        pts = unlab_cycles[c]
        try:
            feats = extract_all_cycle_features(pts, interpolate=interpolate)
        except Exception as e:
            raise RuntimeError(
                f"Extraction failed at {unlab_fname} unlabeled cycle {c}: {e}"
            ) from e

        unlabeled_rows.append(
            {
                "CN": "",  # Unlabeled: no official CN assigned
                "global_cycle": global_cycle_idx,
                "source_file": unlab_fname,
                "source_cycle": c,
                "tv_ch": round(feats["tv_ch"], 4),
                "tv_d": round(feats["tv_d"], 4),
                "tI_ch": round(feats["tI_ch"], 4),
                "Tavg": round(feats["Tavg"], 4),
                "tTpeak_ch": round(feats["tTpeak_ch"], 4),
                "Eavg_ch": round(feats["Eavg_ch"], 6),
                "Eavg_d": round(feats["Eavg_d"], 6),
                "SOH": "",  # Unlabeled: no ground truth SOH
            }
        )
        global_cycle_idx += 1

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_labeled), exist_ok=True)
    os.makedirs(os.path.dirname(output_unlabeled), exist_ok=True)

    fieldnames = [
        "CN",
        "global_cycle",
        "source_file",
        "source_cycle",
        "tv_ch",
        "tv_d",
        "tI_ch",
        "Tavg",
        "tTpeak_ch",
        "Eavg_ch",
        "Eavg_d",
        "SOH",
    ]

    with open(output_labeled, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled_rows)
    print(f"Saved {len(labeled_rows)} labeled rows to: {output_labeled}")

    with open(output_unlabeled, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unlabeled_rows)
    print(f"Saved {len(unlabeled_rows)} unlabeled rows to: {output_unlabeled}")

    return labeled_rows, unlabeled_rows


def validate_extracted_dataset(filepath: str) -> None:
    """Validate data integrity and constraints on the processed labeled dataset."""
    print("\n--- RUNNING VALIDATION ON PROCESSED LABELED DATASET ---")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed file not found: {filepath}")

    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # 1. Row count validation
    if len(rows) != 580:
        raise AssertionError(f"Expected exactly 580 rows, found {len(rows)}")
    print("[OK] Row count check passed: exactly 580 rows.")

    # 2. Unique CN check
    cns = [int(r["CN"]) for r in rows]
    if len(set(cns)) != 580:
        raise AssertionError("Duplicate CN values detected!")
    if min(cns) != 1 or max(cns) != 580:
        raise AssertionError(f"CN range is [{min(cns)}, {max(cns)}], expected [1, 580]")
    print("[OK] CN range check passed: exactly 1 to 580 with 0 duplicates.")

    # 3. Unique global_cycle check
    global_ids = [int(r["global_cycle"]) for r in rows]
    if len(set(global_ids)) != 580 or min(global_ids) != 0 or max(global_ids) != 579:
        raise AssertionError("global_cycle identifier check failed!")
    print("[OK] Global cycle ID check passed: exactly 0 to 579.")

    # 4. Feature completeness & finite range check
    features = ["tv_ch", "tv_d", "tI_ch", "Tavg", "tTpeak_ch", "Eavg_ch", "Eavg_d"]
    for feat in features:
        vals = [float(r[feat]) for r in rows]
        if any(v is None for v in vals):
            raise AssertionError(f"Null value found in feature {feat}")
        print(
            f"[OK] Feature {feat:10s} : range [{min(vals):9.4f} .. {max(vals):9.4f}] (mean={sum(vals)/len(vals):9.4f})"
        )

    # 5. SOH percentage check
    soh_vals = [float(r["SOH"]) for r in rows]
    if min(soh_vals) < 0.0 or max(soh_vals) > 100.0:
        raise AssertionError(f"SOH values out of percentage range [0, 100]: {min(soh_vals)}..{max(soh_vals)}")
    if min(soh_vals) < 50.0 or max(soh_vals) > 100.0:
        raise AssertionError(f"SOH percentage bounds suspicious for healthy battery: {min(soh_vals)}..{max(soh_vals)}")
    print(
        f"[OK] SOH target check    : range [{min(soh_vals):.4f}% .. {max(soh_vals):.4f}%] (expressed as percentage)"
    )
    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    raw_battery1_dir = os.path.join(project_root, "data", "raw", "battery1")
    reference_soh_path = os.path.join(
        project_root, "data", "raw", "reference", "VZ5_008_021_VUT_D_0043_v1.txt"
    )
    processed_labeled_path = os.path.join(
        project_root, "data", "processed", "battery1_features.csv"
    )
    processed_unlabeled_path = os.path.join(
        project_root, "data", "processed", "unlabeled_cycles.csv"
    )

    labeled, unlabeled = process_dataset(
        raw_dir=raw_battery1_dir,
        ref_file=reference_soh_path,
        output_labeled=processed_labeled_path,
        output_unlabeled=processed_unlabeled_path,
        interpolate=True,
    )

    validate_extracted_dataset(processed_labeled_path)
