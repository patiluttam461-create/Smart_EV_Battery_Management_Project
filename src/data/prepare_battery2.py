"""Battery 2 dataset preparation pipeline for cross-battery generalization evaluation.

Extracts the 7 health indicator features for Battery 2 cycles from raw measurement files
(Zenodo Record 15657829) and pairs them with ground-truth reference SOH from
VZ5_008_021_VUT_D_0043_v1.txt (column 'Battery2').

Enforces strict schema parity, boundary validation, and zero-leakage isolation from Battery 1.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from src.data.feature_extractor import extract_all_cycle_features

# Reference SOH filename and column name
REF_FILENAME = "VZ5_008_021_VUT_D_0043_v1.txt"
BATTERY2_COL_NAME = "Battery2"

# Canonical feature ordering required across all models
FEATURE_NAMES: List[str] = [
    "tv_ch",
    "tv_d",
    "tI_ch",
    "Tavg",
    "tTpeak_ch",
    "Eavg_ch",
    "Eavg_d",
]

# Physical reference validation bounds based on Samsung INR18650-35E experimental cycling
FEATURE_BOUNDS: Dict[str, Tuple[float, float, str]] = {
    "tv_ch": (3000.0, 7000.0, "seconds"),
    "tv_d": (2000.0, 5000.0, "seconds"),
    "tI_ch": (4500.0, 9000.0, "seconds"),
    "Tavg": (10.0, 50.0, "°C"),
    "tTpeak_ch": (100.0, 8500.0, "seconds"),
    "Eavg_ch": (3.80, 4.30, "Volts"),
    "Eavg_d": (3.20, 3.80, "Volts"),
}


def load_battery2_reference_soh(ref_filepath: str) -> Dict[int, float]:
    """Load ground-truth SOH for Battery 2 from reference file VZ5_008_021_VUT_D_0043_v1.txt.

    Args:
        ref_filepath: Path to the reference file containing CN, Battery1, Battery2.

    Returns:
        Dict mapping integer cycle number (CN 1..580) -> SOH percentage (float, 0..100%).

    Raises:
        FileNotFoundError: If reference file does not exist.
        ValueError: If file structure, header, or row count is invalid.
    """
    if not os.path.exists(ref_filepath):
        raise FileNotFoundError(
            f"Reference health file not found at: {ref_filepath}\n"
            f"Ensure data/raw/reference/{REF_FILENAME} is present."
        )

    soh_dict: Dict[int, float] = {}
    with open(ref_filepath, "r", encoding="latin-1") as f:
        header_line = f.readline().strip("\r\n")
        header = header_line.split("\t")
        if "CN" not in header or BATTERY2_COL_NAME not in header:
            raise ValueError(
                f"Invalid reference file header: '{header_line}'. "
                f"Expected columns 'CN' and '{BATTERY2_COL_NAME}'."
            )

        cn_idx = header.index("CN")
        b2_idx = header.index(BATTERY2_COL_NAME)

        for line_num, line in enumerate(f, start=2):
            parts = line.strip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            try:
                cn = int(parts[cn_idx])
                soh_frac = float(parts[b2_idx])
                # Express SOH in percentage points [80.0, 100.0]
                soh_percent = round(soh_frac * 100.0, 6)
                soh_dict[cn] = soh_percent
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Error parsing reference line {line_num} in {ref_filepath}: {line.strip()} - {e}"
                )

    if len(soh_dict) != 580:
        raise ValueError(
            f"Expected exactly 580 reference SOH entries for Battery 2, found {len(soh_dict)}."
        )

    return soh_dict


def check_battery2_raw_availability(raw_dir: str) -> Tuple[bool, List[str], str]:
    """Check whether Battery 2 raw cycling files are present in the designated raw directory.

    Args:
        raw_dir: Path to data/raw/battery2/ directory.

    Returns:
        Tuple of (is_available, file_list, diagnostic_message).
    """
    if not os.path.exists(raw_dir):
        return (
            False,
            [],
            f"Battery 2 raw directory does not exist: {raw_dir}\n"
            f"To evaluate cross-battery generalization, download the Battery 2 raw cycling files "
            f"from Zenodo Record 15657829 and place them in {raw_dir}.",
        )

    files = [f for f in os.listdir(raw_dir) if f.endswith(".txt")]
    if not files:
        return (
            False,
            [],
            f"Battery 2 raw directory exists ({raw_dir}) but contains no .txt measurement files.\n"
            f"Expected raw cycling files from Zenodo Record 15657829 (Samsung INR18650-35E Battery 2).",
        )

    return True, files, f"Found {len(files)} raw files in {raw_dir}."


def process_battery2_dataset(
    raw_dir: str,
    ref_file: str,
    output_path: str,
    cycle_mapping: Optional[List[Tuple[str, int, int, int, int]]] = None,
    interpolate: bool = True,
) -> pd.DataFrame:
    """Extract features from Battery 2 raw measurements and create battery2_features.csv.

    Follows the verified Samsung INR18650-35E experimental cycling protocol from Zenodo Record 15657829:
    - VZ5_..._0032_v1.txt: cycles 0..99 -> CN 1..100 (100 cycles)
    - VZ5_..._0033_v1.txt: cycles 0..28 -> CN 101..129 (29 cycles)
    - CN 130: Inter-run transition cycle (charge completed in 0033 c=29, discharge in 0034 c=0)
    - VZ5_..._0034_v1.txt: cycles 1..70 -> CN 131..200 (70 cycles)
    - VZ5_..._0035_v1.txt: cycles 0..99 -> CN 201..300 (100 cycles)
    - VZ5_..._0036_v1.txt: cycles 0..99 -> CN 301..400 (100 cycles)
    - VZ5_..._0037_v1.txt: cycles 0..99 -> CN 401..500 (100 cycles)
    - VZ5_..._0038_v1.txt: cycles 0..79 -> CN 501..580 (80 cycles)
    Total: exactly 580 cycles matching reference SOH.

    Args:
        raw_dir: Path to directory containing Battery 2 raw files (e.g. data/raw/battery2/).
        ref_file: Path to reference SOH file (data/raw/reference/VZ5_..._0043_v1.txt).
        output_path: Destination path for data/processed/battery2_features.csv.
        cycle_mapping: Optional custom cycle mapping list.
        interpolate: Whether to use linear interpolation for exact voltage crossing times.

    Returns:
        pd.DataFrame containing the processed Battery 2 dataset.

    Raises:
        FileNotFoundError: If Battery 2 raw files or reference file are missing.
        ValueError: If telemetry formatting or validation boundaries fail.
    """
    from src.data.feature_extractor import (
        extract_temperature_features,
        extract_tI_ch,
        extract_tv_ch,
        extract_tv_d,
        extract_voltage_features,
        segment_cycle_phases,
    )
    from src.data.prepare_dataset import read_raw_cycles

    # 1. Verify reference file
    soh_dict = load_battery2_reference_soh(ref_file)

    # 2. Verify raw files availability
    is_avail, raw_files, diag_msg = check_battery2_raw_availability(raw_dir)
    if not is_avail:
        raise FileNotFoundError(
            f"Cannot process Battery 2 dataset: raw measurement files are unavailable.\n"
            f"{diag_msg}\n"
            f"Ground-truth SOH for Battery 2 is verified in {ref_file}, but raw cycling measurements "
            f"must be downloaded from Zenodo Record 15657829 before feature extraction can run."
        )

    # Standard cycle schedule for Battery 2
    standard_segments = [
        ("VZ5_008_021_VUT_D_0032_v1.txt", 0, 99, 1),
        ("VZ5_008_021_VUT_D_0033_v1.txt", 0, 28, 101),
        ("VZ5_008_021_VUT_D_0034_v1.txt", 1, 70, 131),
        ("VZ5_008_021_VUT_D_0035_v1.txt", 0, 99, 201),
        ("VZ5_008_021_VUT_D_0036_v1.txt", 0, 99, 301),
        ("VZ5_008_021_VUT_D_0037_v1.txt", 0, 99, 401),
        ("VZ5_008_021_VUT_D_0038_v1.txt", 0, 79, 501),
    ]

    # Preload required raw files
    raw_files_dict: Dict[str, Dict[int, List[Dict[str, float]]]] = {}
    for fname, _, _, _ in standard_segments:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing required raw file: {fpath}")
        raw_files_dict[fname] = read_raw_cycles(fpath)

    labeled_rows: List[Dict] = []
    global_cycle_idx = 0

    # 1. Segment 0032 (Cycles 0..99 -> CN 1..100)
    for c in range(100):
        cn = c + 1
        pts = raw_files_dict["VZ5_008_021_VUT_D_0032_v1.txt"][c]
        ch, dis = segment_cycle_phases(pts)
        tavg, ttpeak = extract_temperature_features(ch)
        ech, ed = extract_voltage_features(ch, dis)
        feats = {
            "tv_ch": extract_tv_ch(ch, interpolate=interpolate),
            "tv_d": extract_tv_d(dis, interpolate=interpolate),
            "tI_ch": extract_tI_ch(ch, interpolate=interpolate),
            "Tavg": tavg,
            "tTpeak_ch": ttpeak,
            "Eavg_ch": ech,
            "Eavg_d": ed,
        }
        labeled_rows.append({
            "CN": cn,
            "global_cycle": global_cycle_idx,
            "source_file": "VZ5_008_021_VUT_D_0032_v1.txt",
            "source_cycle": c,
            "tv_ch": round(feats["tv_ch"], 4),
            "tv_d": round(feats["tv_d"], 4),
            "tI_ch": round(feats["tI_ch"], 4),
            "Tavg": round(feats["Tavg"], 4),
            "tTpeak_ch": round(feats["tTpeak_ch"], 4),
            "Eavg_ch": round(feats["Eavg_ch"], 6),
            "Eavg_d": round(feats["Eavg_d"], 6),
            "SOH": soh_dict[cn],
        })
        global_cycle_idx += 1

    # 2. Segment 0033 (Cycles 0..28 -> CN 101..129)
    for c in range(29):
        cn = 101 + c
        pts = raw_files_dict["VZ5_008_021_VUT_D_0033_v1.txt"][c]
        ch, dis = segment_cycle_phases(pts)
        tavg, ttpeak = extract_temperature_features(ch)
        ech, ed = extract_voltage_features(ch, dis)
        feats = {
            "tv_ch": extract_tv_ch(ch, interpolate=interpolate),
            "tv_d": extract_tv_d(dis, interpolate=interpolate),
            "tI_ch": extract_tI_ch(ch, interpolate=interpolate),
            "Tavg": tavg,
            "tTpeak_ch": ttpeak,
            "Eavg_ch": ech,
            "Eavg_d": ed,
        }
        labeled_rows.append({
            "CN": cn,
            "global_cycle": global_cycle_idx,
            "source_file": "VZ5_008_021_VUT_D_0033_v1.txt",
            "source_cycle": c,
            "tv_ch": round(feats["tv_ch"], 4),
            "tv_d": round(feats["tv_d"], 4),
            "tI_ch": round(feats["tI_ch"], 4),
            "Tavg": round(feats["Tavg"], 4),
            "tTpeak_ch": round(feats["tTpeak_ch"], 4),
            "Eavg_ch": round(feats["Eavg_ch"], 6),
            "Eavg_d": round(feats["Eavg_d"], 6),
            "SOH": soh_dict[cn],
        })
        global_cycle_idx += 1

    # 3. Inter-Run Transition Cycle CN 130
    # Charge completed in 0033 c=29, Discharge completed in 0034 c=0
    ch_130, _ = segment_cycle_phases(raw_files_dict["VZ5_008_021_VUT_D_0033_v1.txt"][29])
    _, dis_130 = segment_cycle_phases(raw_files_dict["VZ5_008_021_VUT_D_0034_v1.txt"][0])
    tavg_130, ttpeak_130 = extract_temperature_features(ch_130)
    ech_130, ed_130 = extract_voltage_features(ch_130, dis_130)
    labeled_rows.append({
        "CN": 130,
        "global_cycle": global_cycle_idx,
        "source_file": "VZ5_008_021_VUT_D_0033_v1.txt + VZ5_008_021_VUT_D_0034_v1.txt",
        "source_cycle": 29,
        "tv_ch": round(extract_tv_ch(ch_130, interpolate=interpolate), 4),
        "tv_d": round(extract_tv_d(dis_130, interpolate=interpolate), 4),
        "tI_ch": round(extract_tI_ch(ch_130, interpolate=interpolate), 4),
        "Tavg": round(tavg_130, 4),
        "tTpeak_ch": round(ttpeak_130, 4),
        "Eavg_ch": round(ech_130, 6),
        "Eavg_d": round(ed_130, 6),
        "SOH": soh_dict[130],
    })
    global_cycle_idx += 1

    # 4. Remaining Segments (0034 c=1..70, 0035 c=0..99, 0036 c=0..99, 0037 c=0..99, 0038 c=0..79)
    remaining_segments = [
        ("VZ5_008_021_VUT_D_0034_v1.txt", 1, 70, 131),
        ("VZ5_008_021_VUT_D_0035_v1.txt", 0, 99, 201),
        ("VZ5_008_021_VUT_D_0036_v1.txt", 0, 99, 301),
        ("VZ5_008_021_VUT_D_0037_v1.txt", 0, 99, 401),
        ("VZ5_008_021_VUT_D_0038_v1.txt", 0, 79, 501),
    ]

    for fname, start_c, end_c, start_cn in remaining_segments:
        c_dict = raw_files_dict[fname]
        for c in range(start_c, end_c + 1):
            cn = start_cn + (c - start_c)
            pts = c_dict[c]
            ch, dis = segment_cycle_phases(pts)
            tavg, ttpeak = extract_temperature_features(ch)
            ech, ed = extract_voltage_features(ch, dis)
            feats = {
                "tv_ch": extract_tv_ch(ch, interpolate=interpolate),
                "tv_d": extract_tv_d(dis, interpolate=interpolate),
                "tI_ch": extract_tI_ch(ch, interpolate=interpolate),
                "Tavg": tavg,
                "tTpeak_ch": ttpeak,
                "Eavg_ch": ech,
                "Eavg_d": ed,
            }
            labeled_rows.append({
                "CN": cn,
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
                "SOH": soh_dict[cn],
            })
            global_cycle_idx += 1

    df = pd.DataFrame(labeled_rows)

    # Validate physical plausibility bounds
    for feat_name, (low, high, unit) in FEATURE_BOUNDS.items():
        min_v = df[feat_name].min()
        max_v = df[feat_name].max()
        if min_v < low or max_v > high:
            raise ValueError(
                f"Feature '{feat_name}' out of physical bounds [{low}, {high}]: "
                f"observed range [{min_v}, {max_v}] {unit}"
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main():
    """CLI entrypoint for Battery 2 dataset preparation."""
    raw_dir = os.path.join(_PROJECT_ROOT, "data", "raw", "battery2")
    ref_file = os.path.join(_PROJECT_ROOT, "data", "raw", "reference", REF_FILENAME)
    output_path = os.path.join(_PROJECT_ROOT, "data", "processed", "battery2_features.csv")

    print("========================================================================")
    print("BATTERY 2 DATASET PREPARATION PIPELINE")
    print("========================================================================")
    print(f"Reference SOH File: {ref_file}")
    print(f"Raw Telemetry Dir:  {raw_dir}")
    print(f"Output Target:      {output_path}\n")

    # Check reference SOH
    try:
        soh_dict = load_battery2_reference_soh(ref_file)
        print(f"[OK] Reference SOH verified: {len(soh_dict)} cycles loaded from {REF_FILENAME}.")
        print(f"     SOH range: {min(soh_dict.values()):.4f}% to {max(soh_dict.values()):.4f}%\n")
    except Exception as e:
        print(f"[ERROR] Reference SOH check failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Check raw data availability
    is_avail, files, diag_msg = check_battery2_raw_availability(raw_dir)
    if not is_avail:
        print("[STATUS: AWAITING RAW DATA]")
        print(diag_msg, file=sys.stderr)
        sys.exit(1)

    print(f"[OK] Found {len(files)} raw measurement files in {raw_dir}.")
    print("Extracting canonical 7 health indicator features for 580 cycles...")

    try:
        df = process_battery2_dataset(
            raw_dir=raw_dir,
            ref_file=ref_file,
            output_path=output_path,
            interpolate=True,
        )
        print(f"\n[SUCCESS] Successfully processed {len(df)} cycles for Battery 2!")
        print(f"          Feature dataset saved to: {output_path}")
        print(f"          SOH range: {df['SOH'].min():.4f}% to {df['SOH'].max():.4f}%")
        print(f"          Features:  {FEATURE_NAMES}")
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

