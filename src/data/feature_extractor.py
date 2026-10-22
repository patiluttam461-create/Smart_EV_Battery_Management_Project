"""Feature extraction algorithms for Samsung INR18650-35E battery cycling data.

Extracts the seven key health indicator features per charge-discharge cycle:
1. tv_ch: elapsed time between charging voltage crossing 3.20 V and 4.15 V (seconds).
2. tv_d: elapsed time from active discharge onset to crossing 3.60 V (seconds).
3. tI_ch: elapsed time from main charging current start (I >= 1000 mA) until current
   falls to or below +500 mA during CV taper (seconds).
4. Tavg: mean temperature during charging phase (°C).
5. tTpeak_ch: elapsed time from charging-phase start to peak temperature during charging (seconds).
6. Eavg_ch: mean voltage during charging phase (Volts).
7. Eavg_d: mean voltage during discharging phase (Volts).
"""

from typing import Dict, List, Optional, Tuple


def linear_interpolate_crossing(
    t1: float, v1: float, t2: float, v2: float, target_v: float
) -> float:
    """Linearly interpolate timestamp where signal crosses target_v between (t1, v1) and (t2, v2)."""
    if v2 == v1:
        return t1
    frac = (target_v - v1) / (v2 - v1)
    return t1 + frac * (t2 - t1)


def segment_cycle_phases(
    records: List[Dict[str, float]],
    ch_current_threshold: float = 50.0,
    dis_current_threshold: float = -50.0,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Segment cycle telemetry records into charging and discharging subsets.

    Args:
        records: Chronologically sorted records for a single cycle.
        ch_current_threshold: Minimum current (mA) to qualify as charging.
        dis_current_threshold: Maximum current (mA) to qualify as discharging.

    Returns:
        Tuple of (charging_records, discharging_records).
    """
    ch_pts = [p for p in records if p["i"] >= ch_current_threshold]
    dis_pts = [p for p in records if p["i"] <= dis_current_threshold]
    return ch_pts, dis_pts


def extract_tv_ch(ch_pts: List[Dict[str, float]], interpolate: bool = True) -> float:
    """Calculate tv_ch: elapsed time between charging voltage crossing 3.20 V and 4.15 V.

    Args:
        ch_pts: Chronologically sorted charging records.
        interpolate: If True, uses linear interpolation for precise crossing time.

    Returns:
        tv_ch in seconds.

    Raises:
        ValueError: If voltage does not cross 3.20 V or 4.15 V.
    """
    t_320: Optional[float] = None
    t_415: Optional[float] = None

    for k in range(len(ch_pts)):
        p = ch_pts[k]
        if t_320 is None and p["v"] >= 3.20:
            if interpolate and k > 0 and ch_pts[k - 1]["v"] < 3.20:
                t_320 = linear_interpolate_crossing(
                    ch_pts[k - 1]["t"], ch_pts[k - 1]["v"], p["t"], p["v"], 3.20
                )
            else:
                t_320 = p["t"]

        if t_415 is None and p["v"] >= 4.15:
            if interpolate and k > 0 and ch_pts[k - 1]["v"] < 4.15:
                t_415 = linear_interpolate_crossing(
                    ch_pts[k - 1]["t"], ch_pts[k - 1]["v"], p["t"], p["v"], 4.15
                )
            else:
                t_415 = p["t"]
            break

    if t_320 is None:
        raise ValueError("tv_ch failed: charging voltage did not reach 3.20 V")
    if t_415 is None:
        raise ValueError("tv_ch failed: charging voltage did not reach 4.15 V")
    if t_415 < t_320:
        raise ValueError(f"tv_ch invalid: t_415 ({t_415}) < t_320 ({t_320})")

    return t_415 - t_320


def extract_tv_d(dis_pts: List[Dict[str, float]], interpolate: bool = True) -> float:
    """Calculate tv_d: elapsed time from active discharge onset to crossing 3.60 V.

    Note on loaded terminal voltage:
        When -1.7 A discharge starts, terminal voltage drops immediately from ~4.20 V
        to ~4.14 V due to internal resistance (IR drop). Active discharge onset
        anchors the start timestamp.

    Args:
        dis_pts: Chronologically sorted discharging records.
        interpolate: If True, uses linear interpolation for 3.60 V crossing.

    Returns:
        tv_d in seconds.

    Raises:
        ValueError: If active discharge is absent or voltage does not reach 3.60 V.
    """
    # Active discharge onset: first point where discharge current is established (<= -1000 mA)
    main_dis_pts = [p for p in dis_pts if p["i"] <= -1000.0]
    if not main_dis_pts:
        main_dis_pts = dis_pts

    t_dis_onset = main_dis_pts[0]["t"]
    t_360: Optional[float] = None

    for k in range(len(main_dis_pts)):
        p = main_dis_pts[k]
        if p["v"] <= 3.60:
            if interpolate and k > 0 and main_dis_pts[k - 1]["v"] > 3.60:
                t_360 = linear_interpolate_crossing(
                    main_dis_pts[k - 1]["t"], main_dis_pts[k - 1]["v"], p["t"], p["v"], 3.60
                )
            else:
                t_360 = p["t"]
            break

    if t_360 is None:
        raise ValueError("tv_d failed: discharge voltage did not cross 3.60 V")
    if t_360 < t_dis_onset:
        raise ValueError(f"tv_d invalid: t_360 ({t_360}) < t_dis_onset ({t_dis_onset})")

    return t_360 - t_dis_onset


def extract_tI_ch(ch_pts: List[Dict[str, float]], interpolate: bool = True) -> float:
    """Calculate tI_ch: elapsed time from main charging phase start (I >= 1000 mA)
    until charging current falls to or below +500 mA during CV taper.

    Guards against transition artifacts:
        Only checks for current taper once the cell enters high voltage (V >= 4.15 V),
        preventing false triggering during the initial current rise from rest.

    Args:
        ch_pts: Chronologically sorted charging records.
        interpolate: If True, uses linear interpolation for the 500 mA crossing.

    Returns:
        tI_ch in seconds.

    Raises:
        ValueError: If main charging phase is missing or current does not taper to 500 mA.
    """
    main_ch_idx: Optional[int] = None
    for k, p in enumerate(ch_pts):
        if p["i"] >= 1000.0:
            main_ch_idx = k
            break

    if main_ch_idx is None:
        raise ValueError("tI_ch failed: no main charging current (>= 1000 mA) found")

    t_main_ch_start = ch_pts[main_ch_idx]["t"]
    t_I500: Optional[float] = None

    # Search for CV current decay after cell reaches upper voltage regime
    for k in range(main_ch_idx, len(ch_pts)):
        p = ch_pts[k]
        if p["v"] >= 4.15 and p["i"] <= 500.0:
            if interpolate and k > main_ch_idx and ch_pts[k - 1]["i"] > 500.0:
                t_I500 = linear_interpolate_crossing(
                    ch_pts[k - 1]["t"], ch_pts[k - 1]["i"], p["t"], p["i"], 500.0
                )
            else:
                t_I500 = p["t"]
            break

    if t_I500 is None:
        raise ValueError("tI_ch failed: charging current did not taper to <= 500 mA")
    if t_I500 < t_main_ch_start:
        raise ValueError(f"tI_ch invalid: t_I500 ({t_I500}) < t_start ({t_main_ch_start})")

    return t_I500 - t_main_ch_start


def extract_temperature_features(
    ch_pts: List[Dict[str, float]]
) -> Tuple[float, float]:
    """Calculate Tavg and tTpeak_ch during charging phase.

    Args:
        ch_pts: Chronologically sorted charging records.

    Returns:
        Tuple of (Tavg in °C, tTpeak_ch in seconds).

    Raises:
        ValueError: If charging points are empty.
    """
    if not ch_pts:
        raise ValueError("Temperature feature extraction failed: empty charging records")

    # Tavg: arithmetic mean temperature during charging
    Tavg = sum(p["temp"] for p in ch_pts) / len(ch_pts)

    # tTpeak_ch: time from charging start to peak temperature
    t_ch_start = ch_pts[0]["t"]
    max_temp = -float("inf")
    t_peak_temp = t_ch_start

    for p in ch_pts:
        if p["temp"] > max_temp:
            max_temp = p["temp"]
            t_peak_temp = p["t"]

    tTpeak_ch = t_peak_temp - t_ch_start
    return Tavg, tTpeak_ch


def extract_voltage_features(
    ch_pts: List[Dict[str, float]], dis_pts: List[Dict[str, float]]
) -> Tuple[float, float]:
    """Calculate Eavg_ch and Eavg_d.

    Args:
        ch_pts: Chronologically sorted charging records.
        dis_pts: Chronologically sorted discharging records.

    Returns:
        Tuple of (Eavg_ch in Volts, Eavg_d in Volts).

    Raises:
        ValueError: If charging or discharging records are empty.
    """
    if not ch_pts:
        raise ValueError("Eavg_ch failed: empty charging records")
    if not dis_pts:
        raise ValueError("Eavg_d failed: empty discharging records")

    Eavg_ch = sum(p["v"] for p in ch_pts) / len(ch_pts)
    Eavg_d = sum(p["v"] for p in dis_pts) / len(dis_pts)
    return Eavg_ch, Eavg_d


def extract_all_cycle_features(
    records: List[Dict[str, float]], interpolate: bool = True
) -> Dict[str, float]:
    """Extract all 7 features for a single battery cycle.

    Args:
        records: List of measurement dictionaries for a single cycle.
        interpolate: If True, uses linear interpolation for threshold crossings.

    Returns:
        Dictionary mapping feature names to extracted float values:
        - tv_ch (s)
        - tv_d (s)
        - tI_ch (s)
        - Tavg (°C)
        - tTpeak_ch (s)
        - Eavg_ch (V)
        - Eavg_d (V)
    """
    if not records:
        raise ValueError("Cannot extract features: cycle record list is empty")

    ch_pts, dis_pts = segment_cycle_phases(records)

    if not ch_pts:
        raise ValueError("Cannot extract features: no charging phase identified")
    if not dis_pts:
        raise ValueError("Cannot extract features: no discharging phase identified")

    tv_ch = extract_tv_ch(ch_pts, interpolate=interpolate)
    tv_d = extract_tv_d(dis_pts, interpolate=interpolate)
    tI_ch = extract_tI_ch(ch_pts, interpolate=interpolate)
    Tavg, tTpeak_ch = extract_temperature_features(ch_pts)
    Eavg_ch, Eavg_d = extract_voltage_features(ch_pts, dis_pts)

    return {
        "tv_ch": tv_ch,
        "tv_d": tv_d,
        "tI_ch": tI_ch,
        "Tavg": Tavg,
        "tTpeak_ch": tTpeak_ch,
        "Eavg_ch": Eavg_ch,
        "Eavg_d": Eavg_d,
    }
