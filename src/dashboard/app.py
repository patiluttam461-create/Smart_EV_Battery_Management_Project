"""Single-window interactive application for Smart EV Battery Management System.

ANFIS-Based Battery State of Health Estimation dashboard implementing a
dark-mode interface matching the capstone design specification.
"""

from typing import Any, Dict, Optional
import flet as ft
from src.dashboard.inference_engine import (
    FEATURE_BOUNDS,
    FEATURE_NAMES,
    PRESET_EXAMPLES,
    BatteryInferenceEngine,
)


def build_app(page: ft.Page) -> None:
    """Build and configure the single-window SOH Predictor interface."""
    page.title = "SMART EV BATTERY MANAGEMENT SYSTEM"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#0B1220"
    page.padding = 18
    page.scroll = ft.ScrollMode.ADAPTIVE
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # Initialize inference engine
    try:
        inference_engine = BatteryInferenceEngine()
    except Exception as e:
        page.add(
            ft.Container(
                content=ft.Column([
                    ft.Text(
                        "Initialization Error",
                        size=20,
                        weight=ft.FontWeight.BOLD,
                        color="#EF4444",
                    ),
                    ft.Text(
                        f"Could not load required model artifacts:\n{str(e)}",
                        color="#FCA5A5",
                    ),
                ]),
                bgcolor="#111827",
                padding=18,
                border_radius=8,
                border=ft.Border.all(1, "#EF4444"),
            )
        )
        return

    # =========================================================================
    # 1. TOP HEADER CARD
    # =========================================================================
    header_status_badge = ft.Container(
        content=ft.Row(
            [
                ft.Icon(ft.Icons.CHECK_CIRCLE, size=15, color="#10B981"),
                ft.Text(
                    "ANFIS MODEL ACTIVE",
                    size=12,
                    weight=ft.FontWeight.BOLD,
                    color="#10B981",
                ),
            ],
            spacing=6,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        bgcolor="#064E3B",
        padding=ft.Padding.symmetric(horizontal=14, vertical=6),
        border_radius=20,
        border=ft.Border.all(1, "#10B981"),
    )

    header = ft.Container(
        content=ft.Row(
            [
                ft.Column(
                    [
                        ft.Text(
                            "SMART EV BATTERY MANAGEMENT SYSTEM",
                            size=22,
                            weight=ft.FontWeight.BOLD,
                            color="#F8FAFC",
                        ),
                        ft.Text(
                            "ANFIS-Based Battery State of Health Estimation",
                            size=13,
                            weight=ft.FontWeight.W_500,
                            color="#22C7F2",
                        ),
                    ],
                    spacing=2,
                ),
                header_status_badge,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        bgcolor="#111827",
        padding=ft.Padding.symmetric(horizontal=20, vertical=16),
        border_radius=10,
        border=ft.Border.all(1, "#243247"),
    )

    # =========================================================================
    # 2. SEVEN NUMERIC INPUT FIELDS (LEFT PANEL - 2 COLUMNS)
    # =========================================================================
    # Canonical labels matching the exact reference screenshot
    field_labels = {
        "tv_ch": "Charging Voltage Time (tv_ch) [s]",
        "tv_d": "Discharging Voltage Time (tv_d) [s]",
        "tI_ch": "Charging Current Time (tl_ch) [s]",
        "Tavg": "Average Battery Temperature (Tavg) [°C]",
        "tTpeak_ch": "Time to Peak Temperature (tTpeak_ch) [s]",
        "Eavg_ch": "Average Charging Voltage (Eavg_ch) [V]",
        "Eavg_d": "Average Discharging Voltage (Eavg_d) [V]",
    }

    input_fields: Dict[str, ft.TextField] = {}
    for feat in FEATURE_NAMES:
        low, high, unit, desc = FEATURE_BOUNDS[feat]
        placeholder = field_labels[feat]
        input_fields[feat] = ft.TextField(
            hint_text=placeholder,
            value="",
            text_size=13,
            dense=True,
            content_padding=ft.Padding.symmetric(horizontal=14, vertical=12),
            border_color="#243247",
            focused_border_color="#22C7F2",
            color="#F8FAFC",
            bgcolor="#1E293B",
            hint_style=ft.TextStyle(color="#94A3B8", size=12),
            border_radius=8,
            expand=True,
            tooltip=f"{placeholder} (Range: {low:.1f} - {high:.1f} {unit})",
        )

    # Small italic note explaining units (occupies the right slot in Row 4)
    units_note = ft.Container(
        content=ft.Text(
            "Units: Times in seconds [s], Temp in °C, Voltages in Volts [V].",
            size=11,
            italic=True,
            color="#94A3B8",
        ),
        alignment=ft.Alignment(-1, 0),
        padding=ft.Padding.symmetric(horizontal=8),
        expand=True,
    )

    # Status text for background tracking and test verification
    status_text = ft.Text(
        "Ready. Select an example or enter values manually, then click 'PREDICT SOH'.",
        size=11,
        color="#94A3B8",
        visible=False,
    )

    # Reference metadata tracking
    loaded_reference_info = {"name": None, "cn": None, "soh": None}

    # =========================================================================
    # 3. RIGHT PANEL: ANFIS PREDICTION CONTROLS
    # =========================================================================
    result_soh_text = ft.Text(
        "-- %",
        size=44,
        weight=ft.FontWeight.BOLD,
        color="#22C7F2",
    )

    result_health_badge = ft.Container(
        content=ft.Text(
            "AWAITING INPUT",
            size=11,
            weight=ft.FontWeight.BOLD,
            color="#94A3B8",
        ),
        bgcolor="#1E293B",
        padding=ft.Padding.symmetric(horizontal=12, vertical=5),
        border_radius=14,
        border=ft.Border.all(1, "#243247"),
    )

    result_ad_badge = ft.Container(
        content=ft.Text(
            "AD: Awaiting Input",
            size=11,
            weight=ft.FontWeight.BOLD,
            color="#94A3B8",
        ),
        bgcolor="#1E293B",
        padding=ft.Padding.symmetric(horizontal=10, vertical=5),
        border_radius=14,
        border=ft.Border.all(1, "#243247"),
        visible=False,
    )

    result_progress_bar = ft.ProgressBar(
        value=0.0,
        bgcolor="#0B1220",
        color="#22C7F2",
        height=8,
        border_radius=4,
    )

    result_degradation_text = ft.Text(
        "Degradation: -- %",
        size=11,
        color="#94A3B8",
        visible=False,
    )

    result_message_text = ft.Text(
        "Enter 7 battery parameters or select an example above, then click 'PREDICT SOH' to run ANFIS inference.",
        size=12,
        color="#94A3B8",
    )

    result_message_box = ft.Container(
        content=result_message_text,
        bgcolor="#0B1220",
        padding=ft.Padding.symmetric(horizontal=14, vertical=12),
        border_radius=8,
        border=ft.Border.all(1, "#243247"),
    )

    result_warning_banner = ft.Container(
        content=ft.Row(
            [
                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, size=18, color="#F59E0B"),
                ft.Text(
                    "",
                    size=11,
                    color="#FDE68A",
                    weight=ft.FontWeight.W_500,
                    expand=True,
                ),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        bgcolor="#78350F",
        padding=ft.Padding.symmetric(horizontal=12, vertical=8),
        border_radius=6,
        border=ft.Border.all(1, "#F59E0B"),
        visible=False,
    )

    thresholds_note = ft.Text(
        "Project-defined classification thresholds: • HEALTHY: SOH ≥ 90.0% • WARNING: 85.0% ≤ SOH < 90.0% • CRITICAL: SOH < 85.0%",
        size=10,
        italic=True,
        color="#64748B",
    )

    # =========================================================================
    # 4. EVENT HANDLERS
    # =========================================================================
    def load_preset(category_key: str, display_label: str) -> None:
        """Populate 7 input fields from authenticated laboratory cycling telemetry."""
        preset = PRESET_EXAMPLES.get(category_key)
        if not preset or not preset.get("features"):
            result_message_text.value = f"Selected example '{category_key}' is not available."
            result_message_text.color = "#EF4444"
            page.update()
            return

        # Populate features only — never pre-populate predictions
        for feat, val in preset["features"].items():
            input_fields[feat].value = str(val)

        cn = preset["CN"]
        ref_soh = preset["reference_soh"]
        loaded_reference_info["name"] = display_label
        loaded_reference_info["cn"] = cn
        loaded_reference_info["soh"] = ref_soh

        # Update status texts
        status_text.value = (
            f"Loaded '{display_label}' ({preset['cycle_name']}, CN={cn}). "
            f"Dataset Reference SOH: {ref_soh:.2f}% (measured laboratory benchmark). "
            f"Click 'PREDICT SOH' to run ANFIS inference."
        )

        result_soh_text.value = "-- %"
        result_soh_text.color = "#22C7F2"
        result_progress_bar.value = 0.0
        result_progress_bar.color = "#22C7F2"
        result_degradation_text.value = "Degradation: -- %"

        result_health_badge.content = ft.Text(
            "PRESET LOADED",
            size=11,
            weight=ft.FontWeight.BOLD,
            color="#22C7F2",
        )
        result_health_badge.bgcolor = "#0E3A5A"
        result_health_badge.border = ft.Border.all(1, "#22C7F2")

        result_ad_badge.content = ft.Text(
            "AD: Pending",
            size=11,
            weight=ft.FontWeight.BOLD,
            color="#94A3B8",
        )
        result_warning_banner.visible = False

        result_message_text.value = (
            f"Features populated from {display_label} (Cycle {cn}, Reference SOH: {ref_soh:.2f}%). "
            f"Click 'PREDICT SOH' to execute the ANFIS model."
        )
        result_message_text.color = "#94A3B8"
        page.update()

    def on_healthy_click(e) -> None:
        load_preset("Healthy Example", "Healthy Example")

    def on_warning_click(e) -> None:
        load_preset("Warning Example", "Warning Example")

    def on_critical_click(e) -> None:
        load_preset("Critical Example", "Critical Example")

    def on_clear_click(e) -> None:
        for feat in FEATURE_NAMES:
            input_fields[feat].value = ""
        loaded_reference_info["name"] = None
        loaded_reference_info["cn"] = None
        loaded_reference_info["soh"] = None

        status_text.value = (
            "Form reset. All input fields cleared. Select an example or enter values manually, "
            "then click 'PREDICT SOH'."
        )

        result_soh_text.value = "-- %"
        result_soh_text.color = "#22C7F2"
        result_progress_bar.value = 0.0
        result_progress_bar.color = "#22C7F2"
        result_degradation_text.value = "Degradation: -- %"

        result_health_badge.content = ft.Text(
            "AWAITING INPUT",
            size=11,
            weight=ft.FontWeight.BOLD,
            color="#94A3B8",
        )
        result_health_badge.bgcolor = "#1E293B"
        result_health_badge.border = ft.Border.all(1, "#243247")

        result_ad_badge.content = ft.Text(
            "AD: Awaiting Input",
            size=11,
            weight=ft.FontWeight.BOLD,
            color="#94A3B8",
        )
        result_warning_banner.visible = False

        result_message_text.value = (
            "Enter 7 battery parameters or select an example above, then click 'PREDICT SOH' to run ANFIS inference."
        )
        result_message_text.color = "#94A3B8"
        page.update()

    def on_predict_click(e) -> None:
        raw_dict = {feat: input_fields[feat].value for feat in FEATURE_NAMES}

        # Validate inputs against numerical and physical sanity checks
        is_valid, parsed_arr, errors = inference_engine.validate_inputs(raw_dict)
        if not is_valid:
            error_msg = errors[0] if errors else "Invalid feature input."
            status_text.value = f"Validation error: {error_msg}"
            result_soh_text.value = "Error"
            result_soh_text.color = "#EF4444"
            result_progress_bar.value = 0.0
            result_progress_bar.color = "#EF4444"
            result_degradation_text.value = "Degradation: N/A"

            result_health_badge.content = ft.Text(
                "INVALID INPUT",
                size=11,
                weight=ft.FontWeight.BOLD,
                color="#EF4444",
            )
            result_health_badge.bgcolor = "#7F1D1D"
            result_health_badge.border = ft.Border.all(1, "#EF4444")

            result_message_text.value = f"Validation Error: {error_msg} Check physical battery bounds."
            result_message_text.color = "#EF4444"
            result_warning_banner.visible = False
            page.update()
            return

        # Perform actual ANFIS inference
        try:
            res = inference_engine.predict(parsed_arr, model_name="ANFIS")
            pred_soh = res["predicted_soh"]
            degradation = res["degradation_percent"]
            health_status = res["health_status"]
            ad_status = res["ad_status"]
            d_m = res["mahalanobis_distance"]

            result_soh_text.value = f"{pred_soh:.2f} %"
            result_degradation_text.value = f"Degradation: {degradation:.2f}%"

            # Set horizontal progress bar gauge value (scaled 0.0 to 1.0)
            norm_val = min(max(pred_soh / 100.0, 0.0), 1.0)
            result_progress_bar.value = norm_val

            # Color-code health classification badge and gauge
            if pred_soh >= 90.0:
                cond_color = "#10B981"  # Emerald green
                cond_bg = "#064E3B"
                cond_text = "HEALTHY"
            elif pred_soh >= 85.0:
                cond_color = "#F59E0B"  # Amber
                cond_bg = "#78350F"
                cond_text = "WARNING"
            else:
                cond_color = "#EF4444"  # Red
                cond_bg = "#7F1D1D"
                cond_text = "CRITICAL"

            result_soh_text.color = cond_color
            result_progress_bar.color = cond_color

            result_health_badge.content = ft.Text(
                cond_text,
                size=11,
                weight=ft.FontWeight.BOLD,
                color=cond_color,
            )
            result_health_badge.bgcolor = cond_bg
            result_health_badge.border = ft.Border.all(1, cond_color)

            # Update AD badge for test compatibility
            if ad_status == "IN_DOMAIN":
                ad_label = f"In-Domain (D_M={d_m:.2f})"
            elif ad_status == "MODERATE_DRIFT":
                ad_label = f"Moderate Drift (D_M={d_m:.2f})"
            elif ad_status == "EXTRAPOLATION":
                ad_label = f"Extrapolation (D_M={d_m:.2f})"
            else:
                ad_label = f"Outlier (D_M={d_m:.2f})"

            result_ad_badge.content = ft.Text(
                ad_label,
                size=11,
                weight=ft.FontWeight.BOLD,
                color=cond_color,
            )

            # Informative message text
            delta_note = ""
            if loaded_reference_info["soh"] is not None:
                ref_soh = loaded_reference_info["soh"]
                delta = pred_soh - ref_soh
                delta_note = f" [Dataset Ref SOH: {ref_soh:.2f}%, Delta: {delta:+.2f}%]"

            result_message_text.value = (
                f"Predicted SOH: {pred_soh:.2f}% | Condition: {cond_text} "
                f"(Degradation: {degradation:.2f}%){delta_note}. {res['health_desc']}"
            )
            result_message_text.color = "#F8FAFC"

            # Applicability warning banner
            if res.get("ad_warning"):
                result_warning_banner.content.controls[1].value = res["ad_warning"]
                result_warning_banner.visible = True
            else:
                result_warning_banner.visible = False

            status_text.value = (
                f"Inference completed. Estimated SOH: {pred_soh:.2f}%, "
                f"Degradation: {degradation:.2f}%, D_M: {d_m:.2f}."
            )

        except Exception as ex:
            status_text.value = f"Inference execution failed: {str(ex)}"
            result_soh_text.value = "Error"
            result_soh_text.color = "#EF4444"
            result_progress_bar.value = 0.0
            result_health_badge.content = ft.Text(
                "ERROR",
                size=11,
                weight=ft.FontWeight.BOLD,
                color="#EF4444",
            )
            result_health_badge.bgcolor = "#7F1D1D"
            result_message_text.value = f"Prediction failed: {str(ex)}"
            result_message_text.color = "#EF4444"
            result_warning_banner.visible = False

        page.update()

    # =========================================================================
    # 5. BUTTON WIDGETS
    # =========================================================================
    # Primary Predict Button (Blue pill with lightning bolt icon)
    predict_btn = ft.Button(
        content=ft.Row(
            [
                ft.Text("⚡", size=14, color="#F8FAFC"),
                ft.Text("PREDICT SOH", weight=ft.FontWeight.BOLD, color="#F8FAFC", size=13),
            ],
            spacing=6,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        bgcolor="#0284C7",
        height=40,
        width=160,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=20),
        ),
        on_click=on_predict_click,
    )

    # Secondary Clear Button (Slate pill with clear icon)
    clear_btn = ft.Button(
        content=ft.Row(
            [
                ft.Text("☰", size=14, color="#94A3B8"),
                ft.Text("CLEAR", weight=ft.FontWeight.BOLD, color="#94A3B8", size=13),
            ],
            spacing=6,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        bgcolor="#1E293B",
        height=40,
        width=120,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=20),
            side=ft.BorderSide(1, "#243247"),
        ),
        on_click=on_clear_click,
    )

    # 3 Color-Coded Preset Buttons
    healthy_btn = ft.Button(
        content=ft.Row(
            [
                ft.Icon(ft.Icons.CHECK_CIRCLE, size=15, color="#10B981"),
                ft.Text("Healthy Example", size=11, weight=ft.FontWeight.BOLD, color="#10B981"),
            ],
            spacing=6,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        bgcolor="#064E3B",
        height=36,
        expand=True,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=20),
            side=ft.BorderSide(1, "#10B981"),
        ),
        on_click=on_healthy_click,
    )

    warning_btn = ft.Button(
        content=ft.Row(
            [
                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, size=15, color="#F59E0B"),
                ft.Text("Warning Example", size=11, weight=ft.FontWeight.BOLD, color="#F59E0B"),
            ],
            spacing=6,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        bgcolor="#78350F",
        height=36,
        expand=True,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=20),
            side=ft.BorderSide(1, "#F59E0B"),
        ),
        on_click=on_warning_click,
    )

    critical_btn = ft.Button(
        content=ft.Row(
            [
                ft.Icon(ft.Icons.ERROR_OUTLINE, size=15, color="#EF4444"),
                ft.Text("Critical Example", size=11, weight=ft.FontWeight.BOLD, color="#EF4444"),
            ],
            spacing=6,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        bgcolor="#7F1D1D",
        height=36,
        expand=True,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=20),
            side=ft.BorderSide(1, "#EF4444"),
        ),
        on_click=on_critical_click,
    )

    # =========================================================================
    # 6. ASSEMBLE LEFT PANEL: BATTERY CYCLE PARAMETERS
    # =========================================================================
    left_panel = ft.Container(
        content=ft.Column(
            [
                # Heading Row
                ft.Row(
                    [
                        ft.Icon(ft.Icons.TUNE, size=20, color="#22C7F2"),
                        ft.Text(
                            "BATTERY CYCLE PARAMETERS",
                            size=15,
                            weight=ft.FontWeight.BOLD,
                            color="#F8FAFC",
                        ),
                    ],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                # Description
                ft.Text(
                    "Enter battery charging/discharging cycle parameters used by the ANFIS model.",
                    size=12,
                    color="#94A3B8",
                ),
                # 7 Numeric Fields in 2-Column Grid
                ft.Row([input_fields["tv_ch"], input_fields["tv_d"]], spacing=12),
                ft.Row([input_fields["tI_ch"], input_fields["Tavg"]], spacing=12),
                ft.Row([input_fields["tTpeak_ch"], input_fields["Eavg_ch"]], spacing=12),
                ft.Row([input_fields["Eavg_d"], units_note], spacing=12),
                # Buttons Row
                ft.Row([predict_btn, clear_btn], spacing=12),
                # Demonstration Examples Heading
                ft.Text(
                    "DEMONSTRATION EXAMPLES (VERIFIED DATASET RECORDS):",
                    size=11,
                    weight=ft.FontWeight.BOLD,
                    color="#94A3B8",
                ),
                # Presets Row
                ft.Row([healthy_btn, warning_btn, critical_btn], spacing=10),
            ],
            spacing=12,
        ),
        bgcolor="#111827",
        padding=18,
        border_radius=10,
        border=ft.Border.all(1, "#243247"),
        expand=1,
    )

    # =========================================================================
    # 7. ASSEMBLE RIGHT COLUMN: PREDICTION & MODEL INFORMATION
    # =========================================================================
    # Top Card: ANFIS Prediction
    prediction_card = ft.Container(
        content=ft.Column(
            [
                # Card Header
                ft.Row(
                    [
                        ft.Row(
                            [
                                ft.Icon(ft.Icons.INSERT_CHART_OUTLINED, size=20, color="#22C7F2"),
                                ft.Text(
                                    "ANFIS PREDICTION",
                                    size=15,
                                    weight=ft.FontWeight.BOLD,
                                    color="#F8FAFC",
                                ),
                            ],
                            spacing=8,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        result_health_badge,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                # SOH Value Display
                ft.Column(
                    [
                        ft.Text(
                            "Predicted State of Health (SOH):",
                            size=12,
                            color="#94A3B8",
                        ),
                        result_soh_text,
                    ],
                    spacing=1,
                ),
                # Progress Bar Gauge
                result_progress_bar,
                # Result Message Box
                result_message_box,
                result_warning_banner,
                # Thresholds Note
                thresholds_note,
            ],
            spacing=10,
        ),
        bgcolor="#111827",
        padding=18,
        border_radius=10,
        border=ft.Border.all(1, "#243247"),
    )

    # Bottom Card: ANFIS Model Information
    model_info_card = ft.Container(
        content=ft.Column(
            [
                # Card Header
                ft.Row(
                    [
                        ft.Row(
                            [
                                ft.Icon(ft.Icons.MEMORY, size=20, color="#22C7F2"),
                                ft.Text(
                                    "ANFIS MODEL INFORMATION",
                                    size=15,
                                    weight=ft.FontWeight.BOLD,
                                    color="#F8FAFC",
                                ),
                            ],
                            spacing=8,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Container(
                            content=ft.Row(
                                [
                                    ft.Container(width=6, height=6, border_radius=3, bgcolor="#10B981"),
                                    ft.Text(
                                        "ANFIS V5: Ready for Prediction",
                                        size=11,
                                        weight=ft.FontWeight.BOLD,
                                        color="#10B981",
                                    ),
                                ],
                                spacing=6,
                            ),
                            bgcolor="#064E3B",
                            padding=ft.Padding.symmetric(horizontal=10, vertical=5),
                            border_radius=14,
                            border=ft.Border.all(1, "#10B981"),
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                # Two Columns of Specifications
                ft.Row(
                    [
                        ft.Column(
                            [
                                ft.Text("• Model: Adaptive Neuro-Fuzzy Inference System", size=12, color="#94A3B8"),
                                ft.Text("• Inference: First-Order Sugeno ANFIS", size=12, color="#94A3B8"),
                                ft.Text("• Inputs: 7 Cycle Parameters", size=12, color="#94A3B8"),
                                ft.Text("• Output: Battery SOH (%) [Regression]", size=12, color="#94A3B8"),
                            ],
                            spacing=3,
                            expand=True,
                        ),
                        ft.Column(
                            [
                                ft.Text("• Membership Functions: 2 Gaussian MFs / Input", size=12, color="#94A3B8"),
                                ft.Text("• Fuzzy Rules: 2 Rules (Scatter Clustering Matrix)", size=12, color="#94A3B8"),
                                ft.Text("• Features: tv_ch, tv_d, tl_ch, Tavg, tTpeak_ch, Eavg_ch, Eavg_d", size=12, color="#94A3B8"),
                                ft.Text("• Pipeline: Scaler Transform → ANFIS → Scaler Inverse", size=12, color="#94A3B8"),
                            ],
                            spacing=3,
                            expand=True,
                        ),
                    ],
                    spacing=16,
                ),
                # Compact Verified Benchmark row for academic defensibility
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Text("Test MAE", size=10, color="#94A3B8"),
                                        ft.Text("0.8246 %", size=13, weight=ft.FontWeight.BOLD, color="#10B981"),
                                    ],
                                    spacing=1,
                                ),
                                expand=True,
                            ),
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Text("Test RMSE", size=10, color="#94A3B8"),
                                        ft.Text("0.8574 %", size=13, weight=ft.FontWeight.BOLD, color="#22C7F2"),
                                    ],
                                    spacing=1,
                                ),
                                expand=True,
                            ),
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Text("Test R²", size=10, color="#94A3B8"),
                                        ft.Text("-1.4560", size=13, weight=ft.FontWeight.BOLD, color="#F59E0B"),
                                    ],
                                    spacing=1,
                                ),
                                expand=True,
                            ),
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Text("Parameters", size=10, color="#94A3B8"),
                                        ft.Text("44 (28p + 16c)", size=13, weight=ft.FontWeight.BOLD, color="#F8FAFC"),
                                    ],
                                    spacing=1,
                                ),
                                expand=True,
                            ),
                        ],
                        spacing=8,
                    ),
                    bgcolor="#0B1220",
                    padding=ft.Padding.symmetric(horizontal=12, vertical=6),
                    border_radius=6,
                    border=ft.Border.all(1, "#243247"),
                ),
            ],
            spacing=10,
        ),
        bgcolor="#111827",
        padding=18,
        border_radius=10,
        border=ft.Border.all(1, "#243247"),
    )

    right_column = ft.Column(
        [
            prediction_card,
            model_info_card,
        ],
        spacing=14,
        expand=1,
    )

    # Middle Row: Left Panel + Right Column
    middle_row = ft.Row(
        [left_panel, right_column],
        spacing=14,
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )

    # =========================================================================
    # 8. FULL-WIDTH BOTTOM CARD: ANFIS INFERENCE PROCESS FLOW
    # =========================================================================
    def build_step_box(step_num: str, title: str) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(f"STEP {step_num}", size=10, weight=ft.FontWeight.BOLD, color="#22C7F2", text_align=ft.TextAlign.CENTER),
                    ft.Text(title, size=11, weight=ft.FontWeight.BOLD, color="#F8FAFC", text_align=ft.TextAlign.CENTER),
                ],
                spacing=2,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            bgcolor="#0B1220",
            padding=ft.Padding.symmetric(horizontal=8, vertical=12),
            border_radius=8,
            border=ft.Border.all(1, "#243247"),
            expand=True,
            alignment=ft.Alignment(0, 0),
        )

    flow_steps = [
        build_step_box("1", "BATTERY CYCLE DATA"),
        build_step_box("2", "FEATURE INPUT (7)"),
        build_step_box("3", "FUZZIFICATION"),
        build_step_box("4", "2 FUZZY RULES"),
        build_step_box("5", "NORMALIZATION"),
        build_step_box("6", "SUGENO INFERENCE"),
        build_step_box("7", "SOH PREDICTION %"),
    ]

    flow_row_elements = []
    for i, step_box in enumerate(flow_steps):
        flow_row_elements.append(step_box)
        if i < len(flow_steps) - 1:
            flow_row_elements.append(
                ft.Text("→", size=16, weight=ft.FontWeight.BOLD, color="#22C7F2")
            )

    process_flow_card = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.ACCOUNT_TREE, size=20, color="#22C7F2"),
                        ft.Text(
                            "ANFIS INFERENCE PROCESS FLOW",
                            size=15,
                            weight=ft.FontWeight.BOLD,
                            color="#F8FAFC",
                        ),
                    ],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Row(
                    flow_row_elements,
                    spacing=6,
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
            spacing=12,
        ),
        bgcolor="#111827",
        padding=18,
        border_radius=10,
        border=ft.Border.all(1, "#243247"),
    )

    # =========================================================================
    # 9. FOOTER
    # =========================================================================
    footer = ft.Container(
        content=ft.Text(
            "Soft Computing + AIML Project • ANFIS-Based EV Battery State of Health Estimation",
            size=11,
            color="#64748B",
            text_align=ft.TextAlign.CENTER,
        ),
        alignment=ft.Alignment(0, 0),
        padding=ft.Padding.symmetric(vertical=4),
    )

    # Main application column
    main_layout = ft.Container(
        content=ft.Column(
            [
                header,
                middle_row,
                process_flow_card,
                footer,
            ],
            spacing=14,
        ),
        width=1340,
    )

    page.add(main_layout)

    # Expose control references on page for reliable, uncoupled testing
    page.app_controls = {
        "header": header,
        "input_fields": input_fields,
        "healthy_btn": healthy_btn,
        "warning_btn": warning_btn,
        "critical_btn": critical_btn,
        "predict_btn": predict_btn,
        "clear_btn": clear_btn,
        "reset_btn": clear_btn,
        "soh_text": result_soh_text,
        "progress_bar": result_progress_bar,
        "health_badge": result_health_badge,
        "ad_badge": result_ad_badge,
        "degradation_text": result_degradation_text,
        "warning_banner": result_warning_banner,
        "interpretation_text": result_message_text,
        "status_text": status_text,
        "left_panel": left_panel,
        "prediction_card": prediction_card,
        "model_info_card": model_info_card,
        "process_flow_card": process_flow_card,
    }


def main() -> None:
    """Launch the Flet desktop application."""
    ft.run(build_app)


if __name__ == "__main__":
    main()
