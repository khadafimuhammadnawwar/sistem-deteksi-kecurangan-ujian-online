"""Unit test untuk pipeline PCD, kalibrasi 5 titik fisik, proyeksi gabungan POR, EMA, UI, dan logger."""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import (
    BannerState,
    CalibrationStage,
    DetectorConfig,
    IndicationCode,
    IntegrityLevel,
    MonitoringState,
    resolve_integrity_level,
)
from src.detector import (
    CHIN_IDX,
    L_EYE_BOTTOM_IDX,
    L_EYE_INNER_IDX,
    L_EYE_OUTER_IDX,
    L_EYE_TOP_IDX,
    L_IRIS_CENTER_IDX,
    LEFT_CHEEK_IDX,
    NOSE_TIP_IDX,
    R_EYE_BOTTOM_IDX,
    R_EYE_INNER_IDX,
    R_EYE_OUTER_IDX,
    R_EYE_TOP_IDX,
    R_IRIS_CENTER_IDX,
    RIGHT_CHEEK_IDX,
    DetectedFace,
    advance_five_point_calibration,
    calculate_iris_gaze_and_ear,
    calculate_yaw_pitch,
    evaluate_instant_indication,
    extract_faces_sorted_by_area,
    smooth_primary_face_pose,
    step_temporal_monitor,
)
from src.logger import CSV_HEADERS, EvidenceLogger
from src.pcd_pipeline import create_clahe, preprocess_frame_lab_clahe, process_roi_pcd_panel
from src.ui_dashboard import DASHBOARD_HEIGHT, DASHBOARD_WIDTH, render_dashboard


@dataclass(frozen=True)
class DummyLandmark:
    """Representasi ringan titik landmark ternormalisasi untuk pengujian unit."""

    x: float
    y: float


def _build_synthetic_landmarks(
    center_x: float = 0.50,
    center_y: float = 0.50,
    half_w: float = 0.10,
    half_h: float = 0.15,
    nose_x_offset: float = 0.0,
    nose_y_ratio: float = 0.35,
    iris_x_ratio: float = 0.50,
    iris_y_offset_ratio: float = -0.11,
    ear_ratio: float = 0.28,
) -> list[DummyLandmark]:
    """Membangun 478 titik landmark sintetis mencakup mata & pusat iris."""
    points = [DummyLandmark(x=center_x, y=center_y) for _ in range(478)]
    left_x = center_x - half_w
    right_x = center_x + half_w
    eye_y = center_y - half_h * 0.5
    chin_y = center_y + half_h

    points[LEFT_CHEEK_IDX] = DummyLandmark(x=left_x, y=center_y)
    points[RIGHT_CHEEK_IDX] = DummyLandmark(x=right_x, y=center_y)
    points[CHIN_IDX] = DummyLandmark(x=center_x, y=chin_y)

    eye_w = half_w * 0.45
    r_out_x = left_x + 0.02
    r_in_x = r_out_x + eye_w
    l_in_x = right_x - 0.02 - eye_w
    l_out_x = right_x - 0.02

    points[R_EYE_OUTER_IDX] = DummyLandmark(x=r_out_x, y=eye_y)
    points[R_EYE_INNER_IDX] = DummyLandmark(x=r_in_x, y=eye_y)
    points[L_EYE_INNER_IDX] = DummyLandmark(x=l_in_x, y=eye_y)
    points[L_EYE_OUTER_IDX] = DummyLandmark(x=l_out_x, y=eye_y)

    half_open = (ear_ratio * eye_w) / 2.0
    points[R_EYE_TOP_IDX] = DummyLandmark(x=(r_out_x + r_in_x) / 2.0, y=eye_y - half_open)
    points[R_EYE_BOTTOM_IDX] = DummyLandmark(x=(r_out_x + r_in_x) / 2.0, y=eye_y + half_open)
    points[L_EYE_TOP_IDX] = DummyLandmark(x=(l_in_x + l_out_x) / 2.0, y=eye_y - half_open)
    points[L_EYE_BOTTOM_IDX] = DummyLandmark(x=(l_in_x + l_out_x) / 2.0, y=eye_y + half_open)

    points[R_IRIS_CENTER_IDX] = DummyLandmark(
        x=r_out_x + iris_x_ratio * eye_w,
        y=eye_y + iris_y_offset_ratio * eye_w,
    )
    points[L_IRIS_CENTER_IDX] = DummyLandmark(
        x=l_in_x + iris_x_ratio * eye_w,
        y=eye_y + iris_y_offset_ratio * eye_w,
    )

    nose_x = center_x + nose_x_offset
    nose_y = eye_y + nose_y_ratio * (chin_y - eye_y)
    points[NOSE_TIP_IDX] = DummyLandmark(x=nose_x, y=nose_y)
    return points


def _make_face(
    yaw: float = 0.50,
    pitch: float = 0.35,
    gaze_x: float = 0.50,
    gaze_y: float = -0.11,
    ear: float = 0.28,
    area: int = 12_000,
) -> DetectedFace:
    """Membuat objek DetectedFace untuk pengujian matriks aturan kepala & mata."""
    return DetectedFace(
        bbox=(200, 120, 320, 220),
        nose_px=(260, 170),
        iris_px=((240, 150), (280, 150)),
        area=area,
        yaw=yaw,
        pitch=pitch,
        gaze_x=gaze_x,
        gaze_y=gaze_y,
        ear=ear,
    )


def test_pcd_preprocess_dark_bright_and_textured_frames() -> None:
    """Memastikan analisis luminansi dan Global Edge Density bekerja akurat pada 3 kondisi citra."""
    config = DetectorConfig()
    clahe = create_clahe(config)

    dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dark_res = preprocess_frame_lab_clahe(dark_frame, clahe, config)
    assert dark_res.mean_luminance == pytest.approx(0.0)
    assert dark_res.global_edge_density == pytest.approx(0.0)

    white_occlusion = np.full((480, 640, 3), 240, dtype=np.uint8)
    white_res = preprocess_frame_lab_clahe(white_occlusion, clahe, config)
    assert white_res.mean_luminance >= config.min_luminance
    assert white_res.global_edge_density < config.min_global_edge_density

    textured = np.full((480, 640, 3), 90, dtype=np.uint8)
    for row in range(0, 480, 16):
        cv2.line(textured, (0, row), (639, row), (235, 235, 235), 2)
    tex_res = preprocess_frame_lab_clahe(textured, clahe, config)
    assert tex_res.enhanced_bgr.shape == (480, 640, 3)
    assert tex_res.global_edge_density > config.min_global_edge_density


def test_pcd_roi_panel_outputs_four_stages_and_handles_empty_roi() -> None:
    """Memastikan panel ROI menghasilkan 4 citra 110x110 (CLAHE, Gaussian, Otsu, Canny)."""
    config = DetectorConfig()
    clahe = create_clahe(config)

    empty_res = process_roi_pcd_panel(None, clahe, config)
    assert empty_res.roi_edge_density == pytest.approx(0.0)
    assert empty_res.vis_clahe.shape == (110, 110, 3)
    assert empty_res.vis_gaussian.shape == (110, 110, 3)

    roi = np.zeros((140, 140, 3), dtype=np.uint8)
    cv2.circle(roi, (70, 70), 40, (220, 220, 220), -1)
    cv2.rectangle(roi, (45, 50), (95, 90), (40, 40, 40), 2)

    valid_res = process_roi_pcd_panel(roi, clahe, config)
    assert valid_res.roi_edge_density > 0.0
    unique_otsu = set(np.unique(valid_res.vis_otsu).tolist())
    assert unique_otsu.issubset({0, 255})


def test_yaw_pitch_iris_calculation_and_largest_face_sorting() -> None:
    """Wajah peserta utama harus di indeks 0 dan rasio iris dihitung presisi."""
    small_bg_face = _build_synthetic_landmarks(
        center_x=0.20, center_y=0.30, half_w=0.04, half_h=0.05, nose_x_offset=-0.02
    )
    large_main_face = _build_synthetic_landmarks(
        center_x=0.50,
        center_y=0.50,
        half_w=0.15,
        half_h=0.20,
        nose_x_offset=0.0,
        nose_y_ratio=0.35,
        iris_x_ratio=0.65,
        iris_y_offset_ratio=-0.08,
        ear_ratio=0.30,
    )

    yaw, pitch = calculate_yaw_pitch(large_main_face)
    gaze_x, gaze_y, ear = calculate_iris_gaze_and_ear(large_main_face)
    assert yaw == pytest.approx(0.50, abs=1e-3)
    assert pitch == pytest.approx(0.35, abs=1e-3)
    assert gaze_x == pytest.approx(0.65, abs=1e-3)
    assert gaze_y == pytest.approx(-0.08, abs=1e-3)
    assert ear == pytest.approx(0.30, abs=1e-3)

    sorted_faces = extract_faces_sorted_by_area([small_bg_face, large_main_face], 640, 480)
    assert len(sorted_faces) == 2
    assert sorted_faces[0].area > sorted_faces[1].area


def test_five_point_asymmetric_calibration_and_1cm_precision() -> None:
    """
    Menguji kalibrasi 5 titik asimetris penuh (CENTER -> LEFT -> RIGHT -> TOP -> TOUCHPAD):
    - Area layar & keyboard tetap NORMAL (Hijau).
    - Lewat ~1 cm di kiri, kanan, atas layar (baik mata saja maupun kepala) langsung terdeteksi.
    - Tepat di titik akhir bawah touchpad langsung terdeteksi IND-02.
    """
    cfg = DetectorConfig()
    state = MonitoringState()
    state.start_five_point_calibration(now_monotonic=0.0, stage_sec=2.0)

    # Simulasi 5 titik batas fisik asimetris (misal posisi duduk agak ke kanan dari tengah kamera)
    stage_inputs: list[tuple[CalibrationStage, DetectedFace]] = [
        (CalibrationStage.CENTER, _make_face(yaw=0.52, pitch=0.34, gaze_x=0.53, gaze_y=-0.12)),
        (CalibrationStage.LEFT, _make_face(yaw=0.50, pitch=0.34, gaze_x=0.45, gaze_y=-0.12)),
        (CalibrationStage.RIGHT, _make_face(yaw=0.54, pitch=0.34, gaze_x=0.62, gaze_y=-0.12)),
        (CalibrationStage.TOP, _make_face(yaw=0.52, pitch=0.32, gaze_x=0.53, gaze_y=-0.16)),
        (CalibrationStage.TOUCHPAD, _make_face(yaw=0.52, pitch=0.38, gaze_x=0.53, gaze_y=-0.02, ear=0.22)),
    ]

    t = 0.0
    last_step = None
    for idx, (expected_stage, sample_face) in enumerate(stage_inputs):
        assert state.current_calibration_stage == expected_stage
        # Sampel awal (sebelum 0.8s terakhir) diabaikan
        advance_five_point_calibration(state, [sample_face], t + 0.5, cfg)
        # Sampel stabil pada 0.8s terakhir dikoleksi
        advance_five_point_calibration(state, [sample_face], t + 1.5, cfg)
        t += 2.05
        last_step = advance_five_point_calibration(state, [sample_face], t, cfg)
        if idx < 4:
            assert last_step is not None and last_step.calibration_event == "NEXT_STAGE"

    assert last_step is not None and last_step.calibration_event == "COMPLETED"
    assert state.is_calibrated is True
    assert state.safe_bounds is not None

    # 1. Di dalam layar (tengah, dekat tepi kiri, dekat tepi kanan, keyboard tengah) -> NORMAL (Hijau)
    inside_samples = [
        _make_face(yaw=0.52, pitch=0.34, gaze_x=0.53, gaze_y=-0.12),  # Tengah layar
        _make_face(yaw=0.50, pitch=0.34, gaze_x=0.452, gaze_y=-0.12),  # Tepi dalam kiri layar
        _make_face(yaw=0.54, pitch=0.34, gaze_x=0.618, gaze_y=-0.12),  # Tepi dalam kanan layar
        _make_face(yaw=0.52, pitch=0.36, gaze_x=0.53, gaze_y=-0.06, ear=0.28),  # Area keyboard
    ]
    for sample in inside_samples:
        code, _ = evaluate_instant_indication(
            200.0,
            4.0,
            [sample],
            state.baseline_yaw,
            state.baseline_pitch,
            cfg,
            state.baseline_gaze_x,
            state.baseline_gaze_y,
            state.safe_bounds,
        )
        assert code == IndicationCode.NORMAL

    # 2. Keluar ~1 cm di KIRI layar (bola mata saja maupun kepala saja) -> IND-01
    c_eye_left, l_eye_left = evaluate_instant_indication(
        200.0, 4.0, [_make_face(yaw=0.52, pitch=0.34, gaze_x=0.425, gaze_y=-0.12)],
        state.baseline_yaw, state.baseline_pitch, cfg, state.baseline_gaze_x, state.baseline_gaze_y, state.safe_bounds
    )
    c_head_left, l_head_left = evaluate_instant_indication(
        200.0, 4.0, [_make_face(yaw=0.33, pitch=0.34, gaze_x=0.53, gaze_y=-0.12)],
        state.baseline_yaw, state.baseline_pitch, cfg, state.baseline_gaze_x, state.baseline_gaze_y, state.safe_bounds
    )
    assert c_eye_left == IndicationCode.IND_01 and "MELIRIK" in l_eye_left
    assert c_head_left == IndicationCode.IND_01 and "MENOLEH" in l_head_left

    # 3. Keluar ~1 cm di KANAN layar -> IND-01
    c_eye_right, _ = evaluate_instant_indication(
        200.0, 4.0, [_make_face(yaw=0.52, pitch=0.34, gaze_x=0.648, gaze_y=-0.12)],
        state.baseline_yaw, state.baseline_pitch, cfg, state.baseline_gaze_x, state.baseline_gaze_y, state.safe_bounds
    )
    assert c_eye_right == IndicationCode.IND_01

    # 4. Keluar ~1 cm di ATAS bezel layar / kamera -> IND-02
    c_eye_top, l_eye_top = evaluate_instant_indication(
        200.0, 4.0, [_make_face(yaw=0.52, pitch=0.34, gaze_x=0.53, gaze_y=-0.188)],
        state.baseline_yaw, state.baseline_pitch, cfg, state.baseline_gaze_x, state.baseline_gaze_y, state.safe_bounds
    )
    assert c_eye_top == IndicationCode.IND_02 and "ATAS" in l_eye_top

    # 5. Tepat di titik akhir bawah touchpad -> langsung IND-02!
    c_touchpad_end, l_touchpad_end = evaluate_instant_indication(
        200.0, 4.0, [_make_face(yaw=0.52, pitch=0.38, gaze_x=0.53, gaze_y=-0.02, ear=0.22)],
        state.baseline_yaw, state.baseline_pitch, cfg, state.baseline_gaze_x, state.baseline_gaze_y, state.safe_bounds
    )
    assert c_touchpad_end == IndicationCode.IND_02 and "TOUCHPAD" in l_touchpad_end


def test_blink_guard_keeps_head_tracking_active_while_ignoring_iris_spike() -> None:
    """Saat berkedip (EAR < 0.09), lonjakan iris diabaikan tetapi tolehan kepala tetap tertangkap."""
    cfg = DetectorConfig()
    state = MonitoringState()
    state.calibrate(0.50, 0.35, 0.50, -0.11, cfg)

    # Berkedip saat kepala lurus -> tetap NORMAL meski iris spike ekstrem
    c_blink, _ = evaluate_instant_indication(
        200.0, 4.0, [_make_face(yaw=0.50, pitch=0.35, gaze_y=-0.40, ear=0.05)],
        0.50, 0.35, cfg, 0.50, -0.11, state.safe_bounds
    )
    assert c_blink == IndicationCode.NORMAL

    # Menoleh kepala jauh saat mata menyipit -> tetap tertangkap IND-01
    c_squint_turn, _ = evaluate_instant_indication(
        200.0, 4.0, [_make_face(yaw=0.20, pitch=0.35, gaze_x=0.50, ear=0.05)],
        0.50, 0.35, cfg, 0.50, -0.11, state.safe_bounds
    )
    assert c_squint_turn == IndicationCode.IND_01


def test_ema_signal_smoothing_reduces_spike_jitter() -> None:
    """Memastikan filter EMA (alpha=0.45) meredam lonjakan 1 frame."""
    state = MonitoringState()
    f0 = smooth_primary_face_pose([_make_face(yaw=0.50, gaze_x=0.50)], state, alpha=0.45)[0]
    assert f0.yaw == pytest.approx(0.50)

    f1 = smooth_primary_face_pose([_make_face(yaw=0.70, gaze_x=0.60)], state, alpha=0.45)[0]
    assert f1.yaw == pytest.approx(0.59)
    assert f1.gaze_x == pytest.approx(0.545)


def test_instant_indication_priority_and_bidirectional_pitch() -> None:
    """Memverifikasi hierarki prioritas IND-05 > IND-03 > IND-04 > IND-01 > IND-02 > NORMAL."""
    cfg = DetectorConfig()

    code, label = evaluate_instant_indication(20.0, 5.0, [_make_face(), _make_face()], 0.5, 0.35, cfg)
    assert code == IndicationCode.IND_05 and "GELAP" in label

    code, label = evaluate_instant_indication(210.0, 0.2, [], 0.5, 0.35, cfg)
    assert code == IndicationCode.IND_05 and "OKLUSI" in label

    code, _ = evaluate_instant_indication(120.0, 6.0, [_make_face(), _make_face()], 0.5, 0.35, cfg)
    assert code == IndicationCode.IND_03

    code, _ = evaluate_instant_indication(120.0, 6.0, [], 0.5, 0.35, cfg)
    assert code == IndicationCode.IND_04

    code_down, label_down = evaluate_instant_indication(
        120.0, 6.0, [_make_face(pitch=0.52)], 0.50, 0.35, cfg
    )
    code_up, label_up = evaluate_instant_indication(
        120.0, 6.0, [_make_face(pitch=0.20)], 0.50, 0.35, cfg
    )
    assert code_down == IndicationCode.IND_02 and "MENUNDUK" in label_down
    assert code_up == IndicationCode.IND_02 and "MENDONGAK" in label_up


def test_calibration_gate_grace_period_and_cooldown() -> None:
    """Menguji Calibration Gate, durasi lirikan 2.5s, grace period 0.3s, dan cooldown 5.0s."""
    cfg = DetectorConfig()
    state = MonitoringState()

    uncal_step = step_temporal_monitor(state, IndicationCode.IND_03, ">1 ORANG", 10.0, cfg)
    assert uncal_step.banner_state == BannerState.UNCALIBRATED
    assert state.trust_score == pytest.approx(100.0)

    state.calibrate(0.50, 0.35, 0.50, -0.11, cfg)

    step_temporal_monitor(state, IndicationCode.IND_01, "MELIRIK MATA KE KIRI", 10.0, cfg)
    step_temporal_monitor(state, IndicationCode.NORMAL, "FOKUS", 12.2, cfg)
    s_gaze_alert = step_temporal_monitor(
        state, IndicationCode.IND_01, "MELIRIK MATA KE KIRI", 12.55, cfg
    )
    assert s_gaze_alert.banner_state == BannerState.ALERT
    assert s_gaze_alert.should_log_evidence is True
    assert state.trust_score == pytest.approx(95.0)

    s_cool = step_temporal_monitor(
        state, IndicationCode.IND_01, "MELIRIK MATA KE KIRI", 15.0, cfg
    )
    assert s_cool.should_log_evidence is False
    assert state.trust_score == pytest.approx(95.0)


def test_trust_score_floor_and_integrity_levels() -> None:
    """Memastikan Trust Score tidak pernah negatif dan pemetaan IntegrityLevel akurat."""
    assert resolve_integrity_level(100.0) == IntegrityLevel.AMAN
    assert resolve_integrity_level(75.0) == IntegrityLevel.WASPADA
    assert resolve_integrity_level(45.0) == IntegrityLevel.TINJAUAN_PENGAWAS

    cfg = DetectorConfig()
    state = MonitoringState(trust_score=3.0)
    state.calibrate(0.50, 0.35, config=cfg)
    step_temporal_monitor(state, IndicationCode.IND_03, ">1 ORANG", 0.0, cfg)
    step_temporal_monitor(state, IndicationCode.IND_03, ">1 ORANG", 1.2, cfg)
    assert state.trust_score == pytest.approx(0.0)


def test_evidence_logger_and_five_stage_dashboard_render(tmp_path: Path) -> None:
    """Menguji penyimpanan bukti .jpg/.csv serta render kelima panduan panah kalibrasi."""
    cfg = DetectorConfig()
    clahe = create_clahe(cfg)
    raw = np.full((480, 640, 3), 110, dtype=np.uint8)
    frame_pcd = preprocess_frame_lab_clahe(raw, clahe, cfg)
    roi_pcd = process_roi_pcd_panel(raw[100:220, 200:320], clahe, cfg)

    state = MonitoringState()
    state.start_five_point_calibration(0.0, 2.0)

    for stage_idx in range(5):
        state.calibration_stage_idx = stage_idx
        cal_step = advance_five_point_calibration(state, [_make_face()], 0.2, cfg)
        assert cal_step is not None
        dashboard = render_dashboard(
            raw_frame_bgr=raw,
            faces=[_make_face()],
            state=state,
            step_result=cal_step,
            frame_pcd=frame_pcd,
            roi_pcd=roi_pcd,
            min_luminance=cfg.min_luminance,
        )
        assert dashboard.shape == (DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3)

    evidence_dir = tmp_path / "bukti_indikasi"
    log_csv = tmp_path / "log_indikasi_ujian.csv"
    logger = EvidenceLogger(evidence_dir, log_csv)
    saved_jpg = logger.record_indication(
        annotated_frame=dashboard[:, :640],
        code=IndicationCode.IND_02,
        behavior_label="MELIRIK BAWAH (LEWAT TOUCHPAD)",
        duration_sec=2.6,
        mean_luminance=110.0,
        trust_score=95.0,
        flag_number=1,
        timestamp=datetime(2026, 10, 30, 23, 25, 0),
    )
    assert saved_jpg.exists()
    with open(log_csv, mode="r", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))
    assert rows[0] == CSV_HEADERS
