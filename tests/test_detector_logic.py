"""Unit test komprehensif untuk pipeline PCD, kalibrasi countdown, zona aman layar VOR, EMA, UI, dan logger."""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import (
    BannerState,
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
    advance_calibration_countdown,
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


def test_screen_safe_zone_vor_compensation_and_offscreen_gaze() -> None:
    """
    Memastikan membaca seluruh sudut layar laptop (kiri, kanan, atas, bawah + efek VOR dagu turun)
    tetap berstatus NORMAL (Hijau), dan hanya melirik jauh keluar layar yang memicu indikasi.
    """
    cfg = DetectorConfig()
    base_yaw, base_pitch, base_gx, base_gy = 0.46, 0.35, 0.48, -0.12

    # Sampel nyata sudut layar laptop (tengah, tepi kiri, tepi kanan, atas layar dgn dagu turun VOR, bawah layar)
    screen_samples = [
        _make_face(yaw=0.403, pitch=0.326, gaze_x=0.503, gaze_y=-0.118, ear=0.45),
        _make_face(yaw=0.433, pitch=0.382, gaze_x=0.498, gaze_y=-0.171, ear=0.46),
        _make_face(yaw=0.454, pitch=0.347, gaze_x=0.437, gaze_y=-0.138, ear=0.50),
        _make_face(yaw=0.539, pitch=0.325, gaze_x=0.412, gaze_y=-0.092, ear=0.46),
        _make_face(yaw=0.500, pitch=0.360, gaze_x=0.575, gaze_y=-0.100, ear=0.44),
    ]
    for sample in screen_samples:
        code, _ = evaluate_instant_indication(
            205.0, 4.0, [sample], base_yaw, base_pitch, cfg, base_gx, base_gy
        )
        assert code == IndicationCode.NORMAL

    # Melirik tajam keluar sisi kiri & kanan layar
    c_left, lbl_left = evaluate_instant_indication(
        205.0, 4.0, [_make_face(gaze_x=0.30)], 0.50, 0.35, cfg, 0.50, -0.11
    )
    c_right, lbl_right = evaluate_instant_indication(
        205.0, 4.0, [_make_face(gaze_x=0.70)], 0.50, 0.35, cfg, 0.50, -0.11
    )
    assert c_left == IndicationCode.IND_01 and lbl_left == "MELIRIK MATA KE KIRI"
    assert c_right == IndicationCode.IND_01 and lbl_right == "MELIRIK MATA KE KANAN"

    # Melirik tajam ke langit-langit atas & ke bawah luar layar saat kepala lurus
    c_up, lbl_up = evaluate_instant_indication(
        205.0, 4.0, [_make_face(gaze_y=-0.28)], 0.50, 0.35, cfg, 0.50, -0.11
    )
    c_down, lbl_down = evaluate_instant_indication(
        205.0, 4.0, [_make_face(gaze_y=0.06)], 0.50, 0.35, cfg, 0.50, -0.11
    )
    assert c_up == IndicationCode.IND_02 and lbl_up == "MELIRIK MATA KE ATAS"
    assert c_down == IndicationCode.IND_02 and lbl_down == "MELIRIK MATA KE BAWAH"

    # Kedipan mata (EAR < 0.15) diabaikan
    c_blink, _ = evaluate_instant_indication(
        205.0, 4.0, [_make_face(gaze_y=-0.30, ear=0.09)], 0.50, 0.35, cfg, 0.50, -0.11
    )
    assert c_blink == IndicationCode.NORMAL


def test_countdown_calibration_prevents_keyboard_gaze_trap() -> None:
    """
    Saat baru tekan tombol C/SPASI (t=0.0..1.0s), peserta masih melihat keyboard (gaze_y=+0.02);
    sistem mengabaikan frame awal dan hanya merata-ratakan 0.8s terakhir saat mata sudah di layar (-0.12).
    """
    cfg = DetectorConfig()
    state = MonitoringState()
    state.start_calibration_countdown(now_monotonic=10.0, duration_sec=cfg.calibration_countdown_sec)

    # t=10.2s (sisa 1.8s > 0.8s): mata masih di keyboard -> belum dikoleksi
    r1 = advance_calibration_countdown(state, [_make_face(gaze_y=0.03)], 10.2, cfg)
    assert r1 is not None and r1.banner_state == BannerState.CALIBRATING
    assert len(state.calibration_samples) == 0

    # t=11.5s & t=12.0s (sisa <= 0.8s): mata sudah fokus ke tengah layar (-0.12)
    r2 = advance_calibration_countdown(state, [_make_face(gaze_y=-0.11)], 11.5, cfg)
    assert r2 is not None and r2.banner_state == BannerState.CALIBRATING
    r_done = advance_calibration_countdown(state, [_make_face(gaze_y=-0.13)], 12.05, cfg)

    assert r_done is not None and r_done.banner_state == BannerState.SAFE
    assert state.is_calibrated is True
    assert state.baseline_gaze_y == pytest.approx(-0.12, abs=1e-3)


def test_ema_signal_smoothing_reduces_spike_jitter() -> None:
    """Memastikan filter EMA meredam lonjakan 1 frame."""
    state = MonitoringState()
    f0 = smooth_primary_face_pose([_make_face(yaw=0.50, gaze_x=0.50)], state, alpha=0.35)[0]
    assert f0.yaw == pytest.approx(0.50)

    f1 = smooth_primary_face_pose([_make_face(yaw=0.70, gaze_x=0.60)], state, alpha=0.35)[0]
    assert f1.yaw == pytest.approx(0.57)
    assert f1.gaze_x == pytest.approx(0.535)


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
        120.0, 6.0, [_make_face(pitch=0.50)], 0.50, 0.35, cfg
    )
    code_up, label_up = evaluate_instant_indication(
        120.0, 6.0, [_make_face(pitch=0.20)], 0.50, 0.35, cfg
    )
    assert code_down == IndicationCode.IND_02 and "MENUNDUK" in label_down
    assert code_up == IndicationCode.IND_02 and "MENDONGAK" in label_up


def test_calibration_gate_grace_period_and_cooldown() -> None:
    """Menguji Calibration Gate, durasi lirikan mata 2.5s, grace period 0.3s, dan cooldown 5.0s."""
    cfg = DetectorConfig()
    state = MonitoringState()

    uncal_step = step_temporal_monitor(state, IndicationCode.IND_03, ">1 ORANG", 10.0, cfg)
    assert uncal_step.banner_state == BannerState.UNCALIBRATED
    assert state.trust_score == pytest.approx(100.0)

    state.calibrate(0.50, 0.35, 0.50, -0.11)

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
    state.calibrate(0.50, 0.35)
    step_temporal_monitor(state, IndicationCode.IND_03, ">1 ORANG", 0.0, cfg)
    step_temporal_monitor(state, IndicationCode.IND_03, ">1 ORANG", 1.2, cfg)
    assert state.trust_score == pytest.approx(0.0)


def test_evidence_logger_and_dashboard_render(tmp_path: Path) -> None:
    """Menguji penyimpanan screenshot .jpg, penulisan log .csv, dan render target kalibrasi."""
    cfg = DetectorConfig()
    clahe = create_clahe(cfg)
    raw = np.full((480, 640, 3), 110, dtype=np.uint8)
    frame_pcd = preprocess_frame_lab_clahe(raw, clahe, cfg)
    roi_pcd = process_roi_pcd_panel(raw[100:220, 200:320], clahe, cfg)

    state = MonitoringState()
    state.start_calibration_countdown(0.0, 2.0)
    cal_step = advance_calibration_countdown(state, [_make_face()], 0.5, cfg)
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
        code=IndicationCode.IND_01,
        behavior_label="MELIRIK MATA KE KIRI",
        duration_sec=2.6,
        mean_luminance=110.0,
        trust_score=95.0,
        flag_number=1,
        timestamp=datetime(2026, 10, 30, 22, 10, 0),
    )
    assert saved_jpg.exists()
    with open(log_csv, mode="r", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))
    assert rows[0] == CSV_HEADERS
