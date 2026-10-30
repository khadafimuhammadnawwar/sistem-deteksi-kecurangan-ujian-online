"""Unit test komprehensif untuk pipeline PCD, geometri Yaw/Pitch, aturan temporal, UI, dan logger."""

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
    LEFT_CHEEK_IDX,
    LEFT_EYE_OUTER_IDX,
    NOSE_TIP_IDX,
    RIGHT_CHEEK_IDX,
    RIGHT_EYE_OUTER_IDX,
    DetectedFace,
    calculate_yaw_pitch,
    evaluate_instant_indication,
    extract_faces_sorted_by_area,
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
    nose_y_ratio: float = 0.38,
) -> list[DummyLandmark]:
    """Membangun 478 titik landmark sintetis dengan parameter geometri terkontrol."""
    points = [DummyLandmark(x=center_x, y=center_y) for _ in range(478)]
    left_x = center_x - half_w
    right_x = center_x + half_w
    eye_y = center_y - half_h * 0.5
    chin_y = center_y + half_h

    points[LEFT_CHEEK_IDX] = DummyLandmark(x=left_x, y=center_y)
    points[RIGHT_CHEEK_IDX] = DummyLandmark(x=right_x, y=center_y)
    points[LEFT_EYE_OUTER_IDX] = DummyLandmark(x=left_x + 0.02, y=eye_y)
    points[RIGHT_EYE_OUTER_IDX] = DummyLandmark(x=right_x - 0.02, y=eye_y)
    points[CHIN_IDX] = DummyLandmark(x=center_x, y=chin_y)

    nose_x = center_x + nose_x_offset
    nose_y = eye_y + nose_y_ratio * (chin_y - eye_y)
    points[NOSE_TIP_IDX] = DummyLandmark(x=nose_x, y=nose_y)
    return points


def _make_face(yaw: float = 0.50, pitch: float = 0.38, area: int = 12_000) -> DetectedFace:
    """Membuat objek DetectedFace untuk pengujian matriks aturan."""
    return DetectedFace(
        bbox=(200, 120, 320, 220),
        nose_px=(260, 170),
        area=area,
        yaw=yaw,
        pitch=pitch,
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


def test_yaw_pitch_calculation_and_largest_face_sorting() -> None:
    """Wajah peserta utama di depan layar harus selalu berada di indeks 0 meski urutan input terbalik."""
    small_bg_face = _build_synthetic_landmarks(
        center_x=0.20, center_y=0.30, half_w=0.04, half_h=0.05, nose_x_offset=-0.02
    )
    large_main_face = _build_synthetic_landmarks(
        center_x=0.50, center_y=0.50, half_w=0.15, half_h=0.20, nose_x_offset=0.0, nose_y_ratio=0.38
    )

    yaw, pitch = calculate_yaw_pitch(large_main_face)
    assert yaw == pytest.approx(0.50, abs=1e-3)
    assert pitch == pytest.approx(0.38, abs=1e-3)

    sorted_faces = extract_faces_sorted_by_area([small_bg_face, large_main_face], 640, 480)
    assert len(sorted_faces) == 2
    assert sorted_faces[0].area > sorted_faces[1].area
    assert sorted_faces[0].yaw == pytest.approx(0.50, abs=1e-3)


def test_instant_indication_priority_and_bidirectional_pitch() -> None:
    """Memverifikasi hierarki prioritas IND-05 > IND-03 > IND-04 > IND-01 > IND-02 > NORMAL."""
    cfg = DetectorConfig()

    # IND-05 Gelap mengalahkan segalanya
    code, label = evaluate_instant_indication(20.0, 5.0, [_make_face(), _make_face()], 0.5, 0.38, cfg)
    assert code == IndicationCode.IND_05
    assert "GELAP" in label

    # IND-05 Oklusi benda terang polos (misal kertas putih menutup kamera)
    code, label = evaluate_instant_indication(210.0, 0.2, [], 0.5, 0.38, cfg)
    assert code == IndicationCode.IND_05
    assert "OKLUSI" in label

    # IND-03 >1 orang
    code, _ = evaluate_instant_indication(120.0, 6.0, [_make_face(), _make_face()], 0.5, 0.38, cfg)
    assert code == IndicationCode.IND_03

    # IND-04 Wajah hilang di ruangan bertekstur normal
    code, _ = evaluate_instant_indication(120.0, 6.0, [], 0.5, 0.38, cfg)
    assert code == IndicationCode.IND_04

    # IND-01 Menoleh kiri & kanan
    code_left, label_left = evaluate_instant_indication(
        120.0, 6.0, [_make_face(yaw=0.30)], 0.50, 0.38, cfg
    )
    code_right, label_right = evaluate_instant_indication(
        120.0, 6.0, [_make_face(yaw=0.70)], 0.50, 0.38, cfg
    )
    assert code_left == IndicationCode.IND_01 and "KIRI" in label_left
    assert code_right == IndicationCode.IND_01 and "KANAN" in label_right

    # IND-02 Menunduk & Mendongak (Mode Teori)
    code_down, label_down = evaluate_instant_indication(
        120.0, 6.0, [_make_face(pitch=0.50)], 0.50, 0.38, cfg
    )
    code_up, label_up = evaluate_instant_indication(
        120.0, 6.0, [_make_face(pitch=0.25)], 0.50, 0.38, cfg
    )
    assert code_down == IndicationCode.IND_02 and "MENUNDUK" in label_down
    assert code_up == IndicationCode.IND_02 and "MENDONGAK" in label_up

    # NORMAL
    code_norm, _ = evaluate_instant_indication(120.0, 6.0, [_make_face()], 0.50, 0.38, cfg)
    assert code_norm == IndicationCode.NORMAL


def test_calibration_gate_grace_period_and_cooldown() -> None:
    """Menguji Calibration Gate, anti-flicker grace period 0.3s, cooldown 5.0s, dan batas skor 0%."""
    cfg = DetectorConfig()
    state = MonitoringState()

    # Sebelum tekan 'C', pelanggaran tidak boleh memotong Trust Score
    uncal_step = step_temporal_monitor(state, IndicationCode.IND_03, ">1 ORANG", 10.0, cfg)
    assert uncal_step.banner_state == BannerState.UNCALIBRATED
    assert state.trust_score == pytest.approx(100.0)
    assert not uncal_step.should_log_evidence

    state.calibrate(0.50, 0.38)

    # Mulai menoleh pada t=10.0s (ambang IND-01 = 2.5s)
    s1 = step_temporal_monitor(state, IndicationCode.IND_01, "MENOLEH KE KIRI", 10.0, cfg)
    assert s1.banner_state == BannerState.WARNING

    # Flicker NORMAL singkat 0.15s (< 0.3s grace period) tidak boleh mereset timer awal (10.0s)
    step_temporal_monitor(state, IndicationCode.NORMAL, "FOKUS", 12.0, cfg)
    s_after_flicker = step_temporal_monitor(
        state, IndicationCode.IND_01, "MENOLEH KE KIRI", 12.15, cfg
    )
    assert s_after_flicker.elapsed_sec == pytest.approx(2.15)

    # Melewati 2.5s pada t=12.6s -> Penalti pertama (-5%) & simpan bukti
    s_alert = step_temporal_monitor(state, IndicationCode.IND_01, "MENOLEH KE KIRI", 12.6, cfg)
    assert s_alert.banner_state == BannerState.ALERT
    assert s_alert.should_log_evidence is True
    assert state.trust_score == pytest.approx(95.0)
    assert state.total_flags == 1

    # Dalam masa cooldown (< 5.0s sejak 12.6s) -> tetap ALERT tapi tidak potong skor lagi
    s_cooldown = step_temporal_monitor(state, IndicationCode.IND_01, "MENOLEH KE KIRI", 15.0, cfg)
    assert s_cooldown.banner_state == BannerState.ALERT
    assert s_cooldown.should_log_evidence is False
    assert state.trust_score == pytest.approx(95.0)

    # Setelah cooldown lewat (t=17.7s) -> potong skor lagi
    s_second = step_temporal_monitor(state, IndicationCode.IND_01, "MENOLEH KE KIRI", 17.7, cfg)
    assert s_second.should_log_evidence is True
    assert state.trust_score == pytest.approx(90.0)

    # Kembali fokus >= 0.3s -> timer benar-benar reset ke NORMAL
    step_temporal_monitor(state, IndicationCode.NORMAL, "FOKUS", 18.0, cfg)
    step_temporal_monitor(state, IndicationCode.NORMAL, "FOKUS", 18.4, cfg)
    assert state.active_code == IndicationCode.NORMAL
    assert state.behavior_start_time is None


def test_trust_score_floor_and_integrity_levels() -> None:
    """Memastikan Trust Score tidak pernah negatif dan pemetaan IntegrityLevel akurat."""
    assert resolve_integrity_level(100.0) == IntegrityLevel.AMAN
    assert resolve_integrity_level(80.0) == IntegrityLevel.AMAN
    assert resolve_integrity_level(75.0) == IntegrityLevel.WASPADA
    assert resolve_integrity_level(50.0) == IntegrityLevel.WASPADA
    assert resolve_integrity_level(45.0) == IntegrityLevel.TINJAUAN_PENGAWAS

    cfg = DetectorConfig()
    state = MonitoringState(trust_score=3.0)
    state.calibrate(0.50, 0.38)
    step_temporal_monitor(state, IndicationCode.IND_03, ">1 ORANG", 0.0, cfg)
    step_temporal_monitor(state, IndicationCode.IND_03, ">1 ORANG", 1.2, cfg)
    assert state.trust_score == pytest.approx(0.0)


def test_evidence_logger_and_dashboard_render(tmp_path: Path) -> None:
    """Menguji penyimpanan screenshot .jpg, penulisan log .csv, dan ukuran kanvas dashboard."""
    cfg = DetectorConfig()
    clahe = create_clahe(cfg)
    raw = np.full((480, 640, 3), 110, dtype=np.uint8)
    frame_pcd = preprocess_frame_lab_clahe(raw, clahe, cfg)
    roi_pcd = process_roi_pcd_panel(raw[100:220, 200:320], clahe, cfg)

    state = MonitoringState()
    state.calibrate(0.50, 0.38)
    step_res = step_temporal_monitor(state, IndicationCode.NORMAL, "PESERTA FOKUS UJIAN", 1.0, cfg)

    dashboard = render_dashboard(
        raw_frame_bgr=raw,
        faces=[_make_face()],
        state=state,
        step_result=step_res,
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
        behavior_label="MENOLEH KE KIRI",
        duration_sec=2.7,
        mean_luminance=110.0,
        trust_score=95.0,
        flag_number=1,
        timestamp=datetime(2026, 10, 30, 20, 55, 0),
    )

    assert saved_jpg.exists()
    with open(log_csv, mode="r", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))
    assert rows[0] == CSV_HEADERS
    assert rows[1][1] == "IND-01"
    assert rows[1][5] == "95.0"
