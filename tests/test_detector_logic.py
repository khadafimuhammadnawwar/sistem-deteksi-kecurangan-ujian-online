"""Unit test untuk PCD, 3D Hardware-Locked Ray-Casting (1cm/21cm), 5-Layer Anti-Spoof, UI, & CSV."""

import csv
from dataclasses import dataclass
from datetime import datetime
import math
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
    advance_lens_calibration,
    calculate_iris_gaze_and_ear,
    evaluate_instant_indication,
    extract_faces_sorted_by_area,
    project_gaze_3d,
    smooth_primary_face_pose,
    step_temporal_monitor,
)
from src.logger import CSV_HEADERS, EvidenceLogger
from src.pcd_pipeline import (
    create_clahe,
    preprocess_frame_lab_clahe,
    process_roi_pcd_panel,
)
from src.ui_dashboard import DASHBOARD_HEIGHT, DASHBOARD_WIDTH, render_dashboard


@dataclass(frozen=True)
class DummyLandmark:
    """Representasi ringan titik landmark ternormalisasi untuk pengujian unit."""

    x: float
    y: float


def _build_synthetic_landmarks(
    center_x: float = 0.50,
    center_y: float = 0.50,
    half_w: float = 0.12,
    half_h: float = 0.16,
    nose_x_offset: float = 0.0,
    nose_y_ratio: float = 0.35,
    iris_x_ratio: float = 0.520,
    iris_y_offset_ratio: float = -0.130,
    ear_ratio: float = 0.30,
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
    points[R_EYE_TOP_IDX] = DummyLandmark(
        x=(r_out_x + r_in_x) / 2.0, y=eye_y - half_open
    )
    points[R_EYE_BOTTOM_IDX] = DummyLandmark(
        x=(r_out_x + r_in_x) / 2.0, y=eye_y + half_open
    )
    points[L_EYE_TOP_IDX] = DummyLandmark(
        x=(l_in_x + l_out_x) / 2.0, y=eye_y - half_open
    )
    points[L_EYE_BOTTOM_IDX] = DummyLandmark(
        x=(l_in_x + l_out_x) / 2.0, y=eye_y + half_open
    )

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


def _make_face_3d(
    eye_x_cm: float = 0.0,
    eye_y_cm: float = 0.5,
    eye_z_cm: float = 42.0,
    head_yaw_deg: float = 0.0,
    head_pitch_deg: float = 0.0,
    gaze_x: float = 0.520,
    gaze_y: float = -0.130,
    ear: float = 0.32,
    area: int = 14_000,
) -> DetectedFace:
    """Membuat objek DetectedFace metrik 3D untuk pengujian presisi sentimeter."""
    return DetectedFace(
        bbox=(200, 120, 340, 220),
        nose_px=(270, 170),
        iris_px=((245, 150), (295, 150)),
        area=area,
        eye_x_cm=eye_x_cm,
        eye_y_cm=eye_y_cm,
        eye_z_cm=eye_z_cm,
        head_yaw_deg=head_yaw_deg,
        head_pitch_deg=head_pitch_deg,
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


def test_3d_matrix_extraction_and_largest_face_sorting() -> None:
    """Mengekstrak matriks 4x4 MediaPipe ke cm/derajat akurat & mengurutkan Wajah #1 terbesar."""
    small_bg_face = _build_synthetic_landmarks(
        center_x=0.20, center_y=0.30, half_w=0.04, half_h=0.05
    )
    large_main_face = _build_synthetic_landmarks(
        center_x=0.50,
        center_y=0.50,
        half_w=0.15,
        half_h=0.20,
        iris_x_ratio=0.545,
        iris_y_offset_ratio=-0.115,
        ear_ratio=0.34,
    )
    gaze_x, gaze_y, ear = calculate_iris_gaze_and_ear(large_main_face)
    assert gaze_x == pytest.approx(0.545, abs=1e-3)
    assert gaze_y == pytest.approx(-0.115, abs=1e-3)
    assert ear == pytest.approx(0.34, abs=1e-3)

    yaw_rad = math.radians(15.0)
    mat_main = np.eye(4, dtype=np.float64)
    mat_main[0, 2] = math.sin(yaw_rad)
    mat_main[2, 2] = math.cos(yaw_rad)
    mat_main[0, 3] = 2.5
    mat_main[1, 3] = -1.2
    mat_main[2, 3] = -44.0

    mat_bg = np.eye(4, dtype=np.float64)
    mat_bg[2, 3] = -80.0

    sorted_faces = extract_faces_sorted_by_area(
        [small_bg_face, large_main_face],
        640,
        480,
        transformation_matrices=[mat_bg, mat_main],
    )
    assert len(sorted_faces) == 2
    primary = sorted_faces[0]
    assert primary.area > sorted_faces[1].area
    assert primary.eye_x_cm == pytest.approx(2.5)
    assert primary.eye_y_cm == pytest.approx(1.2)
    assert primary.eye_z_cm == pytest.approx(44.0)
    assert primary.head_yaw_deg == pytest.approx(15.0, abs=1e-2)


def test_hardware_locked_3d_ray_casting_1cm_screen_and_21cm_touchpad() -> None:
    """
    Memverifikasi geometri fisik terkunci Lenovo IdeaPad Slim 5 14IMH9:
    - Seluruh layar (30.16 x 18.85 cm) + keyboard s/d <21.0 cm touchpad -> NORMAL (Hijau).
    - >1.0 cm di luar Kiri (-16.08 cm) dan Kanan (+16.08 cm) -> IND-01 secara simetris.
    - >1.0 cm di atas bezel kamera (Y < -1.0 cm) -> IND-02.
    - Tepat/lewat titik akhir touchpad (deck_z >= 21.0 cm) -> IND-02.
    """
    cfg = (
        DetectorConfig()
    )  # half_safe_width_cm = 16.08 cm, touchpad_end_depth_cm = 21.0 cm

    # 1. Di dalam layar (Tengah, Tepi Kiri -14.8 cm, Tepi Kanan +14.8 cm) & Keyboard (12 cm) -> NORMAL
    safe_faces = [
        _make_face_3d(head_pitch_deg=12.0),  # Tengah layar (Y ~ +9.4 cm)
        _make_face_3d(
            gaze_x=0.520 - (19.0 / cfg.eye_yaw_gain_deg), head_pitch_deg=10.0
        ),  # Tepi dalam kiri (-14.5 cm)
        _make_face_3d(
            gaze_x=0.520 + (19.0 / cfg.eye_yaw_gain_deg), head_pitch_deg=10.0
        ),  # Tepi dalam kanan (+14.5 cm)
        _make_face_3d(
            head_pitch_deg=5.0, gaze_y=-0.130 + (26.0 / cfg.eye_pitch_down_gain_deg)
        ),  # Tengah keyboard (~10 cm)
        _make_face_3d(
            head_pitch_deg=6.0, gaze_y=-0.130 + (34.0 / cfg.eye_pitch_down_gain_deg)
        ),  # Dalam touchpad (~19.5 cm)
    ]
    for f_ok in safe_faces:
        code, _ = evaluate_instant_indication(180.0, 5.0, [f_ok], cfg)
        assert code == IndicationCode.NORMAL

    # 2. >1 cm di luar KIRI layar (X < -16.08 cm): baik lirikan mata maupun tolehan kepala
    f_eye_left = _make_face_3d(gaze_x=0.520 - (22.0 / cfg.eye_yaw_gain_deg))  # X_hit ~ -17.0 cm
    f_head_left = _make_face_3d(head_yaw_deg=-22.0)
    c_el, l_el = evaluate_instant_indication(180.0, 5.0, [f_eye_left], cfg)
    c_hl, l_hl = evaluate_instant_indication(180.0, 5.0, [f_head_left], cfg)
    assert c_el == IndicationCode.IND_01 and "MELIRIK MATA KE KIRI" == l_el
    assert c_hl == IndicationCode.IND_01 and "MENOLEH KE KIRI" == l_hl

    # 3. >1 cm di luar KANAN layar (X > +16.08 cm): simetris sempurna dengan kiri!
    f_eye_right = _make_face_3d(gaze_x=0.520 + (22.0 / cfg.eye_yaw_gain_deg))  # X_hit ~ +17.0 cm
    c_er, l_er = evaluate_instant_indication(180.0, 5.0, [f_eye_right], cfg)
    assert c_er == IndicationCode.IND_01 and "MELIRIK MATA KE KANAN" == l_er

    # 4. Invarian Pergeseran Kursi (+5 cm ke kanan): batas fisik layar tetap simetris terhadap laptop!
    f_shifted_center = _make_face_3d(
        eye_x_cm=5.0, head_yaw_deg=-6.8
    )  # Menatap tengah layar X ~ 0 cm
    assert (
        evaluate_instant_indication(180.0, 5.0, [f_shifted_center], cfg)[0]
        == IndicationCode.NORMAL
    )

    # 5. >1 cm di ATAS bezel kamera (Y < -1.0 cm): baik dari kepala lurus maupun kepala menghadap tengah layar (+12 deg)
    f_eye_up = _make_face_3d(
        eye_y_cm=0.5, gaze_y=-0.130 - (3.0 / cfg.eye_pitch_up_gain_deg)
    )  # Y_hit ~ -1.7 cm
    c_up, l_up = evaluate_instant_indication(180.0, 5.0, [f_eye_up], cfg)
    assert c_up == IndicationCode.IND_02 and "ATAS" in l_up

    f_center_head_eye_up = _make_face_3d(
        eye_y_cm=0.5,
        eye_z_cm=42.0,
        head_pitch_deg=12.0,
        gaze_y=-0.130 - (14.5 / cfg.eye_pitch_up_gain_deg),  # Lirikan mata murni ke >1 cm di atas bezel saat kepala diam di tengah layar
    )
    c_cup, l_cup = evaluate_instant_indication(180.0, 5.0, [f_center_head_eye_up], cfg)
    assert c_cup == IndicationCode.IND_02 and l_cup == "MELIRIK MATA KE ATAS"

    # 6. Kontinuitas garis engsel (Y_scr tepat di hinge_y_cm -> deck_z_cm mendekati 0.0 cm) pada 90 deg & 102 deg
    hinge_pitch_deg = math.degrees(math.atan2(cfg.hinge_y_cm - 1.0 + 0.01, 42.0))
    f_at_hinge = _make_face_3d(
        eye_y_cm=1.0, eye_z_cm=42.0, head_pitch_deg=hinge_pitch_deg
    )
    hit_hinge_90 = project_gaze_3d(f_at_hinge, cfg)
    hit_hinge_102 = project_gaze_3d(f_at_hinge, DetectorConfig(lid_angle_deg=102.0))
    assert (
        hit_hinge_90.on_keyboard_deck is True
        and hit_hinge_90.deck_z_cm == pytest.approx(0.0, abs=0.05)
    )
    assert (
        hit_hinge_102.on_keyboard_deck is True
        and hit_hinge_102.deck_z_cm == pytest.approx(0.0, abs=0.05)
    )

    # 7. Tepat mencapai / melewati titik akhir bawah touchpad (>= 21.0 cm) pada sudut 90 deg & miring 102 deg
    f_touchpad_end_eye = _make_face_3d(
        eye_y_cm=1.0,
        eye_z_cm=42.0,
        head_pitch_deg=4.0,
        gaze_y=-0.130 + (39.5 / cfg.eye_pitch_down_gain_deg),
    )
    hit_tp = project_gaze_3d(f_touchpad_end_eye, cfg)
    assert hit_tp.on_keyboard_deck is True
    assert hit_tp.deck_z_cm >= 21.0
    c_tp, l_tp = evaluate_instant_indication(180.0, 5.0, [f_touchpad_end_eye], cfg)
    assert c_tp == IndicationCode.IND_02 and "TOUCHPAD" in l_tp

    cfg_tilted = DetectorConfig(lid_angle_deg=102.0)
    alpha_rad = math.radians(12.0)
    tp_y = cfg_tilted.hinge_y_cm + 21.0 * math.sin(alpha_rad)
    tp_z = 21.0 * math.cos(alpha_rad)
    exact_tp_pitch = math.degrees(math.atan2(tp_y - 9.38, 46.26 - tp_z))
    f_tilted_tp = _make_face_3d(
        eye_y_cm=9.38, eye_z_cm=46.26, head_pitch_deg=exact_tp_pitch
    )
    hit_tilted_tp = project_gaze_3d(f_tilted_tp, cfg_tilted)
    assert hit_tilted_tp.deck_z_cm == pytest.approx(21.0, abs=0.05)


def test_anti_spoof_lens_calibration_accepts_honest_and_rejects_cheating() -> None:
    """
    Memastikan 5-Layer Anti-Spoof Gate menolak peserta yang mencoba mengakali kalibrasi
    (melirik jauh, menoleh, menutup mata, goyang cepat, jarak ekstrem) dan menerima tatapan jujur.
    """
    cfg = DetectorConfig()

    # Skenario Curang 1: Peserta melirik jauh ke kiri saat kalibrasi lensa -> WAJIB DITOLAK!
    spoof_state = MonitoringState()
    spoof_state.start_lens_calibration(now_monotonic=0.0, duration_sec=2.0)
    cheater_eye_face = _make_face_3d(gaze_x=0.440)  # Melirik ~38 derajat ke luar layar!
    advance_lens_calibration(spoof_state, [cheater_eye_face], 1.5, cfg)
    res_spoof = advance_lens_calibration(spoof_state, [cheater_eye_face], 2.05, cfg)
    assert res_spoof is not None
    assert res_spoof.calibration_event == "REJECTED"
    assert spoof_state.is_calibrated is False
    assert "ANTI-SPOOF" in res_spoof.banner_text

    # Skenario Curang 2: Peserta menolehkan kepala 20 derajat saat kalibrasi -> WAJIB DITOLAK!
    spoof_head_state = MonitoringState()
    spoof_head_state.start_lens_calibration(0.0, 2.0)
    cheater_head_face = _make_face_3d(head_yaw_deg=20.0)
    advance_lens_calibration(spoof_head_state, [cheater_head_face], 1.6, cfg)
    res_head = advance_lens_calibration(
        spoof_head_state, [cheater_head_face], 2.05, cfg
    )
    assert res_head is not None and res_head.calibration_event == "REJECTED"
    assert spoof_head_state.is_calibrated is False

    # Skenario Curang 3: Peserta memejamkan/menyipitkan mata (EAR=0.08) -> WAJIB DITOLAK!
    spoof_ear_state = MonitoringState()
    spoof_ear_state.start_lens_calibration(0.0, 2.0)
    squint_face = _make_face_3d(ear=0.08)
    advance_lens_calibration(spoof_ear_state, [squint_face], 1.6, cfg)
    res_ear = advance_lens_calibration(spoof_ear_state, [squint_face], 2.05, cfg)
    assert res_ear is not None and res_ear.calibration_event == "REJECTED"

    # Skenario Curang 4: Peserta melirik bola mata ke bawah keyboard saat kalibrasi -> WAJIB DITOLAK!
    spoof_down_state = MonitoringState()
    spoof_down_state.start_lens_calibration(0.0, 2.0)
    down_eye_face = _make_face_3d(gaze_y=-0.040)
    advance_lens_calibration(spoof_down_state, [down_eye_face], 1.6, cfg)
    res_down = advance_lens_calibration(spoof_down_state, [down_eye_face], 2.05, cfg)
    assert res_down is not None and res_down.calibration_event == "REJECTED"

    # Skenario Jujur + Engsel Layar Miring 108 deg (head_pitch_deg=-14 deg di kamera) -> SUKSES!
    honest_state = MonitoringState()
    honest_state.start_lens_calibration(0.0, 2.0)
    honest_face = _make_face_3d(
        eye_x_cm=1.0,
        eye_y_cm=1.2,
        eye_z_cm=43.0,
        head_yaw_deg=-1.0,
        head_pitch_deg=-14.0,
        gaze_x=0.525,
        gaze_y=-0.128,
        ear=0.35,
    )
    advance_lens_calibration(honest_state, [honest_face], 1.4, cfg)
    advance_lens_calibration(honest_state, [honest_face], 1.8, cfg)
    res_ok = advance_lens_calibration(honest_state, [honest_face], 2.05, cfg)
    assert res_ok is not None and res_ok.calibration_event == "COMPLETED"
    assert honest_state.is_calibrated is True
    assert honest_state.smoothed_pose is None
    assert abs(honest_state.calib_yaw_offset_deg) <= cfg.max_calib_offset_yaw_deg

    # Skenario Jujur + Kedipan Alami Sesaat (4 frame blink di tengah 12 frame jendela 0.8s) -> TETAP LULUS!
    blink_cal_state = MonitoringState()
    blink_cal_state.start_lens_calibration(0.0, 2.0)
    transient_blink_face = _make_face_3d(gaze_x=0.650, gaze_y=0.050, ear=0.06)
    for idx in range(12):
        t_step = 1.25 + idx * 0.06
        sample_face = transient_blink_face if idx in (3, 4, 5, 6) else honest_face
        advance_lens_calibration(blink_cal_state, [sample_face], t_step, cfg)
    res_blink_ok = advance_lens_calibration(blink_cal_state, [honest_face], 2.05, cfg)
    assert res_blink_ok is not None and res_blink_ok.calibration_event == "COMPLETED"
    assert blink_cal_state.is_calibrated is True


def test_vor_compensation_and_blink_guard() -> None:
    """Kompensasi VOR menjaga layar tetap hijau, dan kedipan tidak memicu lonjakan iris."""
    cfg = DetectorConfig()

    # VOR: Kepala berputar +10 deg ke kanan, bola mata mengompensasi -10 deg ke kiri (tetap di layar)
    vor_face = _make_face_3d(
        head_yaw_deg=10.0, gaze_x=0.520 - (10.0 / cfg.eye_yaw_gain_deg), head_pitch_deg=8.0
    )
    code_vor, _ = evaluate_instant_indication(180.0, 5.0, [vor_face], cfg)
    assert code_vor == IndicationCode.NORMAL

    # Blink Guard: Mata berkedip (EAR=0.05) dengan spike gaze_y ekstrem -> tetap NORMAL
    blink_face = _make_face_3d(head_pitch_deg=8.0, gaze_y=-0.45, ear=0.05)
    assert (
        evaluate_instant_indication(180.0, 5.0, [blink_face], cfg)[0]
        == IndicationCode.NORMAL
    )

    # Menoleh kepala jauh sambil mata menyipit (EAR=0.05) -> tetap tertangkap IND-01!
    squint_turn = _make_face_3d(head_yaw_deg=-26.0, ear=0.05)
    assert (
        evaluate_instant_indication(180.0, 5.0, [squint_turn], cfg)[0]
        == IndicationCode.IND_01
    )


def test_ema_priority_grace_period_cooldown_and_trust_floor() -> None:
    """Menguji EMA, prioritas IND-05..NORMAL, grace period 0.3s, cooldown 5s, & floor 0%."""
    cfg = DetectorConfig()
    state = MonitoringState()

    f0 = smooth_primary_face_pose([_make_face_3d(head_yaw_deg=0.0)], state, alpha=0.42)[
        0
    ]
    assert f0.head_yaw_deg == pytest.approx(0.0)
    f1 = smooth_primary_face_pose(
        [_make_face_3d(head_yaw_deg=10.0)], state, alpha=0.42
    )[0]
    assert f1.head_yaw_deg == pytest.approx(4.2)

    assert (
        evaluate_instant_indication(20.0, 5.0, [_make_face_3d()], cfg)[0]
        == IndicationCode.IND_05
    )
    assert evaluate_instant_indication(210.0, 0.2, [], cfg)[0] == IndicationCode.IND_05
    assert (
        evaluate_instant_indication(
            150.0, 5.0, [_make_face_3d(), _make_face_3d()], cfg
        )[0]
        == IndicationCode.IND_03
    )
    assert evaluate_instant_indication(150.0, 5.0, [], cfg)[0] == IndicationCode.IND_04

    uncal = step_temporal_monitor(state, IndicationCode.IND_03, ">1 ORANG", 5.0, cfg)
    assert uncal.banner_state == BannerState.UNCALIBRATED

    state.apply_lens_calibration(0.0, 0.0, cfg)
    step_temporal_monitor(
        state, IndicationCode.IND_01, "MELIRIK MATA KE KIRI", 10.0, cfg
    )
    step_temporal_monitor(
        state, IndicationCode.NORMAL, "FOKUS", 12.2, cfg
    )  # < 0.3s flicker
    alert_step = step_temporal_monitor(
        state, IndicationCode.IND_01, "MELIRIK MATA KE KIRI", 12.6, cfg
    )
    assert alert_step.banner_state == BannerState.ALERT
    assert alert_step.should_log_evidence is True
    assert state.trust_score == pytest.approx(95.0)

    cool_step = step_temporal_monitor(
        state, IndicationCode.IND_01, "MELIRIK MATA KE KIRI", 15.0, cfg
    )
    assert cool_step.should_log_evidence is False

    assert resolve_integrity_level(100.0) == IntegrityLevel.AMAN
    assert resolve_integrity_level(70.0) == IntegrityLevel.WASPADA
    assert resolve_integrity_level(40.0) == IntegrityLevel.TINJAUAN_PENGAWAS

    low_state = MonitoringState(trust_score=2.0)
    low_state.apply_lens_calibration(0.0, 0.0, cfg)
    step_temporal_monitor(low_state, IndicationCode.IND_03, ">1 ORANG", 0.0, cfg)
    step_temporal_monitor(low_state, IndicationCode.IND_03, ">1 ORANG", 1.2, cfg)
    assert low_state.trust_score == pytest.approx(0.0)


def test_evidence_logger_and_dashboard_3d_hud_render(tmp_path: Path) -> None:
    """Menguji penyimpanan bukti .jpg/.csv serta render HUD 3D (Layar, Touchpad, & Anti-Spoof)."""
    cfg = DetectorConfig()
    clahe = create_clahe(cfg)
    raw_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    frame_pcd = preprocess_frame_lab_clahe(raw_frame, clahe, cfg)
    roi_pcd = process_roi_pcd_panel(raw_frame[120:220, 200:340], clahe, cfg)

    state = MonitoringState()
    state.start_lens_calibration(0.0, 2.0)
    cal_step = advance_lens_calibration(state, [_make_face_3d()], 1.0, cfg)
    assert cal_step is not None
    dash_cal = render_dashboard(
        raw_frame,
        [_make_face_3d()],
        state,
        cal_step,
        frame_pcd,
        roi_pcd,
        cfg.min_luminance,
        cfg,
    )
    assert dash_cal.shape == (DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3)

    # Render saat di atas dek touchpad
    state.apply_lens_calibration(0.0, 0.0, cfg)
    deck_face = _make_face_3d(head_pitch_deg=6.0, gaze_y=-0.05)
    safe_step = step_temporal_monitor(state, IndicationCode.NORMAL, "FOKUS", 3.0, cfg)
    dash_deck = render_dashboard(
        raw_frame,
        [deck_face],
        state,
        safe_step,
        frame_pcd,
        roi_pcd,
        cfg.min_luminance,
        cfg,
    )
    assert dash_deck.shape == (DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3)

    evidence_dir = tmp_path / "bukti_indikasi"
    log_csv = tmp_path / "log_indikasi_ujian.csv"
    logger = EvidenceLogger(evidence_dir, log_csv)
    saved_jpg = logger.record_indication(
        annotated_frame=dash_deck[:, :640],
        code=IndicationCode.IND_02,
        behavior_label="MELIRIK BAWAH (LEWAT TOUCHPAD)",
        duration_sec=2.6,
        mean_luminance=120.0,
        trust_score=95.0,
        flag_number=1,
        timestamp=datetime(2026, 10, 30, 23, 50, 0),
    )
    assert saved_jpg.exists()
    with open(log_csv, encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))
    assert rows[0] == CSV_HEADERS
    assert rows[1][1] == "IND-02"
