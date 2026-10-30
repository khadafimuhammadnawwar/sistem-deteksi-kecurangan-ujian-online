"""Rendering OpenCV Dashboard (920x480), panduan Lensa Anti-Spoof, HUD 3D (cm), & Grid 2x2 PCD."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import (
    BannerState,
    DetectorConfig,
    IntegrityLevel,
    MonitoringState,
    resolve_integrity_level,
)
from src.detector import DetectedFace, TemporalStepResult, project_gaze_3d
from src.pcd_pipeline import FramePreprocessResult, ROIPCDResult

UInt8Image = NDArray[np.uint8]

DASHBOARD_WIDTH: int = 920
DASHBOARD_HEIGHT: int = 480
CAMERA_PANEL_WIDTH: int = 640

COLOR_PRIMARY_FACE: tuple[int, int, int] = (255, 200, 0)
COLOR_EXTRA_FACE: tuple[int, int, int] = (0, 0, 255)
COLOR_NOSE_DOT: tuple[int, int, int] = (0, 255, 255)
COLOR_IRIS_DOT: tuple[int, int, int] = (255, 0, 255)
COLOR_PANEL_BG: tuple[int, int, int] = (30, 30, 30)
COLOR_BORDER: tuple[int, int, int] = (85, 85, 85)

BANNER_COLORS: dict[BannerState, tuple[int, int, int]] = {
    BannerState.UNCALIBRATED: (160, 90, 0),
    BannerState.CALIBRATING: (170, 0, 170),
    BannerState.SAFE: (0, 160, 0),
    BannerState.WARNING: (0, 150, 240),
    BannerState.ALERT: (0, 0, 220),
}

INTEGRITY_COLORS: dict[IntegrityLevel, tuple[int, int, int]] = {
    IntegrityLevel.AMAN: (0, 240, 0),
    IntegrityLevel.WASPADA: (0, 200, 255),
    IntegrityLevel.TINJAUAN_PENGAWAS: (0, 0, 255),
}


def render_dashboard(
    raw_frame_bgr: UInt8Image,
    faces: Sequence[DetectedFace],
    state: MonitoringState,
    step_result: TemporalStepResult,
    frame_pcd: FramePreprocessResult,
    roi_pcd: ROIPCDResult,
    min_luminance: float,
    config: DetectorConfig | None = None,
) -> UInt8Image:
    """Merender keseluruhan kanvas dashboard 920x480 piksel (Panel Kamera + Panel PCD 2x2)."""
    cfg = config or DetectorConfig(min_luminance=min_luminance)
    dashboard = np.zeros((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8)
    dashboard[:, :CAMERA_PANEL_WIDTH] = raw_frame_bgr
    dashboard[:, CAMERA_PANEL_WIDTH:] = COLOR_PANEL_BG

    _draw_face_overlays(dashboard, faces)
    if step_result.banner_state == BannerState.CALIBRATING:
        _draw_lens_calibration_guide(dashboard)
    _draw_camera_banner_and_footer(dashboard, faces, state, step_result, cfg)
    _draw_pcd_side_panel(dashboard, len(faces), state, frame_pcd, roi_pcd, min_luminance)

    return dashboard


def _draw_lens_calibration_guide(dashboard: UInt8Image) -> None:
    """Menggambar penanda visual arah ke Lensa Fisik Webcam di bezel atas saat kalibrasi."""
    cx = CAMERA_PANEL_WIDTH // 2
    guide_color = (0, 255, 255)

    cv2.arrowedLine(dashboard, (cx, 135), (cx, 48), guide_color, 4, tipLength=0.35)
    cv2.circle(dashboard, (cx, 48), 9, (0, 0, 255), 2)
    cv2.putText(
        dashboard,
        "TATAP LURUS LENSA FISIK KAMERA DI BEZEL ATAS",
        (cx - 215, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        guide_color,
        2,
    )
    cv2.putText(
        dashboard,
        "Anti-Spoof Aktif: Lirikan/Tolehan Palsu Akan Ditolak",
        (cx - 205, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 220, 255),
        1,
    )


def _draw_face_overlays(dashboard: UInt8Image, faces: Sequence[DetectedFace]) -> None:
    """Menggambar bounding box terurut, titik hidung, dan titik pupil/iris pada wajah utama."""
    for idx, face in enumerate(faces):
        x_min, y_min, x_max, y_max = face.bbox
        box_color = COLOR_PRIMARY_FACE if idx == 0 else COLOR_EXTRA_FACE
        cv2.rectangle(dashboard, (x_min, y_min), (x_max, y_max), box_color, 2)
        cv2.putText(
            dashboard,
            f"Wajah #{idx + 1}",
            (x_min, max(55, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
        )
        if idx == 0:
            cv2.circle(dashboard, face.nose_px, 4, COLOR_NOSE_DOT, -1)
            for pupil_pt in face.iris_px:
                cv2.circle(dashboard, pupil_pt, 3, COLOR_IRIS_DOT, -1)


def _draw_camera_banner_and_footer(
    dashboard: UInt8Image,
    faces: Sequence[DetectedFace],
    state: MonitoringState,
    step_result: TemporalStepResult,
    config: DetectorConfig,
) -> None:
    """Menggambar banner atas dan HUD koordinat pandang 3D (cm) real-time di bawah."""
    banner_color = BANNER_COLORS[step_result.banner_state]
    if (
        step_result.banner_state == BannerState.UNCALIBRATED
        and state.last_calibration_error is not None
    ):
        banner_color = (0, 65, 210)

    cv2.rectangle(dashboard, (0, 0), (CAMERA_PANEL_WIDTH, 40), banner_color, -1)
    cv2.putText(
        dashboard,
        step_result.banner_text,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        2,
    )

    cv2.rectangle(dashboard, (0, 450), (CAMERA_PANEL_WIDTH, 480), (20, 20, 20), -1)
    if state.is_calibrating:
        cal_text = "VALIDASI LENSA..."
        cal_color = (255, 180, 255)
    elif state.is_calibrated:
        cal_text = "3D LOCKED [SPASI/C]"
        cal_color = (100, 255, 100)
    else:
        cal_text = "BELUM KALIBRASI [SPASI/C]"
        cal_color = (0, 255, 255)

    cv2.putText(
        dashboard,
        cal_text,
        (10, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        cal_color,
        1,
    )

    if faces:
        primary = faces[0]
        hit = project_gaze_3d(
            primary,
            config,
            calib_yaw_offset_deg=state.calib_yaw_offset_deg,
            calib_pitch_offset_deg=state.calib_pitch_offset_deg,
        )
        if hit.on_keyboard_deck:
            hud_text = (
                f"X:{hit.hit_x_cm:+5.1f}cm | "
                f"Touchpad:{hit.deck_z_cm:4.1f}/{config.touchpad_end_depth_cm:.0f}cm | "
                f"Z:{primary.eye_z_cm:.0f}cm"
            )
        else:
            hud_text = (
                f"X:{hit.hit_x_cm:+5.1f}cm | "
                f"Y:{hit.screen_y_cm:+5.1f}cm | "
                f"Z:{primary.eye_z_cm:.0f}cm"
            )
    else:
        hud_text = "X:  -- cm | Y:  -- cm | Z: -- cm"

    cv2.putText(
        dashboard,
        hud_text,
        (242, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 255, 200),
        1,
    )


def _draw_pcd_side_panel(
    dashboard: UInt8Image,
    face_count: int,
    state: MonitoringState,
    frame_pcd: FramePreprocessResult,
    roi_pcd: ROIPCDResult,
    min_luminance: float,
) -> None:
    """Menggambar metrik sistem, status integritas, dan Grid 2x2 tahapan PCD di sisi kanan."""
    cv2.putText(
        dashboard,
        "PANEL MONITORING PCD",
        (652, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
    )

    integrity = resolve_integrity_level(state.trust_score)
    integrity_color = INTEGRITY_COLORS[integrity]
    cv2.putText(
        dashboard,
        f"Trust Score   : {state.trust_score:.0f}%",
        (652, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        integrity_color,
        2,
    )
    cv2.putText(
        dashboard,
        f"Status: {integrity.value}",
        (652, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        integrity_color,
        1,
    )
    cv2.putText(
        dashboard,
        f"Total Indikasi: {state.total_flags}",
        (652, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        dashboard,
        f"Jumlah Wajah  : {face_count}",
        (652, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (220, 220, 220),
        1,
    )

    lum_color = (0, 255, 0) if frame_pcd.mean_luminance >= min_luminance else (0, 0, 255)
    cv2.putText(
        dashboard,
        f"Luminansi (Y) : {frame_pcd.mean_luminance:.1f}",
        (652, 142),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        lum_color,
        1,
    )
    cv2.putText(
        dashboard,
        f"Edge ROI/Glob : {roi_pcd.roi_edge_density:.1f}% / {frame_pcd.global_edge_density:.1f}%",
        (652, 164),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
    )

    cv2.line(dashboard, (652, 175), (905, 175), COLOR_BORDER, 1)

    tiles: list[tuple[int, int, UInt8Image, str]] = [
        (655, 185, roi_pcd.vis_clahe, "1.Gray+CLAHE"),
        (790, 185, roi_pcd.vis_gaussian, "2.Gauss 5x5"),
        (655, 325, roi_pcd.vis_otsu, "3.Otsu Biner"),
        (790, 325, roi_pcd.vis_canny, "4.Canny Edge"),
    ]
    for x_pos, y_pos, tile_img, caption in tiles:
        tile_h, tile_w, _ = tile_img.shape
        dashboard[y_pos : y_pos + tile_h, x_pos : x_pos + tile_w] = tile_img
        cv2.rectangle(
            dashboard,
            (x_pos, y_pos),
            (x_pos + tile_w, y_pos + tile_h),
            COLOR_BORDER,
            1,
        )
        cv2.putText(
            dashboard,
            caption,
            (x_pos, y_pos + tile_h + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (205, 205, 205),
            1,
        )
