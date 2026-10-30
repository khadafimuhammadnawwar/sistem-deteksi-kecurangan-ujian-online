"""Rendering antarmuka visual OpenCV Dashboard (920x480), penanda iris, dan Panel Grid 2x2 PCD."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import BannerState, IntegrityLevel, MonitoringState, resolve_integrity_level
from src.detector import DetectedFace, TemporalStepResult
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
) -> UInt8Image:
    """Merender keseluruhan kanvas dashboard 920x480 piksel (Panel Kamera + Panel PCD 2x2)."""
    dashboard = np.zeros((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8)
    dashboard[:, :CAMERA_PANEL_WIDTH] = raw_frame_bgr
    dashboard[:, CAMERA_PANEL_WIDTH:] = COLOR_PANEL_BG

    _draw_face_overlays(dashboard, faces)
    _draw_camera_banner_and_footer(dashboard, faces, state, step_result)
    _draw_pcd_side_panel(dashboard, len(faces), state, frame_pcd, roi_pcd, min_luminance)

    return dashboard


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
) -> None:
    """Menggambar status banner atas dan informasi kalibrasi/geometri kepala & mata di bawah."""
    banner_color = BANNER_COLORS[step_result.banner_state]
    cv2.rectangle(dashboard, (0, 0), (CAMERA_PANEL_WIDTH, 40), banner_color, -1)
    cv2.putText(
        dashboard,
        step_result.banner_text,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
    )

    cv2.rectangle(dashboard, (0, 450), (CAMERA_PANEL_WIDTH, 480), (20, 20, 20), -1)
    cal_text = (
        "TERKALIBRASI (Mode 1)"
        if state.is_calibrated
        else "BELUM KALIBRASI [Tekan 'C']"
    )
    cal_color = (100, 255, 100) if state.is_calibrated else (0, 255, 255)
    cv2.putText(
        dashboard,
        cal_text,
        (10, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        cal_color,
        1,
    )

    yaw = faces[0].yaw if faces else state.baseline_yaw
    pitch = faces[0].pitch if faces else state.baseline_pitch
    gaze_x = faces[0].gaze_x if faces else state.baseline_gaze_x
    gaze_y = faces[0].gaze_y if faces else state.baseline_gaze_y
    cv2.putText(
        dashboard,
        f"Y:{yaw:.2f} P:{pitch:.2f} | EyeX:{gaze_x:.2f} EyeY:{gaze_y:+.2f}",
        (275, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
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
