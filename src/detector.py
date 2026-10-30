"""Detektor MediaPipe 3D FaceLandmarker, L-Shaped 3D Gaze Ray-Casting, & Anti-Spoof Gate."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self
import urllib.error
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import numpy as np
from numpy.typing import NDArray

from src.config import (
    BannerState,
    CalibrationSample,
    DetectorConfig,
    IndicationCode,
    MonitoringState,
)

UInt8Image = NDArray[np.uint8]
FloatMatrix = NDArray[np.floating]

# Indeks kanonik 478 titik landmark MediaPipe FaceLandmarker (Kepala & Iris)
NOSE_TIP_IDX: int = 1
LEFT_CHEEK_IDX: int = 234
RIGHT_CHEEK_IDX: int = 454
CHIN_IDX: int = 152

# Mata sisi kiri citra tercermin (Anatomical Right Eye: sudut 33-133, kelopak 159-145, iris 468)
R_EYE_OUTER_IDX: int = 33
R_EYE_INNER_IDX: int = 133
R_EYE_TOP_IDX: int = 159
R_EYE_BOTTOM_IDX: int = 145
R_IRIS_CENTER_IDX: int = 468

# Mata sisi kanan citra tercermin (Anatomical Left Eye: sudut 362-263, kelopak 386-374, iris 473)
L_EYE_INNER_IDX: int = 362
L_EYE_OUTER_IDX: int = 263
L_EYE_TOP_IDX: int = 386
L_EYE_BOTTOM_IDX: int = 374
L_IRIS_CENTER_IDX: int = 473

LEFT_EYE_OUTER_IDX: int = R_EYE_OUTER_IDX
RIGHT_EYE_OUTER_IDX: int = L_EYE_OUTER_IDX

BBOX_PADDING_PX: int = 15


class LandmarkPoint(Protocol):
    """Kontrak koordinat ternormalisasi landmark wajah (0.0 - 1.0)."""

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...


@dataclass(frozen=True)
class DetectedFace:
    """Data geometri 3D metrik kepala (cm & derajat), rasio iris, dan bounding box wajah."""

    bbox: tuple[int, int, int, int]
    nose_px: tuple[int, int]
    iris_px: tuple[tuple[int, int], tuple[int, int]]
    area: int
    eye_x_cm: float = 0.0
    eye_y_cm: float = 0.5
    eye_z_cm: float = 43.0
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    gaze_x: float = 0.520
    gaze_y: float = -0.130
    ear: float = 0.30


@dataclass(frozen=True)
class GazeRayHit3D:
    """Hasil perpotongan sinar pandang 3D terhadap bidang layar vertikal & dek keyboard/touchpad."""

    screen_x_cm: float
    screen_y_cm: float
    hit_x_cm: float
    deck_z_cm: float
    on_keyboard_deck: bool
    total_yaw_deg: float
    total_pitch_deg: float
    eye_yaw_deg: float
    eye_pitch_deg: float


@dataclass(frozen=True)
class TemporalStepResult:
    """Keputusan evaluasi temporal untuk satu siklus frame."""

    banner_state: BannerState
    banner_text: str
    effective_code: IndicationCode
    effective_label: str
    elapsed_sec: float
    should_log_evidence: bool
    calibration_event: str | None = None  # "STARTED", "COMPLETED", atau "REJECTED"


def ensure_landmarker_model(model_path: Path, model_url: str) -> Path:
    """Memastikan file model face_landmarker.task tersedia; unduh otomatis jika belum ada."""
    if model_path.exists() and model_path.stat().st_size > 100_000:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_suffix(".task.tmp")
    print(f"[INFO] Mengunduh model MediaPipe FaceLandmarker ke '{model_path}'...")
    try:
        with urllib.request.urlopen(model_url, timeout=30) as response:
            data = response.read()
        if len(data) < 100_000:
            raise OSError("Ukuran file model yang diunduh tidak valid (terlalu kecil).")
        temp_path.write_bytes(data)
        temp_path.replace(model_path)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Gagal mengunduh model FaceLandmarker dari '{model_url}': {exc}"
        ) from exc

    return model_path


def calculate_yaw_pitch_ratios(landmarks: Sequence[LandmarkPoint]) -> tuple[float, float]:
    """Menghitung rasio geometri kepala 2D Yaw dan Pitch (digunakan sebagai fallback pinhole)."""
    nose = landmarks[NOSE_TIP_IDX]
    left_cheek = landmarks[LEFT_CHEEK_IDX]
    right_cheek = landmarks[RIGHT_CHEEK_IDX]
    left_eye = landmarks[R_EYE_OUTER_IDX]
    right_eye = landmarks[L_EYE_OUTER_IDX]
    chin = landmarks[CHIN_IDX]

    face_width = max(0.001, right_cheek.x - left_cheek.x)
    yaw_ratio = (nose.x - left_cheek.x) / face_width

    eye_mid_y = (left_eye.y + right_eye.y) / 2.0
    face_height = max(0.001, chin.y - eye_mid_y)
    pitch_ratio = (nose.y - eye_mid_y) / face_height

    return float(yaw_ratio), float(pitch_ratio)


def calculate_iris_gaze_and_ear(
    landmarks: Sequence[LandmarkPoint],
) -> tuple[float, float, float]:
    """Menghitung rasio posisi pupil/iris mata (Gaze X, Gaze Y) dan Eye Aspect Ratio (EAR)."""
    if len(landmarks) <= L_IRIS_CENTER_IDX:
        return 0.520, -0.130, 0.30

    r_width = max(0.001, landmarks[R_EYE_INNER_IDX].x - landmarks[R_EYE_OUTER_IDX].x)
    l_width = max(0.001, landmarks[L_EYE_OUTER_IDX].x - landmarks[L_EYE_INNER_IDX].x)

    r_gaze_x = (landmarks[R_IRIS_CENTER_IDX].x - landmarks[R_EYE_OUTER_IDX].x) / r_width
    l_gaze_x = (landmarks[L_IRIS_CENTER_IDX].x - landmarks[L_EYE_INNER_IDX].x) / l_width
    gaze_x = (r_gaze_x + l_gaze_x) / 2.0

    r_corner_mid_y = (landmarks[R_EYE_OUTER_IDX].y + landmarks[R_EYE_INNER_IDX].y) / 2.0
    l_corner_mid_y = (landmarks[L_EYE_INNER_IDX].y + landmarks[L_EYE_OUTER_IDX].y) / 2.0
    r_gaze_y = (landmarks[R_IRIS_CENTER_IDX].y - r_corner_mid_y) / r_width
    l_gaze_y = (landmarks[L_IRIS_CENTER_IDX].y - l_corner_mid_y) / l_width
    gaze_y = (r_gaze_y + l_gaze_y) / 2.0

    r_ear = max(0.0, landmarks[R_EYE_BOTTOM_IDX].y - landmarks[R_EYE_TOP_IDX].y) / r_width
    l_ear = max(0.0, landmarks[L_EYE_BOTTOM_IDX].y - landmarks[L_EYE_TOP_IDX].y) / l_width
    ear = (r_ear + l_ear) / 2.0

    return float(gaze_x), float(gaze_y), float(ear)


def extract_3d_pose_from_matrix_or_pinhole(
    landmarks: Sequence[LandmarkPoint],
    transform_matrix: FloatMatrix | None,
    frame_width: int,
    frame_height: int,
    config: DetectorConfig,
) -> tuple[float, float, float, float, float]:
    """
    Mengekstrak posisi mata 3D (X_cm, Y_down_cm, Z_cm) & sudut kepala (head_yaw_deg, head_pitch_deg)
    relatif terhadap lensa fisik webcam (0, 0, 0) menggunakan matriks 4x4 MediaPipe atau pinhole.
    """
    if transform_matrix is not None and transform_matrix.shape == (4, 4):
        eye_x_cm = float(transform_matrix[0, 3])
        eye_y_cm = float(-transform_matrix[1, 3])
        eye_z_cm = float(min(120.0, max(18.0, abs(float(transform_matrix[2, 3])))))
        rz_x = float(transform_matrix[0, 2])
        rz_y = float(transform_matrix[1, 2])
        rz_z = float(transform_matrix[2, 2])
        head_yaw_deg = math.degrees(math.atan2(rz_x, rz_z))
        head_pitch_deg = math.degrees(math.atan2(-rz_y, rz_z))
        return eye_x_cm, eye_y_cm, eye_z_cm, head_yaw_deg, head_pitch_deg

    focal_px = frame_width / (2.0 * math.tan(math.radians(config.camera_fov_deg * 0.5)))
    cheek_width_px = max(
        8.0,
        (landmarks[RIGHT_CHEEK_IDX].x - landmarks[LEFT_CHEEK_IDX].x) * frame_width,
    )
    eye_z_cm = min(120.0, max(18.0, (config.avg_face_width_cm * focal_px) / cheek_width_px))
    mid_eye_x_px = (
        (landmarks[R_EYE_OUTER_IDX].x + landmarks[L_EYE_OUTER_IDX].x) * 0.5 * frame_width
    )
    mid_eye_y_px = (
        (landmarks[R_EYE_OUTER_IDX].y + landmarks[L_EYE_OUTER_IDX].y) * 0.5 * frame_height
    )
    eye_x_cm = (mid_eye_x_px - frame_width * 0.5) * eye_z_cm / focal_px
    eye_y_cm = (mid_eye_y_px - frame_height * 0.5) * eye_z_cm / focal_px

    yaw_ratio, pitch_ratio = calculate_yaw_pitch_ratios(landmarks)
    head_yaw_deg = (yaw_ratio - 0.50) * 90.0
    head_pitch_deg = (pitch_ratio - 0.35) * 90.0
    return (
        float(eye_x_cm),
        float(eye_y_cm),
        float(eye_z_cm),
        float(head_yaw_deg),
        float(head_pitch_deg),
    )


def compute_eyeball_angles_deg(
    gaze_x: float,
    gaze_y: float,
    ear: float,
    config: DetectorConfig,
) -> tuple[float, float]:
    """
    Mengonversi pergeseran rasio iris di dalam soket mata menjadi sudut rotasi bola mata (derajat).
    Dilengkapi EAR Blink Guard: saat berkedip (EAR < min_ear_for_gaze), sudut bola mata dinolkan
    agar kedipan tidak memicu lonjakan palsu sementara pelacakan kepala 3D tetap berjalan.
    """
    if ear < config.min_ear_for_gaze:
        return 0.0, 0.0

    dgx = gaze_x - config.neutral_gaze_x
    dgy = gaze_y - config.neutral_gaze_y
    eye_yaw_deg = dgx * config.eye_yaw_gain_deg
    pitch_gain = (
        config.eye_pitch_down_gain_deg if dgy >= 0.0 else config.eye_pitch_up_gain_deg
    )
    eye_pitch_deg = dgy * pitch_gain
    return float(eye_yaw_deg), float(eye_pitch_deg)


def project_gaze_3d(
    face: DetectedFace,
    config: DetectorConfig,
    calib_yaw_offset_deg: float = 0.0,
    calib_pitch_offset_deg: float = 0.0,
) -> GazeRayHit3D:
    """
    Memproyeksikan sinar pandang 3D dari mata peserta (X_e, Y_e, Z_e) ke geometri fisik laptop
    berbentuk L (Bidang Layar Vertikal Z = 0 dan Bidang Dek Keyboard/Touchpad Y = hinge_y_cm).
    """
    eye_yaw_deg, eye_pitch_deg = compute_eyeball_angles_deg(
        face.gaze_x, face.gaze_y, face.ear, config
    )
    total_yaw_deg = max(
        -75.0,
        min(75.0, face.head_yaw_deg + eye_yaw_deg + calib_yaw_offset_deg),
    )
    total_pitch_deg = max(
        -75.0,
        min(75.0, face.head_pitch_deg + eye_pitch_deg + calib_pitch_offset_deg),
    )

    tan_yaw = math.tan(math.radians(total_yaw_deg))
    tan_pitch = math.tan(math.radians(total_pitch_deg))

    screen_x_cm = face.eye_x_cm + face.eye_z_cm * tan_yaw
    screen_y_cm = face.eye_y_cm + face.eye_z_cm * tan_pitch

    if screen_y_cm <= config.hinge_y_cm or tan_pitch <= 0.001:
        return GazeRayHit3D(
            screen_x_cm=float(screen_x_cm),
            screen_y_cm=float(screen_y_cm),
            hit_x_cm=float(screen_x_cm),
            deck_z_cm=0.0,
            on_keyboard_deck=False,
            total_yaw_deg=float(total_yaw_deg),
            total_pitch_deg=float(total_pitch_deg),
            eye_yaw_deg=float(eye_yaw_deg),
            eye_pitch_deg=float(eye_pitch_deg),
        )

    drop_cm = max(6.0, config.hinge_y_cm - face.eye_y_cm)
    t_ray_cm = drop_cm / tan_pitch
    deck_z_cm = max(0.0, face.eye_z_cm - t_ray_cm)
    hit_x_cm = face.eye_x_cm + t_ray_cm * tan_yaw

    return GazeRayHit3D(
        screen_x_cm=float(screen_x_cm),
        screen_y_cm=float(screen_y_cm),
        hit_x_cm=float(hit_x_cm),
        deck_z_cm=float(deck_z_cm),
        on_keyboard_deck=True,
        total_yaw_deg=float(total_yaw_deg),
        total_pitch_deg=float(total_pitch_deg),
        eye_yaw_deg=float(eye_yaw_deg),
        eye_pitch_deg=float(eye_pitch_deg),
    )


def validate_and_compute_lens_offset(
    samples: Sequence[CalibrationSample],
    config: DetectorConfig,
) -> tuple[bool, float, float, str]:
    """
    Memvalidasi sampel kalibrasi 1 titik menatap Lensa Fisik Webcam (0, 0, 0) menggunakan
    5-Layer Anti-Spoof Gate agar peserta tidak dapat memalsukan sudut pandang saat kalibrasi.
    """
    if not samples:
        return False, 0.0, 0.0, "KALIBRASI GAGAL: SAMPEL WAJAH TIDAK DITEMUKAN"

    n = len(samples)
    mean_x = sum(s.eye_x_cm for s in samples) / n
    mean_y = sum(s.eye_y_cm for s in samples) / n
    mean_z = sum(s.eye_z_cm for s in samples) / n
    mean_hyaw = sum(s.head_yaw_deg for s in samples) / n
    mean_hpitch = sum(s.head_pitch_deg for s in samples) / n
    mean_gx = sum(s.gaze_x for s in samples) / n
    mean_gy = sum(s.gaze_y for s in samples) / n
    mean_ear = sum(s.ear for s in samples) / n

    # Layer 1: Validasi jarak fisik wajar di depan laptop (25 - 85 cm)
    if not (config.min_calib_distance_cm <= mean_z <= config.max_calib_distance_cm):
        return (
            False,
            0.0,
            0.0,
            f"DITOLAK: JARAK WAJAH {mean_z:.0f}CM (WAJIB 25-85CM)",
        )

    # Layer 2: Validasi mata terbuka (mencegah kalibrasi sambil merem/menyipit ekstrem)
    if mean_ear < config.min_calib_ear:
        return False, 0.0, 0.0, "DITOLAK: BUKA MATA & TATAP LENSA KAMERA"

    # Layer 3: Validasi kestabilan tatapan (mencegah lirikan cepat / saccade spoofing)
    if n >= 2:
        var_yaw = sum((s.head_yaw_deg - mean_hyaw) ** 2 for s in samples) / n
        var_gx = sum((s.gaze_x - mean_gx) ** 2 for s in samples) / n
        if (
            math.sqrt(var_yaw) > config.max_calib_std_yaw_deg
            or math.sqrt(var_gx) > config.max_calib_std_gaze_x
        ):
            return False, 0.0, 0.0, "DITOLAK: TAHAN KEPALA & MATA TETAP STABIL"

    # Sudut optik sejati dari mata peserta (mean_x, mean_y, mean_z) ke lensa fisik (0, 0, 0)
    target_yaw_deg = math.degrees(math.atan2(-mean_x, mean_z))
    target_pitch_deg = math.degrees(math.atan2(-mean_y, mean_z))

    # Layer 4: Validasi pose kepala menghadap kamera (mendukung kemiringan engsel layar 90-115 deg)
    pitch_err_deg = mean_hpitch - target_pitch_deg
    if (
        abs(mean_hyaw - target_yaw_deg) > config.max_calib_head_yaw_error_deg
        or not (
            config.min_calib_head_pitch_error_deg
            <= pitch_err_deg
            <= config.max_calib_head_pitch_error_deg
        )
    ):
        return False, 0.0, 0.0, "ANTI-SPOOF: HADAPKAN WAJAH LURUS KE KAMERA!"

    # Layer 5: Validasi batas anatomi iris vertikal & Angle-Kappa (menolak lirikan curang)
    if not (config.min_calib_gaze_y <= mean_gy <= config.max_calib_gaze_y):
        return False, 0.0, 0.0, "ANTI-SPOOF: JANGAN MELIRIK ATAS/BAWAH SAAT KALIBRASI!"

    raw_eye_yaw, raw_eye_pitch = compute_eyeball_angles_deg(
        mean_gx, mean_gy, mean_ear, config
    )
    yaw_offset = target_yaw_deg - (mean_hyaw + raw_eye_yaw)
    pitch_offset = target_pitch_deg - (mean_hpitch + raw_eye_pitch)

    if abs(yaw_offset) > config.max_calib_offset_yaw_deg or not (
        config.min_calib_offset_pitch_deg
        <= pitch_offset
        <= config.max_calib_offset_pitch_deg
    ):
        return False, 0.0, 0.0, "ANTI-SPOOF: JANGAN MELIRIK, TATAP TEPAT LENSA!"

    return (
        True,
        float(yaw_offset),
        float(pitch_offset),
        "KALIBRASI 3D AKTIF: ZONA 1CM LAYAR & 21CM TOUCHPAD TERKUNCI",
    )


def _clamp_pixel(norm_x: float, norm_y: float, width: int, height: int) -> tuple[int, int]:
    """Mengonversi koordinat ternormalisasi ke piksel utuh dalam batas dimensi citra."""
    return (
        min(max(0, int(norm_x * width)), width - 1),
        min(max(0, int(norm_y * height)), height - 1),
    )


def extract_faces_sorted_by_area(
    multi_face_landmarks: Sequence[Sequence[LandmarkPoint]],
    frame_width: int,
    frame_height: int,
    transformation_matrices: Sequence[FloatMatrix] | None = None,
    config: DetectorConfig | None = None,
) -> list[DetectedFace]:
    """Mengekstrak geometri 3D seluruh wajah dan mengurutkannya dari luas terbesar (Wajah #1)."""
    cfg = config or DetectorConfig(frame_width=frame_width, frame_height=frame_height)
    detected_faces: list[DetectedFace] = []

    for idx, landmarks in enumerate(multi_face_landmarks):
        xs = [int(lm.x * frame_width) for lm in landmarks]
        ys = [int(lm.y * frame_height) for lm in landmarks]

        x_min = max(0, min(xs) - BBOX_PADDING_PX)
        x_max = min(frame_width, max(xs) + BBOX_PADDING_PX)
        y_min = max(0, min(ys) - BBOX_PADDING_PX)
        y_max = min(frame_height, max(ys) + BBOX_PADDING_PX)

        area = max(0, x_max - x_min) * max(0, y_max - y_min)
        nose = landmarks[NOSE_TIP_IDX]
        nose_px = _clamp_pixel(nose.x, nose.y, frame_width, frame_height)

        if len(landmarks) > L_IRIS_CENTER_IDX:
            r_iris = landmarks[R_IRIS_CENTER_IDX]
            l_iris = landmarks[L_IRIS_CENTER_IDX]
            iris_px = (
                _clamp_pixel(r_iris.x, r_iris.y, frame_width, frame_height),
                _clamp_pixel(l_iris.x, l_iris.y, frame_width, frame_height),
            )
        else:
            iris_px = (nose_px, nose_px)

        mat: FloatMatrix | None = None
        if transformation_matrices is not None and idx < len(transformation_matrices):
            mat = np.asarray(transformation_matrices[idx], dtype=np.float64)

        eye_x_cm, eye_y_cm, eye_z_cm, head_yaw_deg, head_pitch_deg = (
            extract_3d_pose_from_matrix_or_pinhole(
                landmarks, mat, frame_width, frame_height, cfg
            )
        )
        gaze_x, gaze_y, ear = calculate_iris_gaze_and_ear(landmarks)

        detected_faces.append(
            DetectedFace(
                bbox=(x_min, y_min, x_max, y_max),
                nose_px=nose_px,
                iris_px=iris_px,
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
        )

    detected_faces.sort(key=lambda face: face.area, reverse=True)
    return detected_faces


def smooth_primary_face_pose(
    faces: list[DetectedFace],
    state: MonitoringState,
    alpha: float,
) -> list[DetectedFace]:
    """Menerapkan EMA smoothing pada sinyal posisi 3D, sudut kepala, dan rasio iris Wajah #1."""
    if not faces:
        state.reset_ema()
        return faces

    primary = faces[0]
    smoothed = state.update_ema_pose(
        raw_pose=(
            primary.eye_x_cm,
            primary.eye_y_cm,
            primary.eye_z_cm,
            primary.head_yaw_deg,
            primary.head_pitch_deg,
            primary.gaze_x,
            primary.gaze_y,
        ),
        alpha=alpha,
    )
    smoothed_primary = replace(
        primary,
        eye_x_cm=smoothed[0],
        eye_y_cm=smoothed[1],
        eye_z_cm=smoothed[2],
        head_yaw_deg=smoothed[3],
        head_pitch_deg=smoothed[4],
        gaze_x=smoothed[5],
        gaze_y=smoothed[6],
    )
    return [smoothed_primary, *faces[1:]]


def advance_lens_calibration(
    state: MonitoringState,
    raw_faces: Sequence[DetectedFace],
    now_monotonic: float,
    config: DetectorConfig,
) -> TemporalStepResult | None:
    """
    Menjalankan kalibrasi 1 titik menatap Lensa Fisik Webcam (2.0 detik) yang divalidasi oleh
    5-Layer Anti-Spoof Gate. Batas layar & touchpad tetap terkunci secara hardware di config.
    """
    if state.calibration_deadline is None:
        return None

    if len(raw_faces) != 1:
        msg = "KALIBRASI BATAL: PASTIKAN TEPAT 1 WAJAH DI KAMERA"
        state.cancel_calibration(msg)
        return TemporalStepResult(
            banner_state=BannerState.UNCALIBRATED,
            banner_text=msg,
            effective_code=IndicationCode.NORMAL,
            effective_label="KALIBRASI BATAL",
            elapsed_sec=0.0,
            should_log_evidence=False,
            calibration_event="REJECTED",
        )

    remaining = state.calibration_deadline - now_monotonic
    primary = raw_faces[0]
    if remaining <= config.calibration_sample_window_sec:
        state.calibration_samples.append(
            CalibrationSample(
                eye_x_cm=primary.eye_x_cm,
                eye_y_cm=primary.eye_y_cm,
                eye_z_cm=primary.eye_z_cm,
                head_yaw_deg=primary.head_yaw_deg,
                head_pitch_deg=primary.head_pitch_deg,
                gaze_x=primary.gaze_x,
                gaze_y=primary.gaze_y,
                ear=primary.ear,
            )
        )

    if remaining > 0.0:
        return TemporalStepResult(
            banner_state=BannerState.CALIBRATING,
            banner_text=f"TATAP LURUS LENSA FISIK WEBCAM DI ATAS ({remaining:.1f}s)",
            effective_code=IndicationCode.NORMAL,
            effective_label="VALIDASI ANTI-SPOOF",
            elapsed_sec=0.0,
            should_log_evidence=False,
        )

    samples = state.calibration_samples or [
        CalibrationSample(
            eye_x_cm=primary.eye_x_cm,
            eye_y_cm=primary.eye_y_cm,
            eye_z_cm=primary.eye_z_cm,
            head_yaw_deg=primary.head_yaw_deg,
            head_pitch_deg=primary.head_pitch_deg,
            gaze_x=primary.gaze_x,
            gaze_y=primary.gaze_y,
            ear=primary.ear,
        )
    ]
    is_valid, yaw_offset, pitch_offset, message = validate_and_compute_lens_offset(
        samples, config
    )
    if not is_valid:
        state.cancel_calibration(message)
        print(f"[KALIBRASI DITOLAK] {message}")
        return TemporalStepResult(
            banner_state=BannerState.UNCALIBRATED,
            banner_text=message,
            effective_code=IndicationCode.NORMAL,
            effective_label="KALIBRASI DITOLAK",
            elapsed_sec=0.0,
            should_log_evidence=False,
            calibration_event="REJECTED",
        )

    state.apply_lens_calibration(yaw_offset, pitch_offset, config)
    print(
        f"[KALIBRASI 3D ANTI-SPOOF SUKSES] "
        f"Offset Anatomi Yaw:{state.calib_yaw_offset_deg:+.2f}deg "
        f"Pitch:{state.calib_pitch_offset_deg:+.2f}deg | "
        f"Batas Aman: X=+-{config.half_safe_width_cm:.1f}cm, "
        f"Atas=-{config.edge_margin_cm:.1f}cm, Touchpad={config.touchpad_end_depth_cm:.1f}cm"
    )
    return TemporalStepResult(
        banner_state=BannerState.SAFE,
        banner_text=message,
        effective_code=IndicationCode.NORMAL,
        effective_label="PESERTA FOKUS UJIAN",
        elapsed_sec=0.0,
        should_log_evidence=False,
        calibration_event="COMPLETED",
    )


def evaluate_instant_indication(
    mean_luminance: float,
    global_edge_density: float,
    faces: Sequence[DetectedFace],
    config: DetectorConfig,
    calib_yaw_offset_deg: float = 0.0,
    calib_pitch_offset_deg: float = 0.0,
) -> tuple[IndicationCode, str]:
    """
    Mengevaluasi indikasi instan satu frame menggunakan L-Shaped 3D Gaze Ray-Casting:
    - IND-05 > IND-03 > IND-04 > IND-01 (>1 cm kiri/kanan) > IND-02 (>1 cm atas / >=21 cm touchpad).
    """
    if mean_luminance < config.min_luminance:
        return IndicationCode.IND_05, "KAMERA TERTUTUP / RUANGAN GELAP"

    if len(faces) == 0 and global_edge_density < config.min_global_edge_density:
        return IndicationCode.IND_05, "KAMERA TERTUTUP / OKLUSI POLOS"

    if len(faces) > 1:
        return IndicationCode.IND_03, "TERDETEKSI >1 ORANG DI KAMERA"

    if len(faces) == 0:
        return IndicationCode.IND_04, "WAJAH TIDAK TERDETEKSI"

    primary = faces[0]
    hit = project_gaze_3d(
        primary,
        config,
        calib_yaw_offset_deg=calib_yaw_offset_deg,
        calib_pitch_offset_deg=calib_pitch_offset_deg,
    )

    center_head_yaw = math.degrees(math.atan2(-primary.eye_x_cm, primary.eye_z_cm))
    center_head_pitch = math.degrees(math.atan2(-primary.eye_y_cm, primary.eye_z_cm))
    rel_head_yaw = primary.head_yaw_deg - center_head_yaw
    rel_head_pitch = primary.head_pitch_deg - center_head_pitch

    half_safe_x = config.half_safe_width_cm
    if hit.hit_x_cm < -half_safe_x:
        label = (
            "MENOLEH KE KIRI"
            if rel_head_yaw <= -config.head_label_yaw_deg
            else "MELIRIK MATA KE KIRI"
        )
        return IndicationCode.IND_01, label

    if hit.hit_x_cm > half_safe_x:
        label = (
            "MENOLEH KE KANAN"
            if rel_head_yaw >= config.head_label_yaw_deg
            else "MELIRIK MATA KE KANAN"
        )
        return IndicationCode.IND_01, label

    if hit.screen_y_cm < -config.edge_margin_cm:
        label = (
            "MENDONGAK (LIHAT ATAS)"
            if rel_head_pitch <= -config.head_label_pitch_deg
            else "MELIRIK MATA KE ATAS"
        )
        return IndicationCode.IND_02, label

    if hit.on_keyboard_deck and hit.deck_z_cm >= config.touchpad_end_depth_cm:
        label = (
            "MENUNDUK (LEWAT TOUCHPAD)"
            if rel_head_pitch >= config.head_label_pitch_deg
            else "MELIRIK BAWAH (LEWAT TOUCHPAD)"
        )
        return IndicationCode.IND_02, label

    return IndicationCode.NORMAL, "PESERTA FOKUS UJIAN"


def step_temporal_monitor(
    state: MonitoringState,
    instant_code: IndicationCode,
    instant_label: str,
    now_monotonic: float,
    config: DetectorConfig,
) -> TemporalStepResult:
    """
    Memperbarui state analisis durasi temporal dengan Calibration Gate,
    Grace Period anti-flicker (0.3s), dan Cooldown penyimpanan bukti (5.0s).
    """
    if not state.is_calibrated:
        state.clear_active_behavior()
        prompt = (
            state.last_calibration_error
            or "TEKAN [SPASI/C]: TATAP LENSA KAMERA FISIK (ANTI-SPOOF)"
        )
        return TemporalStepResult(
            banner_state=BannerState.UNCALIBRATED,
            banner_text=prompt,
            effective_code=IndicationCode.NORMAL,
            effective_label="BELUM KALIBRASI",
            elapsed_sec=0.0,
            should_log_evidence=False,
        )

    if instant_code == IndicationCode.NORMAL:
        if state.active_code != IndicationCode.NORMAL and state.behavior_start_time is not None:
            if state.normal_since_time is None:
                state.normal_since_time = now_monotonic
            if (now_monotonic - state.normal_since_time) >= config.grace_period_sec:
                state.clear_active_behavior()
        else:
            state.clear_active_behavior()

        return TemporalStepResult(
            banner_state=BannerState.SAFE,
            banner_text="STATUS AMAN: PESERTA FOKUS UJIAN",
            effective_code=IndicationCode.NORMAL,
            effective_label="PESERTA FOKUS UJIAN",
            elapsed_sec=0.0,
            should_log_evidence=False,
        )

    state.normal_since_time = None
    if state.active_code != instant_code or state.behavior_start_time is None:
        state.active_code = instant_code
        state.active_label = instant_label
        state.behavior_start_time = now_monotonic
    else:
        state.active_label = instant_label

    elapsed_sec = max(0.0, now_monotonic - state.behavior_start_time)
    limit_sec = config.threshold_for(state.active_code, state.active_label)

    if elapsed_sec < limit_sec:
        return TemporalStepResult(
            banner_state=BannerState.WARNING,
            banner_text=(
                f"MEMANTAU: {state.active_label} ({elapsed_sec:.1f}s / {limit_sec:.1f}s)"
            ),
            effective_code=state.active_code,
            effective_label=state.active_label,
            elapsed_sec=elapsed_sec,
            should_log_evidence=False,
        )

    cooldown_ready = (
        state.last_logged_time is None
        or (now_monotonic - state.last_logged_time) >= config.log_cooldown_sec
    )
    should_log = False
    if cooldown_ready:
        state.trust_score = max(0.0, state.trust_score - config.score_penalty)
        state.total_flags += 1
        state.last_logged_time = now_monotonic
        should_log = True

    return TemporalStepResult(
        banner_state=BannerState.ALERT,
        banner_text=(
            f"INDIKASI [{state.active_code.value}]: {state.active_label} ({elapsed_sec:.1f}s)"
        ),
        effective_code=state.active_code,
        effective_label=state.active_label,
        elapsed_sec=elapsed_sec,
        should_log_evidence=should_log,
    )


class FaceMeshDetector:
    """Wrapper MediaPipe FaceLandmarker (RunningMode.VIDEO + 4x4 3D Metric Matrix)."""

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._last_timestamp_ms: int = -1
        model_file = ensure_landmarker_model(config.model_path, config.model_url)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_file)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=config.max_num_faces,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def detect_faces(self, enhanced_bgr: UInt8Image, timestamp_ms: int) -> list[DetectedFace]:
        """Mendeteksi dan mengurutkan wajah beserta matriks 3D & iris dari frame CLAHE."""
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb_frame = np.ascontiguousarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        height, width, _ = enhanced_bgr.shape
        return extract_faces_sorted_by_area(
            result.face_landmarks,
            width,
            height,
            transformation_matrices=result.facial_transformation_matrixes,
            config=self._config,
        )

    def close(self) -> None:
        """Menutup resource native MediaPipe FaceLandmarker."""
        self._landmarker.close()
