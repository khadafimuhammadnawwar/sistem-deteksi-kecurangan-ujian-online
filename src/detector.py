"""Detektor MediaPipe Tasks FaceLandmarker, Unified Point-of-Regard (POR), & kalibrasi 5 titik."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
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
    CALIBRATION_SEQUENCE,
    BannerState,
    CalibrationStage,
    DetectorConfig,
    IndicationCode,
    MonitoringState,
    SafeZoneBounds,
)

UInt8Image = NDArray[np.uint8]

# Indeks kanonik 478 titik landmark MediaPipe FaceLandmarker (Kepala & Iris)
NOSE_TIP_IDX: int = 1
LEFT_CHEEK_IDX: int = 234
RIGHT_CHEEK_IDX: int = 454
CHIN_IDX: int = 152

# Mata sisi kiri citra tercermin (Anatomical Right Eye: sudut 33-133, kelopak 159-145, pusat iris 468)
R_EYE_OUTER_IDX: int = 33
R_EYE_INNER_IDX: int = 133
R_EYE_TOP_IDX: int = 159
R_EYE_BOTTOM_IDX: int = 145
R_IRIS_CENTER_IDX: int = 468

# Mata sisi kanan citra tercermin (Anatomical Left Eye: sudut 362-263, kelopak 386-374, pusat iris 473)
L_EYE_INNER_IDX: int = 362
L_EYE_OUTER_IDX: int = 263
L_EYE_TOP_IDX: int = 386
L_EYE_BOTTOM_IDX: int = 374
L_IRIS_CENTER_IDX: int = 473

LEFT_EYE_OUTER_IDX: int = R_EYE_OUTER_IDX
RIGHT_EYE_OUTER_IDX: int = L_EYE_OUTER_IDX

BBOX_PADDING_PX: int = 15

STAGE_PROMPTS: dict[CalibrationStage, str] = {
    CalibrationStage.CENTER: "1/5 TATAP TENGAH LAYAR FISIK LAPTOP",
    CalibrationStage.LEFT: "2/5 TATAP BINGKAI TEPI KIRI LAYAR",
    CalibrationStage.RIGHT: "3/5 TATAP BINGKAI TEPI KANAN LAYAR",
    CalibrationStage.TOP: "4/5 TATAP BEZEL ATAS LAYAR / WEBCAM",
    CalibrationStage.TOUCHPAD: "5/5 TATAP TITIK AKHIR BAWAH TOUCHPAD",
}


class LandmarkPoint(Protocol):
    """Kontrak koordinat ternormalisasi landmark wajah (0.0 - 1.0)."""

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...


@dataclass(frozen=True)
class DetectedFace:
    """Data geometri kepala, iris mata, dan bounding box satu wajah dalam frame."""

    bbox: tuple[int, int, int, int]
    nose_px: tuple[int, int]
    iris_px: tuple[tuple[int, int], tuple[int, int]]
    area: int
    yaw: float
    pitch: float
    gaze_x: float = 0.50
    gaze_y: float = -0.11
    ear: float = 0.30


@dataclass(frozen=True)
class TemporalStepResult:
    """Keputusan evaluasi temporal untuk satu siklus frame."""

    banner_state: BannerState
    banner_text: str
    effective_code: IndicationCode
    effective_label: str
    elapsed_sec: float
    should_log_evidence: bool
    calibration_event: str | None = None  # "NEXT_STAGE" atau "COMPLETED" untuk sinyal suara


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


def calculate_yaw_pitch(landmarks: Sequence[LandmarkPoint]) -> tuple[float, float]:
    """Menghitung rasio geometri kepala Yaw (horizontal) dan Pitch (vertikal)."""
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
        return 0.50, -0.11, 0.30

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


def compute_point_of_regard(
    yaw: float,
    pitch: float,
    gaze_x: float,
    gaze_y: float,
    ear: float,
    baseline_yaw: float,
    baseline_pitch: float,
    baseline_gaze_x: float,
    baseline_gaze_y: float,
    config: DetectorConfig,
) -> tuple[float, float]:
    """
    Menghitung koordinat pandangan gabungan kepala + bola mata (Unified Point-of-Regard):
    - POR_x = Gaze_x + k_x * (Yaw - Yaw_center)
    - POR_y = Gaze_y + k_y * (Pitch - Pitch_center)
    Saat kelopak mata tertutup rapat karena berkedip (EAR < min_ear_for_gaze), posisi iris
    diganti baseline_gaze agar kedipan tidak menimbulkan lonjakan namun gerakan kepala tetap aktif.
    """
    eff_gx = gaze_x if ear >= config.min_ear_for_gaze else baseline_gaze_x
    eff_gy = gaze_y if ear >= config.min_ear_for_gaze else baseline_gaze_y

    por_x = eff_gx + config.yaw_to_gaze_gain * (yaw - baseline_yaw)
    por_y = eff_gy + config.pitch_to_gaze_gain * (pitch - baseline_pitch)
    return float(por_x), float(por_y)


def build_safe_zone_bounds(
    stage_averages: dict[CalibrationStage, tuple[float, float, float, float]],
    config: DetectorConfig,
) -> tuple[tuple[float, float, float, float], SafeZoneBounds]:
    """
    Menghitung batas zona aman 4 sisi dari hasil rata-rata 5 titik fisik laptop:
    - Kiri, Kanan, dan Atas diberi margin proporsional ~1 cm di luar bingkai layar.
    - Bawah dikunci tepat pada titik akhir bawah touchpad.
    """
    c_yaw, c_pitch, c_gx, c_gy = stage_averages[CalibrationStage.CENTER]

    def _stage_por(stage: CalibrationStage) -> tuple[float, float]:
        s_yaw, s_pitch, s_gx, s_gy = stage_averages[stage]
        return (
            s_gx + config.yaw_to_gaze_gain * (s_yaw - c_yaw),
            s_gy + config.pitch_to_gaze_gain * (s_pitch - c_pitch),
        )

    left_por_x, _ = _stage_por(CalibrationStage.LEFT)
    right_por_x, _ = _stage_por(CalibrationStage.RIGHT)
    _, top_por_y = _stage_por(CalibrationStage.TOP)
    _, touchpad_por_y = _stage_por(CalibrationStage.TOUCHPAD)

    span_left = max(0.025, c_gx - left_por_x)
    span_right = max(0.025, right_por_x - c_gx)
    span_top = max(0.020, c_gy - top_por_y)
    span_bottom = max(0.030, touchpad_por_y - c_gy)

    margin_left = min(max(0.005, config.screen_edge_margin_ratio_x * span_left), 0.012)
    margin_right = min(max(0.005, config.screen_edge_margin_ratio_x * span_right), 0.012)
    margin_top = min(max(0.005, config.screen_edge_margin_ratio_top * span_top), 0.012)

    bounds = SafeZoneBounds(
        min_por_x=(c_gx - span_left) - margin_left,
        max_por_x=(c_gx + span_right) + margin_right,
        min_por_y=(c_gy - span_top) - margin_top,
        max_por_y=(c_gy + span_bottom) + config.touchpad_end_offset,
    )
    return (c_yaw, c_pitch, c_gx, c_gy), bounds


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
) -> list[DetectedFace]:
    """Mengekstrak geometri seluruh wajah dan mengurutkannya dari luas terbesar (peserta utama)."""
    detected_faces: list[DetectedFace] = []
    for landmarks in multi_face_landmarks:
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

        yaw, pitch = calculate_yaw_pitch(landmarks)
        gaze_x, gaze_y, ear = calculate_iris_gaze_and_ear(landmarks)

        detected_faces.append(
            DetectedFace(
                bbox=(x_min, y_min, x_max, y_max),
                nose_px=nose_px,
                iris_px=iris_px,
                area=area,
                yaw=yaw,
                pitch=pitch,
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
    """Menerapkan EMA smoothing pada orientasi kepala dan iris wajah peserta utama."""
    if not faces:
        state.reset_ema()
        return faces

    primary = faces[0]
    s_yaw, s_pitch, s_gx, s_gy = state.update_ema(
        yaw=primary.yaw,
        pitch=primary.pitch,
        gaze_x=primary.gaze_x,
        gaze_y=primary.gaze_y,
        alpha=alpha,
    )
    smoothed_primary = replace(
        primary,
        yaw=s_yaw,
        pitch=s_pitch,
        gaze_x=s_gx,
        gaze_y=s_gy,
    )
    return [smoothed_primary, *faces[1:]]


def advance_five_point_calibration(
    state: MonitoringState,
    raw_faces: Sequence[DetectedFace],
    now_monotonic: float,
    config: DetectorConfig,
) -> TemporalStepResult | None:
    """
    Menjalankan mesin tahapan kalibrasi 5 titik (CENTER -> LEFT -> RIGHT -> TOP -> TOUCHPAD)
    menggunakan sampel mentah bebas lag pada jendela 0.8 detik terakhir tiap tahap.
    """
    stage = state.current_calibration_stage
    if stage is None or state.calibration_deadline is None:
        return None

    if len(raw_faces) != 1:
        state.cancel_calibration()
        return TemporalStepResult(
            banner_state=BannerState.UNCALIBRATED,
            banner_text="KALIBRASI BATAL: PASTIKAN TEPAT 1 WAJAH DI KAMERA",
            effective_code=IndicationCode.NORMAL,
            effective_label="KALIBRASI BATAL",
            elapsed_sec=0.0,
            should_log_evidence=False,
        )

    remaining = state.calibration_deadline - now_monotonic
    primary = raw_faces[0]
    if remaining <= config.calibration_sample_window_sec:
        state.calibration_samples.append(
            (primary.yaw, primary.pitch, primary.gaze_x, primary.gaze_y)
        )

    prompt = STAGE_PROMPTS[stage]
    if remaining > 0.0:
        return TemporalStepResult(
            banner_state=BannerState.CALIBRATING,
            banner_text=f"{prompt} ({remaining:.1f}s)",
            effective_code=IndicationCode.NORMAL,
            effective_label="PROSES KALIBRASI",
            elapsed_sec=0.0,
            should_log_evidence=False,
        )

    samples = state.calibration_samples or [
        (primary.yaw, primary.pitch, primary.gaze_x, primary.gaze_y)
    ]
    n_samples = len(samples)
    avg_tuple = (
        sum(s[0] for s in samples) / n_samples,
        sum(s[1] for s in samples) / n_samples,
        sum(s[2] for s in samples) / n_samples,
        sum(s[3] for s in samples) / n_samples,
    )
    state.stage_averages[stage] = avg_tuple
    state.calibration_samples.clear()

    next_idx = (state.calibration_stage_idx or 0) + 1
    if next_idx < len(CALIBRATION_SEQUENCE):
        state.calibration_stage_idx = next_idx
        state.calibration_deadline = now_monotonic + config.calibration_stage_sec
        next_stage = CALIBRATION_SEQUENCE[next_idx]
        return TemporalStepResult(
            banner_state=BannerState.CALIBRATING,
            banner_text=f"{STAGE_PROMPTS[next_stage]} ({config.calibration_stage_sec:.1f}s)",
            effective_code=IndicationCode.NORMAL,
            effective_label="PROSES KALIBRASI",
            elapsed_sec=0.0,
            should_log_evidence=False,
            calibration_event="NEXT_STAGE",
        )

    center_vals, bounds = build_safe_zone_bounds(state.stage_averages, config)
    state.apply_calibrated_bounds(*center_vals, bounds)
    print(
        f"[KALIBRASI 5 TITIK SELESAI] "
        f"POR_X:[{bounds.min_por_x:.3f} .. {bounds.max_por_x:.3f}] | "
        f"POR_Y:[{bounds.min_por_y:+.3f} .. {bounds.max_por_y:+.3f}]"
    )
    return TemporalStepResult(
        banner_state=BannerState.SAFE,
        banner_text="KALIBRASI 5 TITIK BERHASIL: ZONA PRESISI AKTIF",
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
    baseline_yaw: float,
    baseline_pitch: float,
    config: DetectorConfig,
    baseline_gaze_x: float = 0.50,
    baseline_gaze_y: float = -0.11,
    safe_bounds: SafeZoneBounds | None = None,
) -> tuple[IndicationCode, str]:
    """
    Mengevaluasi indikasi instan satu frame menggunakan Unified Point-of-Regard (POR_x, POR_y)
    sehingga baik lirikan bola mata saja maupun gerakan kepala melewati batas 1 cm luar layar
    atau titik akhir touchpad langsung terdeteksi secara simetris.
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
    bounds = safe_bounds or SafeZoneBounds(
        min_por_x=baseline_gaze_x - config.tol_gaze_x,
        max_por_x=baseline_gaze_x + config.tol_gaze_x,
        min_por_y=baseline_gaze_y - config.tol_gaze_up,
        max_por_y=baseline_gaze_y + config.tol_gaze_down,
    )

    por_x, por_y = compute_point_of_regard(
        yaw=primary.yaw,
        pitch=primary.pitch,
        gaze_x=primary.gaze_x,
        gaze_y=primary.gaze_y,
        ear=primary.ear,
        baseline_yaw=baseline_yaw,
        baseline_pitch=baseline_pitch,
        baseline_gaze_x=baseline_gaze_x,
        baseline_gaze_y=baseline_gaze_y,
        config=config,
    )

    yaw_delta = primary.yaw - baseline_yaw
    pitch_delta = primary.pitch - baseline_pitch

    if por_x < bounds.min_por_x:
        label = (
            "MENOLEH KE KIRI"
            if yaw_delta <= -config.head_label_yaw_delta
            else "MELIRIK MATA KE KIRI"
        )
        return IndicationCode.IND_01, label

    if por_x > bounds.max_por_x:
        label = (
            "MENOLEH KE KANAN"
            if yaw_delta >= config.head_label_yaw_delta
            else "MELIRIK MATA KE KANAN"
        )
        return IndicationCode.IND_01, label

    if por_y < bounds.min_por_y:
        label = (
            "MENDONGAK (LIHAT ATAS)"
            if pitch_delta <= -config.head_label_pitch_delta
            else "MELIRIK MATA KE ATAS"
        )
        return IndicationCode.IND_02, label

    if por_y > bounds.max_por_y:
        label = (
            "MENUNDUK (LEWAT TOUCHPAD)"
            if pitch_delta >= config.head_label_pitch_delta
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
        return TemporalStepResult(
            banner_state=BannerState.UNCALIBRATED,
            banner_text="TEKAN [SPASI] ATAU [C] UNTUK KALIBRASI 5 TITIK LAPTOP",
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
    """Wrapper MediaPipe Tasks FaceLandmarker (RunningMode.VIDEO) dengan context manager."""

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
        """Mendeteksi dan mengurutkan wajah beserta iris dari frame BGR hasil CLAHE."""
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb_frame = np.ascontiguousarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        height, width, _ = enhanced_bgr.shape
        return extract_faces_sorted_by_area(result.face_landmarks, width, height)

    def close(self) -> None:
        """Menutup resource native MediaPipe FaceLandmarker."""
        self._landmarker.close()
