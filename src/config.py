"""Konfigurasi sistem, Enum kode indikasi, geometri fisik laptop 3D, & state Anti-Spoof."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

FACE_LANDMARKER_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH: Path = Path("models") / "face_landmarker.task"
DEFAULT_EVIDENCE_DIR: Path = Path("bukti_indikasi")
DEFAULT_LOG_FILE: Path = Path("log_indikasi_ujian.csv")


class IndicationCode(StrEnum):
    """Kode standar indikasi perilaku ujian sesuai spesifikasi PRD & AGENTS.md."""

    NORMAL = "NORMAL"
    IND_01 = "IND-01"  # Penyimpangan Horizontal (>1 cm di luar tepi kiri/kanan layar)
    IND_02 = "IND-02"  # Penyimpangan Vertikal (>1 cm di atas bezel / >=21 cm akhir touchpad)
    IND_03 = "IND-03"  # Terdeteksi >1 Orang
    IND_04 = "IND-04"  # Wajah Tidak Terdeteksi
    IND_05 = "IND-05"  # Kamera Tertutup / Gelap / Oklusi Polos


class BannerState(StrEnum):
    """Kategori visual banner status pada dashboard."""

    UNCALIBRATED = "UNCALIBRATED"
    CALIBRATING = "CALIBRATING"
    SAFE = "SAFE"
    WARNING = "WARNING"
    ALERT = "ALERT"


class IntegrityLevel(StrEnum):
    """Kategori eskalasi skor kepercayaan untuk pengawas ujian."""

    AMAN = "AMAN"
    WASPADA = "WASPADA"
    TINJAUAN_PENGAWAS = "TINJAUAN PENGAWAS"


@dataclass(frozen=True)
class CalibrationSample:
    """Satu sampel metrik 3D wajah & iris mentah untuk validasi Anti-Spoof kalibrasi."""

    eye_x_cm: float
    eye_y_cm: float
    eye_z_cm: float
    head_yaw_deg: float
    head_pitch_deg: float
    gaze_x: float
    gaze_y: float
    ear: float


@dataclass(frozen=True)
class DetectorConfig:
    """Parameter pipeline PCD, geometri fisik 3D laptop (Lenovo Slim 5 14"), & Anti-Spoof."""

    frame_width: int = 640
    frame_height: int = 480
    roi_size: int = 110

    # Parameter Pengolahan Citra Digital (PCD)
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    gaussian_kernel_size: tuple[int, int] = (5, 5)
    canny_threshold1: int = 50
    canny_threshold2: int = 130
    min_luminance: float = 35.0
    min_global_edge_density: float = 1.0

    # Geometri Fisik Terkunci — Lenovo IdeaPad Slim 5 14IMH9 (14.0" 16:10 + 21 cm Touchpad)
    # Origin (0, 0, 0) berada tepat pada Lensa Fisik Webcam di tengah bezel atas layar.
    screen_width_cm: float = 30.16
    screen_height_cm: float = 18.85
    hinge_y_cm: float = 20.20
    touchpad_end_depth_cm: float = 21.00
    edge_margin_cm: float = 1.00
    lid_angle_deg: float = 90.0

    # Parameter MediaPipe 3D Pose, Iris Socket Angular Gain, & EMA
    max_num_faces: int = 3
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    camera_fov_deg: float = 65.0
    avg_face_width_cm: float = 14.8
    ema_alpha: float = 0.42
    neutral_gaze_x: float = 0.520
    neutral_gaze_y: float = -0.130
    eye_yaw_gain_deg: float = 320.0
    eye_pitch_up_gain_deg: float = 240.0
    eye_pitch_down_gain_deg: float = 260.0
    head_label_yaw_deg: float = 15.0
    head_label_pitch_deg: float = 14.0
    min_ear_for_gaze: float = 0.12

    # Parameter Kalibrasi 1-Titik Lensa Kamera & 5-Layer Anti-Spoof Gate
    calibration_duration_sec: float = 2.0
    calibration_sample_window_sec: float = 0.8
    min_valid_calib_samples: int = 5
    min_calib_distance_cm: float = 25.0
    max_calib_distance_cm: float = 85.0
    min_calib_ear: float = 0.16
    min_calib_gaze_y: float = -0.200
    max_calib_gaze_y: float = -0.065
    max_calib_head_yaw_error_deg: float = 10.0
    min_calib_head_pitch_error_deg: float = -24.0  # Mengizinkan kemiringan engsel layar 90-115 deg
    max_calib_head_pitch_error_deg: float = 12.0
    max_calib_offset_yaw_deg: float = 2.5  # Sumbu horizontal laptop simetris fisik X=0, tidak boleh dibelokkan
    min_calib_offset_pitch_deg: float = -12.0
    max_calib_offset_pitch_deg: float = 24.0
    max_calib_std_yaw_deg: float = 3.5
    max_calib_std_gaze_x: float = 0.025

    # Parameter Temporal (Detik) & Penilaian Trust Score
    thresh_yaw_sec: float = 2.5
    thresh_pitch_sec: float = 3.0
    thresh_gaze_sec: float = 2.5
    thresh_multiface_sec: float = 1.0
    thresh_absence_sec: float = 3.0
    thresh_dark_sec: float = 2.0
    grace_period_sec: float = 0.3
    log_cooldown_sec: float = 5.0
    score_penalty: float = 5.0

    # Path & Resource
    model_path: Path = DEFAULT_MODEL_PATH
    model_url: str = FACE_LANDMARKER_MODEL_URL
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR
    log_file: Path = DEFAULT_LOG_FILE

    @property
    def half_safe_width_cm(self) -> float:
        """Setengah lebar zona aman horizontal (setengah lebar layar + 1.0 cm)."""
        return (self.screen_width_cm * 0.5) + self.edge_margin_cm

    def threshold_for(self, code: IndicationCode, behavior_label: str = "") -> float:
        """Mengembalikan ambang durasi (detik) berdasarkan kode dan subtipe perilaku."""
        if "MELIRIK" in behavior_label:
            return self.thresh_gaze_sec

        thresholds: dict[IndicationCode, float] = {
            IndicationCode.IND_01: self.thresh_yaw_sec,
            IndicationCode.IND_02: self.thresh_pitch_sec,
            IndicationCode.IND_03: self.thresh_multiface_sec,
            IndicationCode.IND_04: self.thresh_absence_sec,
            IndicationCode.IND_05: self.thresh_dark_sec,
        }
        return thresholds.get(code, self.thresh_pitch_sec)


@dataclass
class MonitoringState:
    """State dinamis sesi pemantauan dengan koreksi anatomi Lensa Kamera Anti-Spoof."""

    calib_yaw_offset_deg: float = 0.0
    calib_pitch_offset_deg: float = 0.0
    smoothed_pose: tuple[float, float, float, float, float, float, float] | None = None
    is_calibrated: bool = False
    calibration_deadline: float | None = None
    calibration_samples: list[CalibrationSample] = field(default_factory=list)
    last_calibration_error: str | None = None
    trust_score: float = 100.0
    total_flags: int = 0
    active_code: IndicationCode = IndicationCode.NORMAL
    active_label: str = "NORMAL"
    behavior_start_time: float | None = None
    normal_since_time: float | None = None
    last_logged_time: float | None = None

    @property
    def is_calibrating(self) -> bool:
        """Bernilai True apabila sedang dalam hitung mundur kalibrasi lensa kamera."""
        return self.calibration_deadline is not None

    def start_lens_calibration(self, now_monotonic: float, duration_sec: float) -> None:
        """Memulai kalibrasi 1 titik menatap lensa fisik webcam (Anti-Spoof)."""
        self.calibration_deadline = now_monotonic + duration_sec
        self.calibration_samples.clear()
        self.last_calibration_error = None
        self.clear_active_behavior()

    def cancel_calibration(self, error_message: str | None = None) -> None:
        """Membatalkan atau menolak kalibrasi tanpa mengaktifkan sesi pemantauan."""
        self.calibration_deadline = None
        self.calibration_samples.clear()
        self.last_calibration_error = error_message

    def apply_lens_calibration(
        self,
        yaw_offset_deg: float,
        pitch_offset_deg: float,
        config: DetectorConfig | None = None,
    ) -> None:
        """Menyimpan koreksi sudut anatomi mata tervalidasi (terkunci dalam batas Anti-Spoof)."""
        cfg = config or DetectorConfig()
        self.calib_yaw_offset_deg = max(
            -cfg.max_calib_offset_yaw_deg,
            min(cfg.max_calib_offset_yaw_deg, yaw_offset_deg),
        )
        self.calib_pitch_offset_deg = max(
            cfg.min_calib_offset_pitch_deg,
            min(cfg.max_calib_offset_pitch_deg, pitch_offset_deg),
        )
        self.calibration_deadline = None
        self.calibration_samples.clear()
        self.last_calibration_error = None
        self.is_calibrated = True
        self.reset_ema()
        self.clear_active_behavior()

    def update_ema_pose(
        self,
        raw_pose: tuple[float, float, float, float, float, float, float],
        alpha: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        """Menghaluskan 7 sinyal geometri 3D wajah & iris dengan Exponential Moving Average."""
        if self.smoothed_pose is None:
            self.smoothed_pose = raw_pose
            return raw_pose

        inv = 1.0 - alpha
        smoothed = (
            alpha * raw_pose[0] + inv * self.smoothed_pose[0],
            alpha * raw_pose[1] + inv * self.smoothed_pose[1],
            alpha * raw_pose[2] + inv * self.smoothed_pose[2],
            alpha * raw_pose[3] + inv * self.smoothed_pose[3],
            alpha * raw_pose[4] + inv * self.smoothed_pose[4],
            alpha * raw_pose[5] + inv * self.smoothed_pose[5],
            alpha * raw_pose[6] + inv * self.smoothed_pose[6],
        )
        self.smoothed_pose = smoothed
        return smoothed

    def reset_ema(self) -> None:
        """Mereset riwayat filter EMA saat wajah hilang dari frame."""
        self.smoothed_pose = None

    def clear_active_behavior(self) -> None:
        """Mereset status perilaku mencurigakan kembali ke kondisi NORMAL."""
        self.active_code = IndicationCode.NORMAL
        self.active_label = "NORMAL"
        self.behavior_start_time = None
        self.normal_since_time = None

    def reset_session(self) -> None:
        """Mereset skor kepercayaan, jumlah indikasi, dan timer tanpa menghapus kalibrasi."""
        self.trust_score = 100.0
        self.total_flags = 0
        self.last_logged_time = None
        self.clear_active_behavior()


def resolve_integrity_level(trust_score: float) -> IntegrityLevel:
    """Menentukan kategori status integritas berdasarkan nilai Trust Score."""
    if trust_score >= 80.0:
        return IntegrityLevel.AMAN
    if trust_score >= 50.0:
        return IntegrityLevel.WASPADA
    return IntegrityLevel.TINJAUAN_PENGAWAS
