"""Konfigurasi sistem, Enum kode indikasi/kalibrasi 5 titik, dan Dataclass state pemantauan."""

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
    IND_02 = "IND-02"  # Penyimpangan Vertikal (>1 cm di atas layar / mencapai ujung akhir touchpad)
    IND_03 = "IND-03"  # Terdeteksi >1 Orang
    IND_04 = "IND-04"  # Wajah Tidak Terdeteksi
    IND_05 = "IND-05"  # Kamera Tertutup / Gelap / Oklusi Polos


class CalibrationStage(StrEnum):
    """Lima tahapan kalibrasi batas fisik laptop secara berurutan."""

    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    TOP = "TOP"
    TOUCHPAD = "TOUCHPAD"


CALIBRATION_SEQUENCE: tuple[CalibrationStage, ...] = (
    CalibrationStage.CENTER,
    CalibrationStage.LEFT,
    CalibrationStage.RIGHT,
    CalibrationStage.TOP,
    CalibrationStage.TOUCHPAD,
)


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
class SafeZoneBounds:
    """Batas koordinat gabungan kepala + mata (Point-of-Regard) hasil kalibrasi 5 titik fisik."""

    min_por_x: float  # Batas ~1 cm di luar tepi kiri fisik layar
    max_por_x: float  # Batas ~1 cm di luar tepi kanan fisik layar
    min_por_y: float  # Batas ~1 cm di atas bezel atas / webcam
    max_por_y: float  # Batas tepat di titik akhir bawah touchpad


@dataclass(frozen=True)
class DetectorConfig:
    """Parameter konfigurasi pengolahan citra, geometri Point-of-Regard, EMA, dan aturan temporal."""

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

    # Parameter Geometri Kepala + Iris Unified Point-of-Regard (POR)
    max_num_faces: int = 3
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    ema_alpha: float = 0.45
    yaw_to_gaze_gain: float = 0.55
    pitch_to_gaze_gain: float = 0.85
    head_label_yaw_delta: float = 0.12
    head_label_pitch_delta: float = 0.08
    min_ear_for_gaze: float = 0.09

    # Fallback default apabila menggunakan kalibrasi 1 titik pada unit test
    tol_yaw: float = 0.15
    tol_pitch: float = 0.10
    tol_gaze_x: float = 0.11
    tol_gaze_up: float = 0.08
    tol_gaze_down: float = 0.11

    # Parameter Kalibrasi 5 Titik Batas Fisik (~1 cm luar layar & tepat akhir touchpad)
    calibration_stage_sec: float = 2.0
    calibration_sample_window_sec: float = 0.8
    screen_edge_margin_ratio_x: float = 0.06  # ~1 cm pada setengah lebar layar ~17 cm
    screen_edge_margin_ratio_top: float = 0.10  # ~1 cm pada jarak tengah-ke-atas ~10 cm
    touchpad_end_offset: float = -0.002  # Langsung kuning tepat di titik akhir touchpad

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
    """State dinamis sesi pemantauan peserta termasuk kalibrasi 5 titik & batas SafeZoneBounds."""

    baseline_yaw: float = 0.50
    baseline_pitch: float = 0.35
    baseline_gaze_x: float = 0.50
    baseline_gaze_y: float = -0.11
    safe_bounds: SafeZoneBounds | None = None
    smoothed_yaw: float | None = None
    smoothed_pitch: float | None = None
    smoothed_gaze_x: float | None = None
    smoothed_gaze_y: float | None = None
    is_calibrated: bool = False
    calibration_stage_idx: int | None = None
    calibration_deadline: float | None = None
    calibration_samples: list[tuple[float, float, float, float]] = field(default_factory=list)
    stage_averages: dict[CalibrationStage, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    trust_score: float = 100.0
    total_flags: int = 0
    active_code: IndicationCode = IndicationCode.NORMAL
    active_label: str = "NORMAL"
    behavior_start_time: float | None = None
    normal_since_time: float | None = None
    last_logged_time: float | None = None

    @property
    def is_calibrating(self) -> bool:
        """Bernilai True apabila sedang dalam proses kalibrasi 5 titik."""
        return self.calibration_stage_idx is not None and self.calibration_deadline is not None

    @property
    def current_calibration_stage(self) -> CalibrationStage | None:
        """Mengembalikan tahapan kalibrasi aktif saat ini (CENTER/LEFT/RIGHT/TOP/TOUCHPAD)."""
        if self.calibration_stage_idx is None:
            return None
        if 0 <= self.calibration_stage_idx < len(CALIBRATION_SEQUENCE):
            return CALIBRATION_SEQUENCE[self.calibration_stage_idx]
        return None

    def start_five_point_calibration(self, now_monotonic: float, stage_sec: float) -> None:
        """Memulai rangkaian kalibrasi 5 titik fisik dari tahap pertama (CENTER)."""
        self.calibration_stage_idx = 0
        self.calibration_deadline = now_monotonic + stage_sec
        self.calibration_samples.clear()
        self.stage_averages.clear()
        self.clear_active_behavior()

    def cancel_calibration(self) -> None:
        """Membatalkan proses kalibrasi apabila wajah hilang atau >1 orang."""
        self.calibration_stage_idx = None
        self.calibration_deadline = None
        self.calibration_samples.clear()
        self.stage_averages.clear()

    def update_ema(
        self,
        yaw: float,
        pitch: float,
        gaze_x: float,
        gaze_y: float,
        alpha: float,
    ) -> tuple[float, float, float, float]:
        """Menghaluskan sinyal geometri kepala dan iris menggunakan Exponential Moving Average."""
        if (
            self.smoothed_yaw is None
            or self.smoothed_pitch is None
            or self.smoothed_gaze_x is None
            or self.smoothed_gaze_y is None
        ):
            self.smoothed_yaw = yaw
            self.smoothed_pitch = pitch
            self.smoothed_gaze_x = gaze_x
            self.smoothed_gaze_y = gaze_y
        else:
            inv = 1.0 - alpha
            self.smoothed_yaw = alpha * yaw + inv * self.smoothed_yaw
            self.smoothed_pitch = alpha * pitch + inv * self.smoothed_pitch
            self.smoothed_gaze_x = alpha * gaze_x + inv * self.smoothed_gaze_x
            self.smoothed_gaze_y = alpha * gaze_y + inv * self.smoothed_gaze_y

        return (
            self.smoothed_yaw,
            self.smoothed_pitch,
            self.smoothed_gaze_x,
            self.smoothed_gaze_y,
        )

    def reset_ema(self) -> None:
        """Mereset riwayat filter EMA saat wajah hilang dari frame."""
        self.smoothed_yaw = None
        self.smoothed_pitch = None
        self.smoothed_gaze_x = None
        self.smoothed_gaze_y = None

    def apply_calibrated_bounds(
        self,
        yaw: float,
        pitch: float,
        gaze_x: float,
        gaze_y: float,
        bounds: SafeZoneBounds,
    ) -> None:
        """Menyimpan titik tengah dan batas zona aman 4 sisi hasil kalibrasi 5 titik."""
        self.baseline_yaw = yaw
        self.baseline_pitch = pitch
        self.baseline_gaze_x = gaze_x
        self.baseline_gaze_y = gaze_y
        self.safe_bounds = bounds
        self.smoothed_yaw = yaw
        self.smoothed_pitch = pitch
        self.smoothed_gaze_x = gaze_x
        self.smoothed_gaze_y = gaze_y
        self.cancel_calibration()
        self.is_calibrated = True
        self.clear_active_behavior()

    def calibrate(
        self,
        yaw: float,
        pitch: float,
        gaze_x: float = 0.50,
        gaze_y: float = -0.11,
        config: DetectorConfig | None = None,
    ) -> None:
        """Kalibrasi langsung 1 titik (membentuk batas default simetris di sekitar titik acuan)."""
        cfg = config or DetectorConfig()
        bounds = SafeZoneBounds(
            min_por_x=gaze_x - cfg.tol_gaze_x,
            max_por_x=gaze_x + cfg.tol_gaze_x,
            min_por_y=gaze_y - cfg.tol_gaze_up,
            max_por_y=gaze_y + cfg.tol_gaze_down,
        )
        self.apply_calibrated_bounds(yaw, pitch, gaze_x, gaze_y, bounds)

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
