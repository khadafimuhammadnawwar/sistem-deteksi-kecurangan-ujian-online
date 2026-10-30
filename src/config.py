"""Konfigurasi sistem, Enum kode indikasi, dan Dataclass state pemantauan ujian."""

from dataclasses import dataclass
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
    """Kode standar indikasi perilaku ujian sesuai spesifikasi PRD."""

    NORMAL = "NORMAL"
    IND_01 = "IND-01"  # Menoleh Kiri / Kanan
    IND_02 = "IND-02"  # Menunduk / Mendongak Vertikal
    IND_03 = "IND-03"  # Terdeteksi >1 Orang
    IND_04 = "IND-04"  # Wajah Tidak Terdeteksi
    IND_05 = "IND-05"  # Kamera Tertutup / Gelap / Oklusi Polos


class BannerState(StrEnum):
    """Kategori visual banner status pada dashboard."""

    UNCALIBRATED = "UNCALIBRATED"
    SAFE = "SAFE"
    WARNING = "WARNING"
    ALERT = "ALERT"


class IntegrityLevel(StrEnum):
    """Kategori eskalasi skor kepercayaan untuk pengawas ujian."""

    AMAN = "AMAN"
    WASPADA = "WASPADA"
    TINJAUAN_PENGAWAS = "TINJAUAN PENGAWAS"


@dataclass(frozen=True)
class DetectorConfig:
    """Parameter konfigurasi pengolahan citra, geometri wajah, dan aturan temporal."""

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

    # Parameter Geometri Landmark Wajah
    max_num_faces: int = 3
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    tol_yaw: float = 0.15
    tol_pitch: float = 0.08

    # Parameter Temporal (Detik) & Penilaian Trust Score
    thresh_yaw_sec: float = 2.5
    thresh_pitch_sec: float = 3.0
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

    def threshold_for(self, code: IndicationCode) -> float:
        """Mengembalikan ambang durasi (detik) berdasarkan kode indikasi."""
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
    """State dinamis sesi pemantauan peserta ujian."""

    baseline_yaw: float = 0.50
    baseline_pitch: float = 0.38
    is_calibrated: bool = False
    trust_score: float = 100.0
    total_flags: int = 0
    active_code: IndicationCode = IndicationCode.NORMAL
    active_label: str = "NORMAL"
    behavior_start_time: float | None = None
    normal_since_time: float | None = None
    last_logged_time: float | None = None

    def calibrate(self, yaw: float, pitch: float) -> None:
        """Menetapkan titik acuan orientasi kepala peserta dan mereset timer."""
        self.baseline_yaw = yaw
        self.baseline_pitch = pitch
        self.is_calibrated = True
        self.clear_active_behavior()

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
