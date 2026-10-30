"""Pipeline Pengolahan Citra Digital (PCD): Luminansi, CIE-LAB CLAHE, Gaussian, Otsu, dan Canny."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import DetectorConfig

UInt8Image = NDArray[np.uint8]


@dataclass(frozen=True)
class FramePreprocessResult:
    """Hasil pra-pemrosesan citra global untuk satu frame video."""

    enhanced_bgr: UInt8Image
    mean_luminance: float
    global_edge_density: float


@dataclass(frozen=True)
class ROIPCDResult:
    """Hasil pemrosesan bertahap PCD pada Region of Interest (ROI) wajah utama."""

    vis_clahe: UInt8Image
    vis_gaussian: UInt8Image
    vis_otsu: UInt8Image
    vis_canny: UInt8Image
    roi_edge_density: float


def create_clahe(config: DetectorConfig) -> cv2.CLAHE:
    """Membuat objek CLAHE sesuai parameter konfigurasi PCD."""
    return cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )


def compute_edge_density_percent(edge_mask: UInt8Image) -> float:
    """Menghitung persentase piksel tepi aktif (non-zero) terhadap total piksel."""
    if edge_mask.size == 0:
        return 0.0
    return float((np.count_nonzero(edge_mask) / edge_mask.size) * 100.0)


def preprocess_frame_lab_clahe(
    frame_bgr: UInt8Image,
    clahe: cv2.CLAHE,
    config: DetectorConfig,
) -> FramePreprocessResult:
    """
    Menjalankan tahapan pra-pemrosesan global PCD:
    1. Konversi BGR ke Grayscale & perhitungan rata-rata luminansi global (Y).
    2. Filtering Gaussian (5x5) + Canny Edge global untuk deteksi oklusi benda terang/polos.
    3. Konversi BGR ke CIE-LAB dan penerapan CLAHE pada kanal Lightness (L) sebelum MediaPipe.
    """
    gray_global = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_luminance = float(np.mean(gray_global))

    blurred_global = cv2.GaussianBlur(gray_global, config.gaussian_kernel_size, 0)
    edges_global = cv2.Canny(
        blurred_global,
        config.canny_threshold1,
        config.canny_threshold2,
    )
    global_edge_density = compute_edge_density_percent(edges_global)

    lab_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return FramePreprocessResult(
        enhanced_bgr=enhanced_bgr,
        mean_luminance=mean_luminance,
        global_edge_density=global_edge_density,
    )


def process_roi_pcd_panel(
    roi_bgr: UInt8Image | None,
    clahe: cv2.CLAHE,
    config: DetectorConfig,
) -> ROIPCDResult:
    """
    Memproses ROI wajah utama (110x110 piksel) melalui tahapan wajib PCD:
    1. Grayscale + CLAHE
    2. Konvolusi Spasial Gaussian Blur (5x5)
    3. Segmentasi Biner Otsu Thresholding
    4. Deteksi Tepi Canny (50, 130) + perhitungan persentase Edge Density
    """
    size = config.roi_size
    if roi_bgr is None or roi_bgr.size == 0:
        blank = np.zeros((size, size, 3), dtype=np.uint8)
        return ROIPCDResult(
            vis_clahe=blank,
            vis_gaussian=blank,
            vis_otsu=blank,
            vis_canny=blank,
            roi_edge_density=0.0,
        )

    roi_resized = cv2.resize(roi_bgr, (size, size))
    gray_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    gray_clahe = clahe.apply(gray_roi)
    gaussian_blurred = cv2.GaussianBlur(gray_clahe, config.gaussian_kernel_size, 0)
    _, otsu_binary = cv2.threshold(
        gaussian_blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    canny_edges = cv2.Canny(
        gaussian_blurred,
        config.canny_threshold1,
        config.canny_threshold2,
    )
    roi_edge_density = compute_edge_density_percent(canny_edges)

    return ROIPCDResult(
        vis_clahe=cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR),
        vis_gaussian=cv2.cvtColor(gaussian_blurred, cv2.COLOR_GRAY2BGR),
        vis_otsu=cv2.cvtColor(otsu_binary, cv2.COLOR_GRAY2BGR),
        vis_canny=cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR),
        roi_edge_density=roi_edge_density,
    )

