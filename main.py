"""Entry point utama akuisisi webcam & koordinasi loop pemantauan ujian online (Mode 1: Teori)."""

import time

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import DetectorConfig, MonitoringState
from src.detector import (
    DetectedFace,
    FaceMeshDetector,
    evaluate_instant_indication,
    step_temporal_monitor,
)
from src.logger import EvidenceLogger
from src.pcd_pipeline import create_clahe, preprocess_frame_lab_clahe, process_roi_pcd_panel
from src.ui_dashboard import CAMERA_PANEL_WIDTH, render_dashboard

UInt8Image = NDArray[np.uint8]
WINDOW_TITLE: str = "Sistem Deteksi Indikasi Kecurangan Ujian Online - PCD (Mode 1: Teori)"


def _extract_primary_roi(
    enhanced_bgr: UInt8Image,
    faces: list[DetectedFace],
) -> UInt8Image | None:
    """Memotong Region of Interest (ROI) wajah peserta utama (Wajah #1)."""
    if not faces:
        return None
    x_min, y_min, x_max, y_max = faces[0].bbox
    if x_max <= x_min or y_max <= y_min:
        return None
    return enhanced_bgr[y_min:y_max, x_min:x_max].copy()


def run_monitoring_session(camera_index: int = 0) -> None:
    """Menjalankan loop utama pemantauan webcam Mode 1 (CBT Paperless / Ujian Teori)."""
    config = DetectorConfig()
    state = MonitoringState()
    logger = EvidenceLogger(config.evidence_dir, config.log_file)
    clahe = create_clahe(config)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Tidak dapat mengakses kamera pada indeks {camera_index}. "
            "Pastikan webcam terhubung dan tidak sedang dipakai aplikasi lain."
        )

    print("=== SISTEM MONITORING UJIAN ONLINE (MODE 1: CBT PAPERLESS / TEORI) ===")
    print("Kontrol: [C] Kalibrasi Wajah (Wajib di Awal) | [R] Reset Skor & Timer | [Q] Keluar")

    try:
        with FaceMeshDetector(config) as detector:
            while cap.isOpened():
                ret, raw_frame = cap.read()
                if not ret:
                    print("[PERINGATAN] Gagal membaca frame dari kamera. Menghentikan sesi.")
                    break

                if (raw_frame.shape[1], raw_frame.shape[0]) != (
                    config.frame_width,
                    config.frame_height,
                ):
                    raw_frame = cv2.resize(
                        raw_frame,
                        (config.frame_width, config.frame_height),
                    )

                # 1. Akuisisi & Pencerminan Horizontal
                mirrored_frame = cv2.flip(raw_frame, 1)

                # 2. Pra-pemrosesan PCD Global (Luminansi, Canny Global, CIE-LAB CLAHE)
                frame_pcd = preprocess_frame_lab_clahe(mirrored_frame, clahe, config)

                # 3. Deteksi Landmark Wajah & Pengurutan Luas Bounding Box
                now_mono = time.monotonic()
                timestamp_ms = int(now_mono * 1000)
                faces = detector.detect_faces(frame_pcd.enhanced_bgr, timestamp_ms)

                # 4. Evaluasi Aturan Geometri & Durasi Temporal
                instant_code, instant_label = evaluate_instant_indication(
                    mean_luminance=frame_pcd.mean_luminance,
                    global_edge_density=frame_pcd.global_edge_density,
                    faces=faces,
                    baseline_yaw=state.baseline_yaw,
                    baseline_pitch=state.baseline_pitch,
                    config=config,
                )
                step_result = step_temporal_monitor(
                    state=state,
                    instant_code=instant_code,
                    instant_label=instant_label,
                    now_monotonic=now_mono,
                    config=config,
                )

                # 5. Pemrosesan Panel PCD 2x2 pada ROI Wajah Utama
                primary_roi = _extract_primary_roi(frame_pcd.enhanced_bgr, faces)
                roi_pcd = process_roi_pcd_panel(primary_roi, clahe, config)

                # 6. Render Dashboard Visual (920x480)
                dashboard = render_dashboard(
                    raw_frame_bgr=mirrored_frame,
                    faces=faces,
                    state=state,
                    step_result=step_result,
                    frame_pcd=frame_pcd,
                    roi_pcd=roi_pcd,
                    min_luminance=config.min_luminance,
                )

                # 7. Simpan Bukti Ber-anotasi & Log CSV jika Melewati Ambang Durasi
                if step_result.should_log_evidence:
                    logger.record_indication(
                        annotated_frame=dashboard[:, :CAMERA_PANEL_WIDTH],
                        code=step_result.effective_code,
                        behavior_label=step_result.effective_label,
                        duration_sec=step_result.elapsed_sec,
                        mean_luminance=frame_pcd.mean_luminance,
                        trust_score=state.trust_score,
                        flag_number=state.total_flags,
                    )

                cv2.imshow(WINDOW_TITLE, dashboard)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    if len(faces) == 1:
                        state.calibrate(faces[0].yaw, faces[0].pitch)
                        print(
                            f"[KALIBRASI BERHASIL] Baseline Yaw: {state.baseline_yaw:.3f} | "
                            f"Baseline Pitch: {state.baseline_pitch:.3f}"
                        )
                    else:
                        print(
                            f"[KALIBRASI GAGAL] Harus tepat 1 wajah di kamera "
                            f"(terdeteksi: {len(faces)})."
                        )
                elif key == ord("r"):
                    state.reset_session()
                    print("[RESET] Trust Score kembali 100% dan timer direset.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_monitoring_session()
