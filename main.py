"""Entry point utama akuisisi webcam & koordinasi loop pemantauan ujian online (Mode 1: Teori)."""

import threading
import time
import winsound

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import DetectorConfig, MonitoringState
from src.detector import (
    DetectedFace,
    FaceMeshDetector,
    advance_five_point_calibration,
    evaluate_instant_indication,
    smooth_primary_face_pose,
    step_temporal_monitor,
)
from src.logger import EvidenceLogger
from src.pcd_pipeline import create_clahe, preprocess_frame_lab_clahe, process_roi_pcd_panel
from src.ui_dashboard import CAMERA_PANEL_WIDTH, render_dashboard

UInt8Image = NDArray[np.uint8]
WINDOW_TITLE: str = "Sistem Deteksi Indikasi Kecurangan Ujian Online - PCD (Mode 1: Teori)"


def _play_calibration_cue(event: str | None) -> None:
    """Memainkan bunyi bip singkat non-blocking agar transisi tahap 1-5 terdengar jelas."""
    if event is None:
        return

    def _sound_worker() -> None:
        try:
            if event == "NEXT_STAGE":
                winsound.Beep(880, 90)
            elif event == "COMPLETED":
                winsound.Beep(1180, 110)
                winsound.Beep(1480, 160)
        except RuntimeError:
            pass

    threading.Thread(target=_sound_worker, daemon=True).start()


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
    print(
        "Kontrol: [SPASI / C] Kalibrasi 5 Titik Fisik Laptop | "
        "[R] Reset Skor & Timer | [Q] Keluar"
    )

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

                mirrored_frame = cv2.flip(raw_frame, 1)
                frame_pcd = preprocess_frame_lab_clahe(mirrored_frame, clahe, config)

                now_mono = time.monotonic()
                timestamp_ms = int(now_mono * 1000)
                raw_faces = detector.detect_faces(frame_pcd.enhanced_bgr, timestamp_ms)
                faces = smooth_primary_face_pose(raw_faces, state, config.ema_alpha)

                cal_step = advance_five_point_calibration(state, raw_faces, now_mono, config)
                if cal_step is not None:
                    step_result = cal_step
                    _play_calibration_cue(cal_step.calibration_event)
                else:
                    instant_code, instant_label = evaluate_instant_indication(
                        mean_luminance=frame_pcd.mean_luminance,
                        global_edge_density=frame_pcd.global_edge_density,
                        faces=faces,
                        baseline_yaw=state.baseline_yaw,
                        baseline_pitch=state.baseline_pitch,
                        config=config,
                        baseline_gaze_x=state.baseline_gaze_x,
                        baseline_gaze_y=state.baseline_gaze_y,
                        safe_bounds=state.safe_bounds,
                    )
                    step_result = step_temporal_monitor(
                        state=state,
                        instant_code=instant_code,
                        instant_label=instant_label,
                        now_monotonic=now_mono,
                        config=config,
                    )

                primary_roi = _extract_primary_roi(frame_pcd.enhanced_bgr, faces)
                roi_pcd = process_roi_pcd_panel(primary_roi, clahe, config)

                dashboard = render_dashboard(
                    raw_frame_bgr=mirrored_frame,
                    faces=faces,
                    state=state,
                    step_result=step_result,
                    frame_pcd=frame_pcd,
                    roi_pcd=roi_pcd,
                    min_luminance=config.min_luminance,
                )

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
                if key in (ord("c"), ord(" ")):
                    if len(raw_faces) == 1:
                        state.start_five_point_calibration(
                            now_mono, config.calibration_stage_sec
                        )
                        _play_calibration_cue("NEXT_STAGE")
                        print(
                            "[KALIBRASI 5 TITIK DIMULAI] "
                            "Ikuti 5 panduan arah: Tengah -> Kiri -> Kanan -> Atas -> Akhir Touchpad."
                        )
                    else:
                        print(
                            f"[KALIBRASI GAGAL] Harus tepat 1 wajah di kamera "
                            f"(terdeteksi: {len(raw_faces)})."
                        )
                elif key == ord("r"):
                    state.reset_session()
                    print("[RESET] Trust Score kembali 100% dan timer direset.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_monitoring_session()
