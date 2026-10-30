"""Entry point akuisisi webcam & koordinasi loop 3D Anti-Spoof Ray-Casting (Mode 1: Teori)."""

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
    advance_lens_calibration,
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
    """Memainkan sinyal suara non-blocking saat mulai, sukses, atau ditolak Anti-Spoof."""
    if event is None:
        return

    def _sound_worker() -> None:
        try:
            if event == "STARTED":
                winsound.Beep(880, 90)
            elif event == "COMPLETED":
                winsound.Beep(1180, 110)
                winsound.Beep(1480, 160)
            elif event == "REJECTED":
                winsound.Beep(320, 260)
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

    print("=== SISTEM MONITORING UJIAN ONLINE (MODE 1: 3D HARDWARE-LOCKED RAY-CASTING) ===")
    print(
        f"Profil Perangkat: Layar 14.0\" 16:10 ({config.screen_width_cm:.1f}x"
        f"{config.screen_height_cm:.1f} cm) | Margin Luar: {config.edge_margin_cm:.1f} cm | "
        f"Akhir Touchpad: {config.touchpad_end_depth_cm:.1f} cm"
    )
    print(
        "Kontrol: [SPASI / C] Kalibrasi Lensa Kamera (Anti-Spoof 2s) | "
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

                cal_step = advance_lens_calibration(state, raw_faces, now_mono, config)
                if cal_step is not None:
                    step_result = cal_step
                    _play_calibration_cue(cal_step.calibration_event)
                else:
                    instant_code, instant_label = evaluate_instant_indication(
                        mean_luminance=frame_pcd.mean_luminance,
                        global_edge_density=frame_pcd.global_edge_density,
                        faces=faces,
                        config=config,
                        calib_yaw_offset_deg=state.calib_yaw_offset_deg,
                        calib_pitch_offset_deg=state.calib_pitch_offset_deg,
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
                    config=config,
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
                        state.start_lens_calibration(
                            now_mono, config.calibration_duration_sec
                        )
                        _play_calibration_cue("STARTED")
                        print(
                            "[KALIBRASI DIMULAI] Tatap lurus Lensa Fisik Webcam di bezel atas "
                            "selama 2.0 detik (Anti-Spoof Aktif)."
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
