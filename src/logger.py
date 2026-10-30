"""Pencatatan bukti screenshot (.jpg) dan append log (.csv) menggunakan context manager."""

import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import IndicationCode

UInt8Image = NDArray[np.uint8]

CSV_HEADERS: list[str] = [
    "Waktu",
    "Kode_Indikasi",
    "Jenis_Perilaku",
    "Durasi_Detik",
    "Mean_Luminance",
    "Trust_Score",
    "File_Bukti",
]


class EvidenceLogger:
    """Mengelola penyimpanan screenshot bukti ber-anotasi dan log kejadian CSV."""

    def __init__(self, evidence_dir: Path, log_file: Path) -> None:
        self._evidence_dir = evidence_dir
        self._log_file = log_file
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Memastikan direktori bukti dan header file CSV siap digunakan."""
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        if self._log_file.exists():
            return

        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_file, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CSV_HEADERS)

    def record_indication(
        self,
        annotated_frame: UInt8Image,
        code: IndicationCode,
        behavior_label: str,
        duration_sec: float,
        mean_luminance: float,
        trust_score: float,
        flag_number: int,
        timestamp: datetime | None = None,
    ) -> Path:
        """Menyimpan citra bukti (.jpg) dan menambahkan satu baris log ke CSV."""
        now = timestamp or datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        file_time = now.strftime("%Y%m%d_%H%M%S")
        evidence_path = self._evidence_dir / f"{code.value}_{file_time}_flag{flag_number}.jpg"

        if not cv2.imwrite(str(evidence_path), annotated_frame):
            raise OSError(f"Gagal menyimpan bukti citra ke '{evidence_path}'.")

        relative_str = evidence_path.as_posix()
        with open(self._log_file, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    timestamp_str,
                    code.value,
                    behavior_label,
                    round(duration_sec, 1),
                    round(mean_luminance, 1),
                    round(trust_score, 1),
                    relative_str,
                ]
            )

        print(
            f"[LOG TERSIMPAN] {timestamp_str} | {code.value} - {behavior_label} "
            f"({duration_sec:.1f}s) | Lum: {mean_luminance:.1f} | Skor: {trust_score:.0f}%"
        )
        return evidence_path

