# SYSTEM ARCHITECTURE & CORE INVARIANTS
# Sistem Deteksi Indikasi Kecurangan Ujian Online Berbasis Citra Digital (Mode 1: CBT Paperless)

Dokumen ini mendefinisikan batas modul, *invariants* teknis, aliran data, *state machine*, dan mekanisme kompensasi ergonomi sistem. Untuk daftar angka ambang batas dan persamaan lengkap, rujuk sumber kebenaran tunggal di [PRD.md](./PRD.md).

---

## 1. Core Architectural Invariants (Aturan Mati Sistem)
1. **Unidirectional Pure Pipeline:** Pemrosesan citra di `src/pcd_pipeline.py` dan kalkulasi geometri di `src/detector.py` dirancang sebagai fungsi bebas I/O (*pure/testable functions*) sehingga seluruh logika dapat diuji tanpa perangkat keras kamera di `tests/test_detector_logic.py`.
2. **Monotonic Video Timestamps & Clock:** Seluruh perhitungan durasi wajib menggunakan `time.monotonic()` (kebal sinkronisasi jam OS), dan `FaceMeshDetector.detect_faces` menjamin `timestamp_ms` selalu naik ketat (`strictly increasing`) sesuai syarat graf C++ MediaPipe `RunningMode.VIDEO`.
3. **Zero Frame-Drop Loop:** Objek berat (`CLAHE`, `FaceLandmarker`, `EvidenceLogger`) hanya diinisialisasi satu kali di luar `while cap.isOpened():`. Penulisan log CSV menggunakan modul standar `csv` dengan `with open(...)`.
4. **Urutan Prioritas Deterministik:** Evaluasi per-frame selalu memproses anomali lingkungan/kamera terlebih dahulu sebelum geometri wajah:
   `IND-05 (Gelap/Oklusi)` $\rightarrow$ `IND-03 (>1 Wajah)` $\rightarrow$ `IND-04 (0 Wajah)` $\rightarrow$ `IND-01 (Kepala Horizontal)` $\rightarrow$ `IND-02 (Kepala Vertikal)` $\rightarrow$ `IND-01 (Mata Horizontal)` $\rightarrow$ `IND-02 (Mata Vertikal VOR)` $\rightarrow$ `NORMAL`.

---

## 2. Aliran Data Berlapis & Kontrak Modul (SRP)

```mermaid
flowchart LR
    subgraph IO_IN["1. Acquisition Layer"]
        CAM["Webcam 640x480 (main.py)"]
    end
    subgraph PCD["2. Pure PCD Layer (src/pcd_pipeline.py)"]
        LUM["Grayscale Luminance & Global Canny"]
        LAB["CIE-LAB CLAHE (L-Channel)"]
        ROI["ROI 110x110: CLAHE -> Gauss 5x5 -> Otsu & Canny"]
    end
    subgraph CV["3. Geometry & Temporal Layer (src/detector.py)"]
        MP["MediaPipe Tasks FaceLandmarker (478 pts)"]
        EMA["Area Sort + EMA Smoothing + VOR Iris Gaze"]
        FSM["2.0s Countdown Calibration & Temporal Engine"]
    end
    subgraph OUT["4. Presentation & Audit Layer"]
        UI["OpenCV Dashboard 920x480 (src/ui_dashboard.py)"]
        LOG["CSV & Annotated JPG Logger (src/logger.py)"]
    end

    CAM --> LUM & LAB
    LAB --> MP --> EMA --> FSM
    LUM --> FSM
    EMA --> ROI --> UI
    FSM --> UI & LOG
```

| Modul | Tanggung Jawab Tunggal | Ketergantungan Internal |
| :--- | :--- | :--- |
| `src/config.py` | Definisi `StrEnum`, parameter `DetectorConfig`, dan state sesi `MonitoringState`. | Tidak ada (*Leaf Module*) |
| `src/pcd_pipeline.py` | Transformasi matriks citra (Luminansi, LAB-CLAHE, Gaussian, Otsu, Canny). | `src/config.py` |
| `src/detector.py` | Ekstraksi 478 landmark, sorting luas wajah, kompensasi VOR iris, kalibrasi, & timer. | `src/config.py` |
| `src/logger.py` | Persistensi screenshot bukti ber-anotasi (`.jpg`) dan baris CSV (`.csv`). | `src/config.py` |
| `src/ui_dashboard.py` | Komposisi kanvas piksel `920x480`, target bidik kalibrasi, & Grid 2x2 PCD. | `config`, `detector`, `pcd_pipeline` |
| `main.py` | Koordinator siklus kamera, input keyboard (`SPASI`/`C`, `R`, `Q`), & *cleanup*. | Seluruh modul `src/` |

---

## 3. Desain Ergonomi Layar & Mesin Keadaan (*State Machine*)

1. **Pencegahan *Keyboard Gaze Trap* (Countdown Calibration):**
   Saat peserta menekan tombol kalibrasi di papan ketik, mata cenderung melirik ke bawah sesaat. Sistem menerapkan hitung mundur `2.0 detik` dengan target visual di tengah layar `(320, 240)` dan hanya merata-ratakan sampel pada `0.8 detik` terakhir saat fokus mata sudah stabil di tengah layar.
2. **Kompensasi *Vestibulo-Ocular Reflex* (VOR) & *Blink Guard*:**
   Ketika dagu peserta turun sedikit namun mata tetap membaca layar, bola mata secara alami berputar ke atas di dalam rongga mata. Sistem mengompensasi `Gaze_y` dengan perubahan `Pitch` kepala serta mengabaikan frame kedipan mata (`EAR < 0.15`) agar membaca seluruh area layar laptop tetap berada di zona **Hijau (`AMAN`)**.

```mermaid
stateDiagram-v2
    [*] --> UNCALIBRATED
    UNCALIBRATED --> CALIBRATING: Tekan SPASI/C (1 Wajah)
    CALIBRATING --> UNCALIBRATED: Wajah != 1 (Batal)
    CALIBRATING --> SAFE: Selesai 2.0s (Simpan Rata-rata 0.8s Akhir)
    SAFE --> CALIBRATING: Tekan SPASI/C Ulang
    SAFE --> WARNING: Indikasi Terdeteksi (Durasi < Ambang)
    WARNING --> SAFE: Normal >= 0.3s (Grace Period)
    WARNING --> ALERT: Durasi >= Ambang Indikasi
    ALERT --> SAFE: Normal >= 0.3s
    ALERT --> ALERT: Cooldown >= 5.0s -> Potong Skor -5% & Simpan Bukti
```

---

## 4. Kesiapan Ekstensi Masa Depan (*Mode 2: Ujian Coret-Coret / Kertas Buram*)
Sesuai *Open/Closed Principle (OCP)*, penambahan **Mode 2** di tahap berikutnya cukup dilakukan dengan menambahkan profil konfigurasi `ExamMode.SCRATCH_PAPER` pada `DetectorConfig` (menyesuaikan batas toleransi dan durasi menunduk saat menulis di meja) tanpa perlu merombak `src/pcd_pipeline.py`, `src/logger.py`, maupun arsitektur dasar `src/ui_dashboard.py`.
