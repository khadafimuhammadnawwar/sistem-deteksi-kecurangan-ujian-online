# ARCHITECTURE ESSENTIALS (FAST CONTEXT & INVARIANTS)

Dokumen ringkas berdensitas tinggi ini memuat **aturan mati (*invariants*)** yang wajib dipatuhi manusia maupun AI Agent sebelum memodifikasi kode apa pun di repositori ini.

## 1. Non-Negotiable Invariants
1. **Runtime & API:** Lingkungan target menggunakan **Python 3.12+ (aktif: 3.14)** dan **MediaPipe Tasks API (`mediapipe.tasks.python.vision.FaceLandmarker`, `RunningMode.VIDEO`)**. **DILARANG** menggunakan `mp.solutions.face_mesh` karena sudah dihapus dari MediaPipe modern.
2. **Filosofi Keputusan:** Sistem adalah **Decision Support System (*Non-Auto Banning*)**. Sistem hanya mengurangi `Trust Score` (`-5%` per penalti, minimum `0%`), mengubah status (`AMAN` / `WASPADA` / `TINJAUAN PENGAWAS`), dan mencatat bukti `.jpg` + `.csv`. Dilarang menutup paksa sesi ujian peserta.
3. **Kewajiban Akademis PCD:** Tiga blok pengolahan citra berikut **wajib selalu dieksekusi aktif** di setiap frame:
   - Grayscale Luminance ($\mu_Y < 35.0$) + *Global Canny Edge Density* ($< 1.0\%$ tanpa wajah) untuk `IND-05`.
   - Konversi BGR $\rightarrow$ **CIE-LAB** + **CLAHE (`clipLimit=2.5, tileGridSize=(8,8)`)** pada kanal $L$ sebelum masuk ke `FaceLandmarker`.
   - Ekstraksi 4 panel ROI wajah utama (`110x110`): `Gray+CLAHE` $\rightarrow$ `Gaussian 5x5` $\rightarrow$ `Otsu Binary` & `Canny Edge (50, 130)`.
4. **Performa Real-Time:** Dilarang memanggil `pandas`, melakukan alokasi objek `CLAHE` berulang di dalam *loop*, atau mengeksekusi jaringan di dalam siklus `while cap.isOpened():`.

## 2. Tabel Konstanta & Ambang Batas Kunci (Mode 1: Teori)

| Parameter | Nilai | Tujuan Teknis |
| :--- | :---: | :--- |
| `calibration_countdown_sec` | `2.0 s` | Jeda hitung mundur usai tekan `SPASI`/`C` agar mata tidak terekam saat melihat keyboard. |
| `calibration_sample_window_sec` | `0.8 s` | Durasi pengambilan rata-rata sampel akhir kalibrasi saat menatap titik tengah `(320, 240)`. |
| `ema_alpha` | `0.35` | Bobot penghalus sinyal `Yaw`, `Pitch`, `GazeX`, `GazeY`. |
| `tol_yaw` / `thresh_yaw_sec` | `0.15` / `2.5 s` | Toleransi & durasi menoleh kepala kiri/kanan (`IND-01`). |
| `tol_pitch` / `thresh_pitch_sec` | `0.10` / `3.0 s` | Toleransi & durasi kepala menunduk/mendongak (`IND-02`). |
| `tol_gaze_x` / `thresh_gaze_sec` | `0.135` / `2.5 s` | Zona aman baca layar kiri–kanan sebelum dianggap melirik keluar layar (`IND-01`). |
| `tol_gaze_up` & `tol_gaze_down` | `0.13` / `2.5 s` | Zona aman vertikal mata dengan kompensasi VOR `0.80 * ΔPitch` (`IND-02`). |
| `min_ear_for_gaze` | `0.15` | Batas *Eye Aspect Ratio* untuk mengabaikan pergeseran iris saat berkedip. |
| `grace_period_sec` / `log_cooldown_sec` | `0.3 s` / `5.0 s` | Anti-flicker reset timer & jeda minimal penyimpanan screenshot bukti. |

## 3. Hierarki Prioritas Evaluasi Frame
`IND-05 (Gelap / Oklusi Polos)` $\rightarrow$ `IND-03 (>1 Wajah)` $\rightarrow$ `IND-04 (0 Wajah)` $\rightarrow$ `IND-01 (Kepala Kiri/Kanan)` $\rightarrow$ `IND-02 (Kepala Bawah/Atas)` $\rightarrow$ `IND-01 (Mata Kiri/Kanan)` $\rightarrow$ `IND-02 (Mata Atas/Bawah VOR)` $\rightarrow$ `NORMAL`.
