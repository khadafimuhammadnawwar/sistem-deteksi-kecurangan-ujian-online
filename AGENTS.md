# WORKSPACE RULES & AGENT DIRECTIVES — SISTEM INDIKASI KECURANGAN UJIAN ONLINE (PCD)

## 1. KONTEKS PROYEK & TUJUAN AKADEMIS
- **Nama Proyek:** Sistem Deteksi Indikasi Kecurangan pada Ujian Online Berbasis Citra Digital (Mode 1: CBT Paperless / Ujian Teori).
- **Mata Kuliah:** Praktikum Pengolahan Citra Digital (Semester 5 - D4 Teknik Informatika PNL).
- **Prinsip Utama:** Sistem ini adalah *Decision Support System (Non-Auto Banning)*. Sistem mendeteksi indikasi perilaku mencurigakan berdasarkan analisis temporal (durasi), mengurangi *Trust Score*, menampilkan status integritas (`AMAN`, `WASPADA`, `TINJAUAN PENGAWAS`), dan menyimpan bukti *screenshot* beserta *log* CSV untuk ditinjau pengawas.

## 2. ATURAN TEKNIS PYTHON (AKTIVASI STANDAR PYTHON 3.12+)
- **Standar Gaya Kode:** Wajib patuhi standar PEP 8 (`snake_case` untuk fungsi/variabel, `PascalCase` untuk Class/Enum, `UPPER_SNAKE_CASE` untuk konstanta).
- **Type Hinting Wajib:** Gunakan Python Type Hints (`typing`, `numpy.typing.NDArray[np.uint8]`) pada seluruh parameter fungsi dan nilai kembalian (`return value`).
- **Manajemen Konteks & Timer:** Wajib gunakan *context manager* (`with open(...)`) saat menulis baris baru ke file `log_indikasi_ujian.csv` dan `time.monotonic()` untuk kalkulasi durasi. Dilarang menggunakan `pandas` di dalam *real-time video loop* untuk mencegah *frame drop*.
- **Data Structures & Anti-Magic String:** Gunakan `dataclass` untuk konfigurasi/state dan `StrEnum` untuk kode indikasi (`IND-01` s/d `IND-05`, `NORMAL`).
- **MediaPipe Modern API:** Wajib gunakan `mediapipe.tasks.python.vision.FaceLandmarker` (`RunningMode.VIDEO`, 478 titik landmark kepala & iris). Dilarang menggunakan modul usang `mp.solutions` yang sudah dihapus pada MediaPipe 1.x / Python 3.14.
- **Virtual Environment:** Gunakan lingkungan terisolasi `venv/` (tanpa titik). Dilarang menginstal paket di luar `requirements.txt` tanpa izin eksplisit pengguna.

## 3. KEWAJIBAN PIPELINE PENGOLAHAN CITRA DIGITAL (PCD)
AI Agent **DILARANG** menghapus atau melewati tahapan matematis pengolahan citra berikut karena merupakan syarat penilaian akademis mata kuliah PCD:
1. **Global Luminance & Canny Occlusion Analysis:** Konversi frame BGR ke Grayscale ($Y = 0.299R + 0.587G + 0.114B$), hitung rata-rata intensitas piksel ($\mu_Y$), serta hitung *Global Canny Edge Density* untuk mendeteksi `IND-05` (Kamera Ditutup Gelap $\mu_Y < 35.0$ atau Oklusi Benda Terang Polos *Edge* $< 1.0\%$ tanpa wajah).
2. **Pre-processing CLAHE pada Ruang Warna CIE-LAB:** Konversi BGR ke CIE-LAB, terapkan *CLAHE* (`clipLimit=2.5, tileGridSize=(8,8)`) pada kanal *Lightness* ($L$), lalu kembalikan ke BGR/RGB **SEBELUM** frame diproses oleh MediaPipe `FaceLandmarker`.
3. **ROI PCD Panel Processing (Grid 2x2):** Potong *Region of Interest* (ROI) wajah utama ($110 \times 110$ piksel), lalu terapkan dan visualisasikan secara berurutan:
   - Grayscale + CLAHE
   - Filtering konvolusi *Gaussian Blur* ($5 \times 5$)
   - Segmentasi Biner *Otsu Thresholding*
   - Deteksi Tepi *Canny Edge* ($50, 130$) beserta perhitungan persentase *Edge Density*.

## 4. HIERARKI DOKUMEN & STRUKTUR DIREKTORI (PRAGMATIC SRP, KISS & DRY)
Setiap dokumen memiliki satu peran tegas tanpa duplikasi tabel ambang batas:
1. `AGENTS.md` — Aturan eksekusi AI Agent & kontrak akademis PCD (dimuat otomatis oleh Antigravity IDE).
2. `PRD.md` — Sumber kebenaran tunggal untuk spesifikasi fitur, rumus matematis, dan angka ambang batas (`DetectorConfig`).
3. `ARCHITECTURE.md` — Aliran data antar-modul, *invariants* sistem, *state machine* kalibrasi/temporal, & kesiapan ekstensi Mode 2.
4. `README.md` — Panduan penggunaan manusia & matriks pengujian laporan praktikum (`UJI-01` s/d `UJI-09`).

```text
├── .agent/skills/3d-gaze-proctoring/SKILL.md # Skill 3D Gaze Ray-Casting & Anti-Spoof
├── AGENTS.md                  # Aturan spesifik workspace untuk Antigravity AI Agent
├── PRD.md                     # Spesifikasi produk, rumus PCD + 3D Ray-Casting, & matriks aturan
├── ARCHITECTURE.md            # Desain berlapis, invariants Anti-Spoof, state machine, & SRP
├── README.md                  # Cara menjalankan & protokol pengujian laporan praktikum
├── requirements.txt           # Dependensi Python terkunci
├── .gitignore                 # Abaikan venv, cache, model biner, dan output runtime
├── main.py                    # Entry point utama akuisisi webcam & koordinasi loop
├── src/
│   ├── __init__.py
│   ├── config.py              # Konstanta geometri fisik laptop (cm), StrEnum, & State Anti-Spoof
│   ├── pcd_pipeline.py        # Fungsi murni Pengolahan Citra Digital (LAB-CLAHE, Gaussian, Otsu, Canny)
│   ├── detector.py            # FaceLandmarker 3D Matrix, L-Shaped Ray-Casting, & aturan temporal
│   ├── logger.py              # Pencatatan bukti screenshot (.jpg) dan append log (.csv)
│   └── ui_dashboard.py        # Rendering canvas OpenCV (920x480), HUD 3D (cm), & Grid 2x2 PCD
└── tests/
    └── test_detector_logic.py # Unit testing otomatis PCD, 3D Ray-Casting, Anti-Spoof, & logger
```

## 5. PERINTAH VERIFIKASI WAJIB SEBELUM COMMIT
```powershell
.\venv\Scripts\python.exe -m compileall main.py src tests
.\venv\Scripts\python.exe -m pytest tests/test_detector_logic.py -v
```
Jika mengubah konstanta di `src/config.py`, perbarui `PRD.md` pada commit yang sama agar dokumentasi tetap sinkron.
