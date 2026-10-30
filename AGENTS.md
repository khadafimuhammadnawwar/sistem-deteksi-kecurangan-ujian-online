# WORKSPACE RULES & AGENT DIRECTIVES — SISTEM INDIKASI KECURANGAN UJIAN ONLINE (PCD)

## 1. KONTEKS PROYEK & TUJUAN AKADEMIS
- **Nama Proyek:** Sistem Deteksi Indikasi Kecurangan pada Ujian Online Berbasis Citra Digital (Mode 1: CBT Paperless).
- **Mata Kuliah:** Pengolahan Citra Digital (Semester 5 - D4 Teknik Informatika PNL).
- **Prinsip Utama:** Sistem ini adalah *Decision Support System (Non-Auto Banning)*. Sistem mendeteksi indikasi perilaku mencurigakan berdasarkan analisis temporal (durasi), mengurangi *Trust Score*, dan menyimpan bukti *screenshot* beserta *log* CSV untuk ditinjau pengawas.

## 2. ATURAN TEKNIS PYTHON (AKTIVASI TEMPLATE GEMINI.MD BAGIAN 8A)
- **Standar Gaya Kode:** Wajib patuhi standar PEP 8 (`snake_case` untuk fungsi/variabel, `PascalCase` untuk Class/Enum, `UPPER_SNAKE_CASE` untuk konstanta).
- **Type Hinting Wajib:** Gunakan Python Type Hints (`typing`, `numpy.typing`) pada seluruh parameter fungsi dan nilai kembalian (`return value`).
- **Manajemen Konteks:** Wajib gunakan *context manager* (`with open(...)`) saat menulis baris baru ke file `log_indikasi_ujian.csv`. Dilarang menggunakan `pandas` di dalam *real-time video loop* untuk mencegah *frame drop*.
- **Data Structures & Anti-Magic String:** Gunakan `dataclass` untuk konfigurasi/state dan `Enum` untuk kode indikasi (`IND-01` s/d `IND-05`, `NORMAL`).
- **Virtual Environment:** Gunakan lingkungan terisolasi (`venv`). Dilarang menginstal paket di luar `requirements.txt` tanpa izin eksplisit pengguna.

## 3. KEWAJIBAN PIPELINE PENGOLAHAN CITRA DIGITAL (PCD)
AI Agent **DILARANG** menghapus atau melewati tahapan matematis pengolahan citra berikut karena merupakan syarat penilaian akademis mata kuliah PCD:
1. **Global Luminance Analysis:** Konversi frame BGR ke Grayscale ($Y = 0.299R + 0.587G + 0.114B$) dan hitung rata-rata intensitas piksel ($\mu_Y$) untuk mendeteksi `IND-05` (Kamera Ditutup / Gelap jika $\mu_Y < 35.0$).
2. **Pre-processing CLAHE pada Ruang Warna CIE-LAB:** Konversi BGR ke CIE-LAB, terapkan *CLAHE* (`clipLimit=2.5, tileGridSize=(8,8)`) pada kanal *Lightness* ($L$), lalu kembalikan ke BGR/RGB **SEBELUM** frame diproses oleh MediaPipe Face Mesh.
3. **ROI PCD Panel Processing:** Potong *Region of Interest* (ROI) wajah utama ($110 \times 110$ piksel), lalu terapkan secara berurutan:
   - Grayscale + CLAHE
   - Filtering konvolusi *Gaussian Blur* ($5 \times 5$)
   - Segmentasi Biner *Otsu Thresholding*
   - Deteksi Tepi *Canny Edge* ($50, 130$) beserta perhitungan persentase *Edge Density*.

## 4. ARSITEKTUR DIREKTORI & MODULARITAS (PRAGMATIC SRP & KISS)
Pertahankan struktur direktori yang bersih dan modular tetapi tidak *over-engineered*:
```text
├── AGENTS.md                  # Aturan spesifik workspace untuk AI Agent
├── PRD.md                     # Spesifikasi produk, rumus PCD, dan matriks aturan
├── requirements.txt           # Dependensi Python terkunci
├── .gitignore                 # Abaikan venv, cache, dan output runtime
├── main.py                    # Entry point utama akuisisi webcam & koordinasi loop
├── src/
│   ├── __init__.py
│   ├── config.py              # Konstanta ambang batas, Enum Indikasi, dan Dataclass State
│   ├── pcd_pipeline.py        # Fungsi murni Pengolahan Citra Digital (LAB-CLAHE, Otsu, Canny)
│   ├── detector.py            # Wrapper MediaPipe Face Mesh, kalkulasi rasio Yaw/Pitch & aturan temporal
│   ├── logger.py              # Pencatatan bukti screenshot (.jpg) dan append log (.csv) dengan context manager
│   └── ui_dashboard.py        # Rendering canvas OpenCV (920x480), banner status, bounding box, dan panel PCD
└── tests/
    └── test_detector_logic.py # Unit testing untuk logika geometri Yaw/Pitch, luminansi, dan timer