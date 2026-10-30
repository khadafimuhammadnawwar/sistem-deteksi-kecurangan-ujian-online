# Sistem Deteksi Indikasi Kecurangan pada Ujian Online Berbasis Citra Digital (Mode 1: CBT Paperless / Teori)

Proyek Akhir **Praktikum Pengolahan Citra Digital (Semester 5 — D4 Teknik Informatika PNL)**.  
Sistem ini merupakan *Computer-Assisted Proctoring (Decision Support System / Non-Auto Banning)* untuk ujian teori berbasis komputer tanpa kertas buram. Sistem menggabungkan pipeline matematis **Pengolahan Citra Digital (PCD)** klasik dengan ekstraksi **478 titik landmark wajah & pupil iris** secara *real-time*.

---

## Fitur Utama (Mode 1: Ujian Teori)
1. **Pipeline Pengolahan Citra Digital (PCD) Aktif:**
   - Analisis Luminansi Global ($Y = 0.299R + 0.587G + 0.114B$) untuk deteksi ruangan gelap/kamera ditutup benda gelap ($\mu_Y < 35.0$).
   - *Global Canny Edge Density* ($< 1.0\%$) untuk mendeteksi kamera ditutup benda polos terang (kertas putih/tisu/lakban).
   - Peningkatan kontras adaptif **CIE-LAB CLAHE** (`clipLimit=2.5, tileGridSize=(8,8)`) pada kanal *Lightness* ($L$) sebelum ekstraksi wajah.
   - Visualisasi *real-time* **Grid 2x2 ROI Wajah ($110\times 110$):** `1.Gray+CLAHE`, `2.Gauss 5x5`, `3.Otsu Biner`, dan `4.Canny Edge`.
2. **Deteksi Orientasi Kepala & Lirikan Bola Mata (*Iris Gaze Tracking*):**
   - Mendeteksi menoleh kiri/kanan (`IND-01`), menunduk/mendongak (`IND-02`), lirikan bola mata kiri/kanan (`IND-01: MELIRIK MATA`), serta lirikan bola mata atas/bawah (`IND-02: MELIRIK MATA`).
   - Dilengkapi **EMA Signal Smoothing ($\alpha = 0.35$)** peredam getaran koordinat dan **EAR Blink Guard ($\text{EAR} \ge 0.15$)** agar kedipan mata normal tidak dianggap kecurangan.
3. **Multi-Face Area Sorting & Calibration Gate:**
   - Wajah peserta utama selalu diurutkan dari luas *bounding box* terbesar (`Wajah #1`).
   - Pemotongan *Trust Score* dikunci sampai peserta menekan tombol kalibrasi **`[C]`** di awal sesi.
4. **Audit Bukti Otomatis (`Cooldown 5.0s`):**
   - Menyimpan tangkapan layar ber-anotasi ke `bukti_indikasi/*.jpg` dan mencatat riwayat ke `log_indikasi_ujian.csv`.

---

## Struktur Direktori
```text
├── AGENTS.md                  # Direktif arsitektur & kontrak aturan proyek
├── PRD.md                     # Spesifikasi produk, rumus matematis PCD, & matriks aturan
├── README.md                  # Panduan penggunaan & protokol pengujian akurasi
├── requirements.txt           # Daftar dependensi Python
├── main.py                    # Entry point akuisisi webcam & loop pemantauan
├── src/
│   ├── config.py              # Enum kode indikasi, konstanta, & state EMA/kalibrasi
│   ├── pcd_pipeline.py        # Fungsi murni LAB-CLAHE, Gaussian 5x5, Otsu, & Canny
│   ├── detector.py            # FaceLandmarker (478 titik), geometri kepala/iris, & timer
│   ├── logger.py              # Pencatatan bukti .jpg & .csv dengan context manager
│   └── ui_dashboard.py        # Rendering dashboard 920x480 & Grid 2x2 PCD
└── tests/
    └── test_detector_logic.py # Unit testing otomatis berbasis pytest
```

---

## Cara Menjalankan Aplikasi

1. **Aktifkan Virtual Environment (PowerShell):**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   *(Atau jalankan langsung menggunakan interpreter `.\venv\Scripts\python.exe`).*

2. **Jalankan Sistem Pemantauan:**
   ```powershell
   .\venv\Scripts\python.exe main.py
   ```
   *(Pada eksekusi pertama, model resmi `models/face_landmarker.task` diunduh otomatis).*

3. **Kontrol Tombol saat Aplikasi Berjalan:**
   - **`C`** : Kalibrasi titik tengah kepala & pandangan mata (pastikan tepat 1 wajah fokus ke tengah layar).
   - **`R`** : Reset *Trust Score* kembali ke `100%` dan nolkan penghitung indikasi.
   - **`Q`** : Keluar dan tutup kamera dengan aman.

4. **Menjalankan Unit Test:**
   ```powershell
   .\venv\Scripts\python.exe -m pytest tests/test_detector_logic.py -v
   ```

---

## Protokol Pengujian Mandiri & Tabel Evaluasi Laporan PCD

Gunakan matriks skenario berikut saat mengambil data Bab Hasil & Pembahasan laporan praktikum (ulangi tiap skenario minimal 5–10 kali pada pencahayaan terang dan redup):

| Kode Uji | Skenario Perilaku | Durasi Tahan | Kode Harapan | Hasil Aktual (TP / FP / FN) |
| :---: | :--- | :---: | :---: | :--- |
| **UJI-01** | Fokus menghadap layar & berkedip wajar | $10.0\text{ s}$ | `NORMAL` | ... |
| **UJI-02** | Menolehkan kepala ke kiri / kanan | $\ge 2.5\text{ s}$ | `IND-01` (Menoleh) | ... |
| **UJI-03** | Kepala diam lurus, bola mata melirik tajam ke kiri / kanan | $\ge 2.0\text{ s}$ | `IND-01` (Melirik Mata) | ... |
| **UJI-04** | Menundukkan kepala ke pangkuan / mendongak ke atas | $\ge 3.0\text{ s}$ | `IND-02` (Menunduk/Mendongak) | ... |
| **UJI-05** | Kepala diam lurus, bola mata melirik tajam ke atas / bawah ekstrem | $\ge 2.0\text{ s}$ | `IND-02` (Melirik Mata) | ... |
| **UJI-06** | Orang kedua masuk ke tangkapan kamera | $\ge 1.0\text{ s}$ | `IND-03` (>1 Orang) | ... |
| **UJI-07** | Peserta meninggalkan kursi (latar ruangan terlihat) | $\ge 3.0\text{ s}$ | `IND-04` (Wajah Hilang) | ... |
| **UJI-08** | Lensa kamera ditutup telapak tangan gelap | $\ge 2.0\text{ s}$ | `IND-05` (Gelap $\mu_Y < 35$) | ... |
| **UJI-09** | Lensa kamera ditutup kertas HVS putih / tisu terang | $\ge 2.0\text{ s}$ | `IND-05` (Oklusi Canny $< 1\%$) | ... |
