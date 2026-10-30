# Sistem Deteksi Indikasi Kecurangan pada Ujian Online Berbasis Citra Digital (Mode 1: CBT Paperless / Teori)

Proyek Akhir **Praktikum Pengolahan Citra Digital (Semester 5 — D4 Teknik Informatika PNL)**.  
Sistem ini merupakan *Computer-Assisted Proctoring (Decision Support System / Non-Auto Banning)* untuk ujian teori berbasis komputer tanpa kertas buram. Sistem menggabungkan pipeline matematis **Pengolahan Citra Digital (PCD)** klasik dengan **L-Shaped 3D Gaze Ray-Casting (`cm`)** dan **5-Layer Anti-Spoof Lens Calibration**.

---

## Fitur Utama (Mode 1: Ujian Teori)
1. **Pipeline Pengolahan Citra Digital (PCD) Aktif:**
   - Analisis Luminansi Global ($Y = 0.299R + 0.587G + 0.114B$) untuk deteksi ruangan gelap/oklusi gelap ($\mu_Y < 35.0$).
   - *Global Canny Edge Density* ($< 1.0\%$) untuk mendeteksi kamera ditutup benda polos terang (kertas putih/tisu/lakban).
   - Peningkatan kontras adaptif **CIE-LAB CLAHE** (`clipLimit=2.5, tileGridSize=(8,8)`) pada kanal *Lightness* ($L$) sebelum ekstraksi wajah.
   - Visualisasi *real-time* **Grid 2x2 ROI Wajah ($110\times 110$):** `1.Gray+CLAHE`, `2.Gauss 5x5`, `3.Otsu Biner`, dan `4.Canny Edge`.
2. **Hardware-Locked 3D Gaze Ray-Casting (`cm`) & Anti-Spoof:**
   - Geometri fisik perangkat dikunci di sisi sistem (**Lenovo IdeaPad Slim 5 14IMH9**: layar `14.0"` `16:10` $30.16 \times 18.85\text{ cm}$ + dek keyboard/touchpad $21.0\text{ cm}$), **mustahil diperlebar/diakali peserta**.
   - Memicu kuning seketika saat pandangan keluar area layar laptop di sisi kiri, kanan, atau atas layar (`IND-01` / `IND-02`), maupun tepat mencapai ujung akhir bawah touchpad **`>= 21.0 cm`** (`IND-02`).
   - Dilengkapi **EMA Smoothing ($\alpha = 0.42$)**, kompensasi **VOR (Vestibulo-Ocular Reflex)**, serta **EAR Blink Guard ($\text{EAR} \ge 0.12$)**.
3. **5-Layer Anti-Spoof Lens Calibration (`2.0 Detik`):**
   - Peserta cukup menatap lurus **Lensa Fisik Webcam** di bezel atas selama 2 detik. Jika peserta mencoba melirik ke samping/atas/bawah atau menoleh saat kalibrasi, sistem **menolak kalibrasi secara otomatis**.
4. **Audit Bukti Otomatis (`Cooldown 5.0s`):**
   - Menyimpan tangkapan layar ber-anotasi ke `bukti_indikasi/*.jpg` dan mencatat riwayat ke `log_indikasi_ujian.csv`.

---

## Cara Menjalankan Aplikasi

1. **Jalankan Sistem Pemantauan (PowerShell):**
   ```powershell
   .\venv\Scripts\python.exe main.py
   ```
   *(Atau aktifkan environment terlebih dahulu dengan `.\venv\Scripts\Activate.ps1`).*

2. **Kontrol Tombol & Prosedur Kalibrasi Anti-Spoof:**
   - **`SPASI` atau `C`** : Mulai **Kalibrasi Lensa Fisik Webcam (2.0 detik)**. Tatap tepat ke lubang lensa kamera di bezel atas layar sampai terdengar *double-beep* sukses (jika coba melirik jauh, terdengar *low-beep* dan kalibrasi ditolak).
   - **`R`** : Reset *Trust Score* kembali ke `100%` dan nolkan penghitung indikasi.
   - **`Q`** : Keluar dan tutup kamera dengan aman.
   - *Catatan HUD Bawah:* Koordinat titik pandang ditampilkan langsung dalam sentimeter (`X:+0.0cm | Y:+9.5cm | Z:43cm` saat di layar, atau `Touchpad:14.2/21cm` saat melihat area keyboard/touchpad).

3. **Menjalankan Verifikasi & Unit Test:**
   ```powershell
   .\venv\Scripts\python.exe -m compileall main.py src tests
   .\venv\Scripts\python.exe -m pytest tests/test_detector_logic.py -v
   ```

---

## Protokol Pengujian Mandiri & Tabel Evaluasi Laporan PCD

| Kode Uji | Skenario Perilaku | Durasi Tahan | Kode Harapan | Hasil Aktual (TP / FP / FN) |
| :---: | :--- | :---: | :---: | :--- |
| **UJI-01** | Fokus menatap seluruh sudut layar, keyboard, & berkedip wajar | $10.0\text{ s}$ | `NORMAL` | ... |
| **UJI-02** | Menolehkan kepala keluar dari batas kiri / kanan layar ($|X| > 22.1\text{ cm}$) | $\ge 2.5\text{ s}$ | `IND-01` (Menoleh) | ... |
| **UJI-03** | Kepala diam, bola mata melirik di luar kiri / kanan layar ($|X| > 22.1\text{ cm}$) | $\ge 2.5\text{ s}$ | `IND-01` (Melirik Mata) | ... |
| **UJI-04** | Menunduk melewati ujung akhir touchpad ($\ge 21\text{ cm}$) / mendongak di atas layar ($Y < -7.0\text{ cm}$) | $\ge 3.0\text{ s}$ | `IND-02` (Menunduk/Mendongak) | ... |
| **UJI-05** | Kepala diam, bola mata melirik di atas layar ($Y < -7.0\text{ cm}$) / lewat ujung akhir touchpad | $\ge 2.5\text{ s}$ | `IND-02` (Melirik Mata) | ... |
| **UJI-06** | Orang kedua masuk ke tangkapan kamera | $\ge 1.0\text{ s}$ | `IND-03` (>1 Orang) | ... |
| **UJI-07** | Peserta meninggalkan kursi (latar ruangan terlihat) | $\ge 3.0\text{ s}$ | `IND-04` (Wajah Hilang) | ... |
| **UJI-08** | Lensa kamera ditutup telapak tangan gelap | $\ge 2.0\text{ s}$ | `IND-05` (Gelap $\mu_Y < 35$) | ... |
| **UJI-09** | Lensa kamera ditutup kertas HVS putih / tisu terang | $\ge 2.0\text{ s}$ | `IND-05` (Oklusi Canny $< 1\%$) | ... |
