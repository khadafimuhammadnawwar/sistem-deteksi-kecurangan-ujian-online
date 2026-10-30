# PRODUCT REQUIREMENTS DOCUMENT (PRD)
# Sistem Deteksi Indikasi Kecurangan pada Ujian Online Berbasis Citra Digital

## 1. Visi & Filosofi Sistem
Sistem ini adalah aplikasi *Computer-Assisted Proctoring* berbasis pengolahan citra digital dan *computer vision* menggunakan Python.

**Prinsip Kerja (Non-Auto Banning / Decision Support System):**
Sistem **TIDAK** mendiskualifikasi atau mengeluarkan peserta secara otomatis. Sistem berperan sebagai pendeteksi indikasi awal (*Suspicious Behavior Flagging*) dengan cara:
1. Memantau luminansi pencahayaan global, kepadatan tepi global (*Global Canny Edge Density*), koordinat pandang gabungan kepala & bola mata (*Unified Point-of-Regard* `POR_x` & `POR_y`), keberadaan wajah, dan jumlah orang dalam frame secara *real-time*.
2. Menerapkan **Kalibrasi 5 Titik Batas Fisik Laptop** (`Tengah Layar` $\rightarrow$ `Tepi Kiri` $\rightarrow$ `Tepi Kanan` $\rightarrow$ `Bezel Atas` $\rightarrow$ `Titik Akhir Touchpad`) sehingga seluruh permukaan layar & papan ketik aman dilihat, sedangkan pandangan keluar `~1 cm` di sisi kiri/kanan/atas layar atau mencapai titik akhir bawah touchpad langsung terdeteksi secara presisi dan simetris.
3. Mengurangi *Trust Score* (Skor Integritas), menampilkan eskalasi status integritas (`AMAN`, `WASPADA`, `TINJAUAN PENGAWAS`), mencatat log kejadian ke `log_indikasi_ujian.csv`, dan menyimpan bukti foto ber-anotasi ke direktori `bukti_indikasi/*.jpg`.

---

## 2. Ruang Lingkup Masalah (Mode 1: CBT Paperless / Ujian Teori)
1. **Model Ujian:** Dikhususkan untuk ujian teori berbasis komputer tanpa kertas buram, catatan fisik, maupun gawai/HP (area kerja sah mencakup layar laptop dan papan ketik hingga batas atas titik akhir touchpad).
2. **Perangkat:** 1 unit webcam bawaan laptop beresolusi 640x480 piksel menghadap wajah hingga batas bahu peserta.
3. **Kondisi Lingkungan:** Menggunakan algoritma CLAHE pada kanal luminansi CIE-LAB agar tetap adaptif terhadap pencahayaan redup atau tidak merata, dikombinasikan dengan deteksi tepi Canny untuk menangkap oklusi kamera baik oleh benda gelap maupun benda terang polos.

---

## 3. Pipeline Pengolahan Citra Digital (PCD)
Setiap frame video berukuran 640x480 piksel wajib diproses melalui 6 tahapan matriks citra secara berurutan:

1. **Akuisisi Citra & Pencerminan (Mirroring):**
   Frame BGR diambil dari webcam dan dibalik secara horizontal menggunakan transformasi matriks pencerminan (`cv2.flip(frame, 1)`).
2. **Konversi Ruang Warna, Luminansi Global & Deteksi Oklusi Tepi:**
   - Konversi frame BGR ke Grayscale 1-channel menggunakan persamaan intensitas luminansi:
     $$Y = 0.299R + 0.587G + 0.114B$$
   - Hitung nilai rata-rata piksel global ($\mu_Y$) untuk mendeteksi anomali kamera ditutup benda gelap atau ruangan gelap gulita ($\mu_Y < 35.0$).
   - Terapkan Gaussian Blur $5\times 5$ dan operator Canny (`threshold1=50`, `threshold2=130`) pada citra global untuk menghitung *Global Edge Density* guna mendeteksi kamera ditutup benda terang polos/blur ekstrem ($< 1.0\%$ saat wajah tidak terdeteksi).
3. **Peningkatan Kontras Adaptif (CIE-LAB CLAHE):**
   - Konversi frame BGR ke ruang warna CIE-LAB.
   - Terapkan CLAHE (*Contrast Limited Adaptive Histogram Equalization*) dengan `clipLimit=2.5` dan `tileGridSize=(8,8)` secara eksklusif pada kanal *Lightness* ($L$).
   - Konversi kembali ke format BGR/RGB sebelum masuk ke pipeline deteksi landmark wajah.
4. **Ekstraksi Fitur Geometri Kepala + Iris (*Unified Point-of-Regard*) & Penghalusan EMA:**
   Mengekstrak 478 titik landmark (`RunningMode.VIDEO`), mengurutkan wajah berdasarkan luas *bounding box* terbesar sebagai peserta utama (Wajah #1), menghaluskan sinyal dengan EMA ($\alpha = 0.45$), lalu memetakan koordinat pandang gabungan kepala dan mata:
   $$\text{POR}_{x} = \text{Gaze}_{x} + 0.55 \cdot (\text{Yaw} - \text{Yaw}_{\text{center}}), \quad \text{POR}_{y} = \text{Gaze}_{y} + 0.85 \cdot (\text{Pitch} - \text{Pitch}_{\text{center}})$$
   dengan proteksi kedipan alami ($\text{EAR} \ge 0.09$) agar kedipan mata tidak menimbulkan lonjakan koordinat iris.
5. **Filtering Spasial pada ROI Wajah:**
   Ambil *Region of Interest* (ROI) wajah utama, sesuaikan skala ke $110\times 110$ piksel, konversi ke Grayscale + CLAHE, lalu terapkan konvolusi **Gaussian Blur kernel $5\times 5$** ($\sigma = 0$) untuk meredam noise frekuensi tinggi.
6. **Segmentasi Biner & Ekstraksi Tepi pada ROI:**
   - **Binarisasi Otsu:** Segmentasi adaptif otomatis memisahkan fitur wajah dari latar belakang (`cv2.THRESH_BINARY + cv2.THRESH_OTSU`).
   - **Deteksi Tepi Canny:** Operator gradien intensitas tepi dengan ambang `threshold1=50` dan `threshold2=130`.
   - **Metrik Edge Density ROI:** Dihitung dari persentase jumlah piksel tepi non-nol terhadap total ukuran ROI:
     $$\text{Edge Density} = \frac{\sum \text{Edge Pixels}}{\text{Total Pixels}} \times 100\%$$

---

## 4. Matriks Aturan Indikasi Kecurangan (*Rule-Based Temporal Logic*)
Evaluasi indikasi diproses setelah **Kalibrasi 5 Titik Batas Fisik** selesai berdasarkan hierarki prioritas berikut:

| Prioritas | Kode | Jenis Perilaku | Ambang Pemicu (Batas Fisik `SafeZoneBounds`) | Ambang Durasi | Konsekuensi & Aksi |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | **IND-05** | Kamera Tertutup / Oklusi | $\mu_Y < 35.0$ **atau** (*Global Edge* $< 1.0\%$ & wajah $= 0$) | $\ge 2.0\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **2** | **IND-03** | Terdeteksi >1 Orang | Jumlah wajah $> 1$ | $\ge 1.0\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **3** | **IND-04** | Wajah Tidak Terdeteksi | Jumlah wajah $= 0$ (pencahayaan & tepi normal) | $\ge 3.0\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **4** | **IND-01** | Menoleh / Melirik Kiri-Kanan | $\text{POR}_{x} < x_{\text{kiri}} - 1\text{cm}$ atau $\text{POR}_{x} > x_{\text{kanan}} + 1\text{cm}$ | $\ge 2.5\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **5** | **IND-02** | Menunduk-Mendongak / Lewat Touchpad | $\text{POR}_{y} < y_{\text{atas}} - 1\text{cm}$ atau $\text{POR}_{y} \ge y_{\text{akhir\_touchpad}}$ | Kepala $\ge 3.0\text{ s}$ / Mata $\ge 2.5\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **6** | **NORMAL** | Fokus Area Layar & Keyboard | Tepat 1 wajah, pencahayaan normal, $\text{POR}_{x,y}$ di dalam zona aman laptop | *Real-time* | Banner Hijau, Reset Timer setelah *grace period* $0.3\text{ s}$ |

*Aturan Tambahan:*
- **Kalibrasi 5 Titik Fisik (`[SPASI]` / `[C]`):** Memandu peserta menatap 5 batas berurutan (`2.0 detik` per titik, rata-rata sampel `0.8 detik` terakhir disertai sinyal suara *beep* tiap transisi): **(1) Tengah Layar**, **(2) Tepi Kiri Layar**, **(3) Tepi Kanan Layar**, **(4) Bezel Atas/Webcam**, dan **(5) Titik Akhir Bawah Touchpad**.
- **Cooldown Penyimpanan:** Penyimpanan bukti screenshot dan penulisan log CSV dibatasi jeda minimal 5.0 detik (`LOG_COOLDOWN_SEC = 5.0`).
- **Skor & Status Integritas:** *Trust Score* berawal dari `100%` (minimum `0%`) dengan kategori eskalasi: `AMAN` (`80–100%`), `WASPADA` (`50–79%`), dan `TINJAUAN PENGAWAS` (`< 50%`).

---

## 5. Spesifikasi Antarmuka Visual (Dashboard 920x480 Piksel)
1. **Panel Kamera Utama (Kiri: 640x480 piksel):**
   - *Bounding Box* Wajah: Biru-Cyan `(255, 200, 0)` untuk Wajah #1 (peserta utama) dan Merah `(0, 0, 255)` untuk wajah tambahan.
   - Penanda Landmark & Panduan 5 Titik: Titik kuning pada hidung, 2 titik magenta pada pupil Wajah #1, serta panah pemandu arah kalibrasi (`1/5` s/d `5/5`).
   - *Status Banner* atas dinamis dan baris metrik bawah (`Yaw`, `Pitch`, `EyeX`, `EyeY`).
2. **Panel Monitoring PCD (Kanan: 280x480 piksel):**
   - Metrik `Trust Score`, `Status` integritas, `Total Indikasi`, `Jumlah Wajah`, `Luminansi (Y)`, serta `Edge ROI/Glob`.
   - **Grid 2x2 Visualisasi Tahapan PCD ($110\times 110$ piksel):** `1.Gray+CLAHE`, `2.Gauss 5x5`, `3.Otsu Biner`, dan `4.Canny Edge`.

---

## 6. Kontrol Pengguna & Skema Bukti Audit
- **Kontrol Keyboard:** `[SPASI]` atau `[C]` Mulai Kalibrasi 5 Titik Fisik, `[R]` Reset Skor & Timer, `[Q]` Keluar.
- **Skema `log_indikasi_ujian.csv`:** `Waktu,Kode_Indikasi,Jenis_Perilaku,Durasi_Detik,Mean_Luminance,Trust_Score,File_Bukti`
