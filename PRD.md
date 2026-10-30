# PRODUCT REQUIREMENTS DOCUMENT (PRD)
# Sistem Deteksi Indikasi Kecurangan pada Ujian Online Berbasis Citra Digital

## 1. Visi & Filosofi Sistem
Sistem ini adalah aplikasi *Computer-Assisted Proctoring* berbasis pengolahan citra digital dan *computer vision* menggunakan Python.

**Prinsip Kerja (Non-Auto Banning / Decision Support System):**
Sistem **TIDAK** mendiskualifikasi atau mengeluarkan peserta secara otomatis. Sistem berperan sebagai pendeteksi indikasi awal (*Suspicious Behavior Flagging*) dengan cara:
1. Memantau luminansi pencahayaan global, kepadatan tepi global (*Global Edge Density*), orientasi kepala (*Yaw* & *Pitch*), keberadaan wajah, dan jumlah orang dalam frame secara *real-time*.
2. Menerapkan *Calibration Gate* di awal sesi agar titik acuan orientasi kepala menyesuaikan posisi webcam tiap peserta secara adil, serta menerapkan analisis temporal (*Duration Threshold* + *Grace Period* anti-flicker) agar gerakan wajar sesaat tidak dianggap pelanggaran.
3. Mengurangi *Trust Score* (Skor Integritas), menampilkan eskalasi status integritas (`AMAN`, `WASPADA`, `TINJAUAN PENGAWAS`), mencatat log kejadian ke `log_indikasi_ujian.csv`, dan menyimpan bukti foto ber-anotasi ke direktori `bukti_indikasi/*.jpg`.

---

## 2. Ruang Lingkup Masalah (Mode 1: CBT Paperless / Ujian Teori)
1. **Model Ujian:** Dikhususkan untuk ujian teori berbasis komputer tanpa kertas buram, catatan fisik, maupun gawai/HP (contoh: ujian pilihan ganda, TWK, tes teori pemrograman).
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
4. **Ekstraksi Fitur Geometri Wajah (MediaPipe Tasks FaceLandmarker):**
   Mengekstrak 478 titik landmark wajah 2D (`RunningMode.VIDEO`), mengurutkan wajah berdasarkan luas *bounding box* terbesar sebagai peserta utama (Wajah #1), lalu menghitung rasio geometri:
   - **Rasio Yaw (Menoleh Horizontal):**
     $$\text{Ratio}_{yaw} = \frac{x_{\text{nose}} - x_{\text{left\_cheek}}}{\max(0.001, x_{\text{right\_cheek}} - x_{\text{left\_cheek}})}$$
   - **Rasio Pitch (Orientasi Vertikal Menunduk / Mendongak):**
     $$\text{Ratio}_{pitch} = \frac{y_{\text{nose}} - y_{\text{eye\_mid}}}{\max(0.001, y_{\text{chin}} - y_{\text{eye\_mid}})}$$
     dengan $y_{\text{eye\_mid}} = \frac{y_{\text{left\_eye}} + y_{\text{right\_eye}}}{2}$.
5. **Filtering Spasial pada ROI Wajah:**
   Ambil *Region of Interest* (ROI) wajah utama, sesuaikan skala ke $110\times 110$ piksel, konversi ke Grayscale + CLAHE, lalu terapkan konvolusi **Gaussian Blur kernel $5\times 5$** ($\sigma = 0$) untuk meredam noise frekuensi tinggi.
6. **Segmentasi Biner & Ekstraksi Tepi pada ROI:**
   - **Binarisasi Otsu:** Segmentasi adaptif otomatis memisahkan fitur wajah dari latar belakang (`cv2.THRESH_BINARY + cv2.THRESH_OTSU`).
   - **Deteksi Tepi Canny:** Operator gradien intensitas tepi dengan ambang `threshold1=50` dan `threshold2=130`.
   - **Metrik Edge Density ROI:** Dihitung dari persentase jumlah piksel tepi non-nol terhadap total ukuran ROI:
     $$\text{Edge Density} = \frac{\sum \text{Edge Pixels}}{\text{Total Pixels}} \times 100\%$$

---

## 4. Matriks Aturan Indikasi Kecurangan (*Rule-Based Temporal Logic*)
Evaluasi indikasi diproses setelah *Calibration Gate* aktif berdasarkan hierarki prioritas berikut:

| Prioritas | Kode | Jenis Perilaku | Ambang Pemicu | Ambang Durasi | Konsekuensi & Aksi |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | **IND-05** | Kamera Tertutup / Oklusi | $\mu_Y < 35.0$ **atau** (*Global Edge* $< 1.0\%$ & wajah $= 0$) | $\ge 2.0\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **2** | **IND-03** | Terdeteksi >1 Orang | Jumlah wajah $> 1$ | $\ge 1.0\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **3** | **IND-04** | Wajah Tidak Terdeteksi | Jumlah wajah $= 0$ (pencahayaan & tepi normal) | $\ge 3.0\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **4** | **IND-01** | Menoleh Kiri / Kanan | $\vert{}\text{Yaw} - \text{Baseline}_{yaw}\vert{} > 0.15$ | $\ge 2.5\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **5** | **IND-02** | Menunduk / Mendongak | $\vert{}\text{Pitch} - \text{Baseline}_{pitch}\vert{} > 0.08$ | $\ge 3.0\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **6** | **NORMAL** | Peserta Fokus Ujian | Tepat 1 wajah, pencahayaan normal, orientasi dalam batas baseline | *Real-time* | Banner Hijau, Reset Timer Perilaku setelah *grace period* $0.3\text{ s}$ |

*Aturan Tambahan:*
- **Calibration Gate:** Sebelum peserta menekan tombol kalibrasi `[C]` (dengan syarat tepat 1 wajah terdeteksi), pemotongan *Trust Score* dan pencatatan bukti dikunci.
- **Cooldown Penyimpanan:** Penyimpanan bukti screenshot dan penulisan log CSV dibatasi jeda minimal 5.0 detik (`LOG_COOLDOWN_SEC = 5.0`) untuk perilaku mencurigakan yang berlangsung kontinu.
- **Skor & Status Integritas:** *Trust Score* berawal dari `100%` (minimum `0%`) dengan kategori eskalasi:
  - `80% – 100%`: **AMAN** (Hijau)
  - `50% – 79%`: **WASPADA** (Kuning/Oranye)
  - `< 50%`: **TINJAUAN PENGAWAS** (Merah — rekomendasi evaluasi diskualifikasi oleh pengawas manusia)

---

## 5. Spesifikasi Antarmuka Visual (Dashboard 920x480 Piksel)
Dashboard dibagi menjadi dua panel utama:
1. **Panel Kamera Utama (Kiri: 640x480 piksel):**
   - Menampilkan feed video *real-time* yang telah dicerminkan.
   - *Bounding Box* Wajah: Kotak Biru-Cyan `(255, 200, 0)` untuk Wajah #1 (luas terbesar / peserta utama) dan Merah `(0, 0, 255)` untuk wajah tambahan.
   - Titik acuan hidung: Lingkaran kuning `(0, 255, 255)` pada ujung hidung Wajah #1.
   - *Status Banner* (Tinggi 40 piksel di sisi atas):
     - Biru Tua: `MENUNGGU KALIBRASI WAJAH: TEKAN [C] SAAT FOKUS KE LAYAR`
     - Hijau: `STATUS AMAN: PESERTA FOKUS UJIAN`
     - Oranye: `MEMANTAU GERAKAN: <PERILAKU> (<durasi>s / <ambang>s)`
     - Merah: `INDIKASI [<KODE>]: <PERILAKU> (<durasi>s)`
   - Baris informasi bawah: status kalibrasi di kiri bawah dan nilai numerik `Yaw` serta `Pitch` secara *real-time* di kanan bawah.
2. **Panel Monitoring PCD (Kanan: 280x480 piksel):**
   - Menampilkan metrik `Trust Score` (%), badge `Status` integritas (`AMAN` / `WASPADA` / `TINJAUAN PENGAWAS`), `Total Indikasi`, `Jumlah Wajah`, `Luminansi (Y)`, dan `Edge Density` (%).
   - **Grid 2x2 Visualisasi Tahapan PCD ($110\times 110$ piksel per kotak):**
     1. Kiri-Atas: `1.Gray+CLAHE`
     2. Kanan-Atas: `2.Gauss 5x5`
     3. Kiri-Bawah: `3.Otsu Biner`
     4. Kanan-Bawah: `4.Canny Edge`

---

## 6. Kontrol Pengguna & Skema Bukti Audit
- **Kontrol Keyboard:**
  - `[C]`: Kalibrasi posisi orientasi wajah acuan (`baseline_yaw`, `baseline_pitch`) saat tepat 1 wajah terdeteksi.
  - `[R]`: Reset `Trust Score` ke `100%`, `total_flags` ke `0`, dan mereset timer perilaku.
  - `[Q]`: Menghentikan sesi pemantauan dan menutup resource kamera dengan aman.
- **Skema `log_indikasi_ujian.csv`:**
  `Waktu,Kode_Indikasi,Jenis_Perilaku,Durasi_Detik,Mean_Luminance,Trust_Score,File_Bukti`
