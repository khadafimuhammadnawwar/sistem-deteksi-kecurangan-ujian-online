# PRODUCT REQUIREMENTS DOCUMENT (PRD)
# Sistem Deteksi Indikasi Kecurangan pada Ujian Online Berbasis Citra Digital

## 1. Visi & Filosofi Sistem
Sistem ini adalah aplikasi *Computer-Assisted Proctoring* berbasis pengolahan citra digital dan *3D computer vision* menggunakan Python.

**Prinsip Kerja (Non-Auto Banning / Decision Support System):**
Sistem **TIDAK** mendiskualifikasi atau mengeluarkan peserta secara otomatis. Sistem berperan sebagai pendeteksi indikasi awal (*Suspicious Behavior Flagging*) dengan cara:
1. Memantau luminansi pencahayaan global, kepadatan tepi global (*Global Canny Edge Density*), proyeksi sinar pandang 3D (*L-Shaped 3D Gaze Ray-Casting* dalam satuan `cm`), keberadaan wajah, dan jumlah orang dalam frame secara *real-time*.
2. Mengunci geometri fisik perangkat di sisi pengawas (**Lenovo IdeaPad Slim 5 14IMH9**: layar `14.0"` `16:10` berukuran $30.16 \times 18.85\text{ cm}$, tinggi engsel $20.2\text{ cm}$, dan kedalaman titik akhir bawah touchpad $21.0\text{ cm}$) serta menerapkan **Kalibrasi 1-Titik Lensa Fisik Webcam dengan 5-Layer Anti-Spoof Gate** sehingga mustahil diakali peserta.
3. Mengurangi *Trust Score* (Skor Integritas), menampilkan eskalasi status integritas (`AMAN`, `WASPADA`, `TINJAUAN PENGAWAS`), mencatat log kejadian ke `log_indikasi_ujian.csv`, dan menyimpan bukti foto ber-anotasi ke direktori `bukti_indikasi/*.jpg`.

---

## 2. Ruang Lingkup Masalah (Mode 1: CBT Paperless / Ujian Teori)
1. **Model Ujian:** Dikhususkan untuk ujian teori berbasis komputer tanpa kertas buram, catatan fisik, maupun gawai/HP (area kerja sah mencakup seluruh permukaan layar laptop serta dek keyboard hingga tepat sebelum ujung akhir bawah touchpad $21.0\text{ cm}$).
2. **Perangkat:** 1 unit webcam bawaan laptop di tengah bezel atas `(0, 0, 0)` beresolusi 640x480 piksel menghadap wajah hingga batas bahu peserta.
3. **Kondisi Lingkungan:** Menggunakan algoritma CLAHE pada kanal luminansi CIE-LAB agar adaptif terhadap pencahayaan redup/tidak merata, dikombinasikan dengan deteksi tepi Canny untuk menangkap oklusi kamera oleh benda gelap maupun benda terang polos.

---

## 3. Pipeline Pengolahan Citra Digital (PCD) & Proyeksi 3D
Setiap frame video berukuran 640x480 piksel wajib diproses melalui 6 tahapan berurutan:

1. **Akuisisi Citra & Pencerminan (Mirroring):**
   Frame BGR diambil dari webcam dan dibalik secara horizontal (`cv2.flip(frame, 1)`).
2. **Konversi Ruang Warna, Luminansi Global & Deteksi Oklusi Tepi:**
   - Konversi BGR ke Grayscale 1-channel:
     $$Y = 0.299R + 0.587G + 0.114B$$
   - Hitung rata-rata intensitas piksel global ($\mu_Y$) untuk mendeteksi ruangan gelap/oklusi gelap ($\mu_Y < 35.0$).
   - Terapkan Gaussian Blur $5\times 5$ dan operator Canny (`threshold1=50`, `threshold2=130`) global untuk mendeteksi oklusi benda terang polos (*Global Edge Density* $< 1.0\%$ saat wajah $= 0$).
3. **Peningkatan Kontras Adaptif (CIE-LAB CLAHE):**
   - Konversi BGR ke ruang warna CIE-LAB, terapkan CLAHE (`clipLimit=2.5`, `tileGridSize=(8,8)`) pada kanal *Lightness* ($L$), lalu kembalikan ke BGR/RGB sebelum deteksi landmark.
4. **Ekstraksi Pose Metrik 3D & *L-Shaped 3D Gaze Ray-Casting* (`cm`):**
   - Mengekstrak 478 titik landmark beserta matriks transformasi metrik $4\times 4$ (`output_facial_transformation_matrixes=True`), mengurutkan Wajah #1 berdasarkan luas *bounding box* terbesar, dan meredam getaran dengan EMA ($\alpha = 0.42$).
   - Menghitung perpotongan sinar pandang 3D dari koordinat mata $(X_e, Y_e, Z_e)$ terhadap:
     - **Bidang Layar Vertikal ($Z = 0$):**
       $$X_{\text{scr}} = X_e + Z_e \tan(\theta_{\text{yaw}}), \quad Y_{\text{scr}} = Y_e + Z_e \tan(\theta_{\text{pitch}})$$
     - **Bidang Dek Keyboard & Touchpad Berengsel $\theta_{\text{lid}} = 90^\circ + \alpha_{\text{tilt}}$ (saat $Y_{\text{scr}} > H_{\text{hinge}} = 20.2\text{ cm}$):**
       $$Z_{\text{deck}} = \frac{Y_{\text{scr}} - H_{\text{hinge}}}{\sin\alpha_{\text{tilt}} + \cos\alpha_{\text{tilt}} \tan(\theta_{\text{pitch}})}, \quad X_{\text{deck}} = X_e + (Z_e - Z_{\text{deck}}\cos\alpha_{\text{tilt}})\tan(\theta_{\text{yaw}})$$
   - Dilengkapi proteksi kedipan alami ($\text{EAR} \ge 0.12$) agar kedipan tidak menimbulkan lonjakan sudut iris sementara pelacakan kepala 3D tetap aktif.
5. **Filtering Spasial pada ROI Wajah:**
   Potong *Region of Interest* (ROI) Wajah #1 ke $110\times 110$ piksel, terapkan Grayscale + CLAHE, lalu konvolusi **Gaussian Blur kernel $5\times 5$** ($\sigma = 0$).
6. **Segmentasi Biner & Ekstraksi Tepi pada ROI:**
   - **Binarisasi Otsu:** `cv2.THRESH_BINARY + cv2.THRESH_OTSU`.
   - **Deteksi Tepi Canny:** `threshold1=50`, `threshold2=130`, dengan metrik:
     $$\text{Edge Density} = \frac{\sum \text{Edge Pixels}}{\text{Total Pixels}} \times 100\%$$

---

## 4. Matriks Aturan Indikasi Kecurangan (*Hardware-Locked Temporal Logic*)
Evaluasi indikasi diproses setelah peserta lulus **Kalibrasi Lensa Anti-Spoof** berdasarkan hierarki prioritas berikut:

| Prioritas | Kode | Jenis Perilaku | Ambang Fisik Terkunci (`cm`) | Ambang Durasi | Konsekuensi & Aksi |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | **IND-05** | Kamera Tertutup / Oklusi | $\mu_Y < 35.0$ **atau** (*Global Edge* $< 1.0\%$ & wajah $= 0$) | $\ge 2.0\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **2** | **IND-03** | Terdeteksi >1 Orang | Jumlah wajah $> 1$ | $\ge 1.0\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **3** | **IND-04** | Wajah Tidak Terdeteksi | Jumlah wajah $= 0$ (pencahayaan & tepi normal) | $\ge 3.0\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **4** | **IND-01** | Menoleh / Melirik Kiri-Kanan | $|X_{\text{hit}}| > 22.08\text{ cm}$ ($\frac{30.16}{2} + 7.0\text{ cm}$ area pandang layar) | $\ge 2.5\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **5** | **IND-02** | Atas Layar / Lewat Touchpad | $Y_{\text{scr}} < -7.0\text{ cm}$ (atas layar) **atau** $Z_{\text{deck}} \ge 21.0\text{ cm}$ (akhir touchpad) | Kepala $\ge 3.0\text{ s}$ / Mata $\ge 2.5\text{ s}$ | Trust Score $-5\%$, Banner Merah, Simpan Bukti JPG & Log CSV |
| **6** | **NORMAL** | Fokus Layar & Keyboard | Tepat 1 wajah, $|X_{\text{hit}}| \le 22.08\text{ cm}$, $-7.0 \le Y_{\text{scr}} \le 20.2\text{ cm}$, $Z_{\text{deck}} < 21.0\text{ cm}$ | *Real-time* | Banner Hijau, Reset Timer setelah *grace period* $0.3\text{ s}$ |

*Keamanan Kalibrasi 5-Layer Anti-Spoof (`[SPASI]` / `[C]`, durasi `2.0 s`):*
- Peserta diminta menatap lurus **Lensa Fisik Webcam `(0, 0, 0)`** hanya untuk mengukur offset anatomi *Angle-Kappa* alami serta sudut kemiringan engsel layar ($90^\circ\text{–}115^\circ$).
- Kalibrasi menyaring kedipan alami sesaat ($\text{EAR} \ge 0.16$) namun **DITOLAK OTOMATIS** apabila jarak di luar `25–85 cm`, mata terus menyipit/terpejam (`< 5` sampel terbuka), pandangan bergoyang tidak stabil, kepala menoleh $> 6^\circ$, peserta duduk $> 6\text{ cm}$ dari tengah layar, atau tatapan menyimpang dari lensa (offset yaw $> 4.5^\circ$; offset pitch di luar $-2^\circ \ldots +8^\circ$, batas atas bergeser $+(\theta_{\text{lid}} - 90^\circ)$; offset yaw yang dipakai dijepit maks $\pm 2.5^\circ$). Pesan penolakan menampilkan nilai yaw/pitch terukur untuk penyetelan. Sumbu horizontal laptop simetris fisik ($X = 0$) sehingga batas zona aman fisik `cm` **tidak pernah dapat diperlebar atau dibelokkan oleh peserta**.

---

## 5. Spesifikasi Antarmuka Visual (Dashboard 920x480 Piksel)
1. **Panel Kamera Utama (Kiri: 640x480 piksel):**
   - *Bounding Box* Wajah (`Wajah #1` Biru-Cyan, tambahan Merah), titik kuning hidung, 2 titik magenta pupil, dan panah pemandu ke Lensa Fisik Webcam saat kalibrasi.
   - *Status Banner* atas dinamis dan HUD bawah berformat sentimeter *real-time* (`X_cm`, `Y_cm` / `Touchpad: xx.x/21cm`, `Z_cm`).
2. **Panel Monitoring PCD (Kanan: 280x480 piksel):**
   - Metrik `Trust Score` (`AMAN` `80–100%`, `WASPADA` `50–79%`, `TINJAUAN PENGAWAS` `< 50%`), `Total Indikasi`, `Jumlah Wajah`, `Luminansi (Y)`, `Edge ROI/Glob`, serta **Grid 2x2 ROI PCD ($110\times 110$)**.

---

## 6. Kontrol Pengguna & Skema Bukti Audit
- **Kontrol Keyboard:** `[SPASI]` / `[C]` Kalibrasi Lensa Kamera (Anti-Spoof 2s), `[R]` Reset Skor & Timer, `[Q]` Keluar (`Cooldown` log = `5.0 s`).
- **Skema `log_indikasi_ujian.csv`:** `Waktu,Kode_Indikasi,Jenis_Perilaku,Durasi_Detik,Mean_Luminance,Trust_Score,File_Bukti`
