# CLAUDE.md — AI Coding Agent Context & Operational Guide

File ini menyediakan konteks instan bagi AI Coding Assistant (Claude Code, Antigravity, Cursor, Copilot) saat bekerja di repositori **Sistem Deteksi Indikasi Kecurangan Ujian Online (Mode 1: CBT Paperless / PCD)**.

## 1. Urutan Sumber Kebenaran (*Source of Truth Hierarchy*)
Jika terjadi keraguan desain atau aturan, baca dokumen berikut sesuai urutan prioritas:
1. `AGENTS.md` — Kontrak gaya kode Python, larangan operasional, dan syarat penilaian akademis PCD.
2. `ARCHITECTURE-ESSENTIALS.md` — Ringkasan cepat *invariants*, konstanta ambang batas, dan urutan prioritas aturan.
3. `PRD.md` — Spesifikasi kebutuhan produk, rumus matematis, dan tata letak UI `920x480`.
4. `ARCHITECTURE.md` — Penjelasan mendalam aliran data antar-modul, *state machine*, dan kompensasi ergonomi VOR.

*Aturan Rekonsiliasi:* Apabila Anda mengubah konstanta di `src/config.py` atau alur di `src/detector.py`, perbarui `ARCHITECTURE-ESSENTIALS.md` dan `PRD.md` pada commit yang sama agar dokumentasi tidak pernah menyimpang dari kode.

## 2. Perintah Eksekusi & Verifikasi (Windows PowerShell)
Selalu gunakan interpreter di dalam folder `venv/` (tanpa titik di depan):

```powershell
# Menjalankan aplikasi pemantauan webcam
.\venv\Scripts\python.exe main.py

# Menjalankan seluruh unit test (wajib hijau sebelum commit)
.\venv\Scripts\python.exe -m pytest tests/test_detector_logic.py -v

# Memeriksa validitas sintaks seluruh modul
.\venv\Scripts\python.exe -m compileall main.py src tests
```

## 3. Standar Kode & Guardrails Singkat
- **Bahasa & Tipe Data:** Python 3.12+ dengan *Type Hints* penuh (`typing`, `numpy.typing.NDArray[np.uint8]`), `StrEnum` untuk kode status/indikasi, dan `@dataclass` untuk konfigurasi/state.
- **Larangan Keras:** Dilarang menambah paket ke `requirements.txt` tanpa izin pengguna, dilarang memakai `# type: ignore` / `# noqa`, dilarang memakai `time.time()` untuk durasi (wajib `time.monotonic()`), dan dilarang melakukan `git push` tanpa persetujuan eksplisit pengguna.
- **Definisi Selesai (*Definition of Done*):** Kode bebas *error* kompilasi, seluruh tes di `tests/test_detector_logic.py` lulus (`PASSED`), dokumentasi sinkron, dan tercatat dalam `git commit` berformat *Conventional Commits*.
