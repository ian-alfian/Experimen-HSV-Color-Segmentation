# Eksperimen HSV Color Segmentation & Ripeness Classification Pisang
**Mata Kuliah: Computer Vision**  
**Kelompok 7**

Repositori ini memuat implementasi lengkap penelitian dan eksperimen **HSV Color Segmentation** serta **Sistem Klasifikasi Tingkat Kematangan Buah Pisang** (*Musa paradisiaca*) berbasis bahasa pemrograman Python dan pustaka OpenCV. Metodologi yang diterapkan mengacu pada sintesis 10 artikel ilmiah bereputasi di bidang pemrosesan citra digital.

---

## Daftar Isi
1. [Struktur Direktori Proyek](#1-struktur-direktori-proyek)
2. [Prasyarat & Dependensi Sistem](#2-prasyarat--dependensi-sistem)
3. [Panduan Instalasi & Setup Environment](#3-panduan-instalasi--setup-environment)
4. [Tutorial Langkah-demi-Langkah Menjalankan Program](#4-tutorial-langkah-demi-langkah-menjalankan-program)
   - [Langkah 1: Kalibrasi HSV Interaktif (01_calibrate_trackbar.py)](#langkah-1-kalibrasi-hsv-interaktif-01_calibrate_trackbarpy)
   - [Langkah 2: Pipeline Segmentasi & Morfologi (02_banana_segmentation.py)](#langkah-2-pipeline-segmentasi--morfologi-02_banana_segmentationpy)
   - [Langkah 3: Klasifikasi Kematangan Multi-Layer (03_ripeness_classifier.py)](#langkah-3-klasifikasi-kematangan-multi-layer-03_ripeness_classifierpy)
   - [Langkah 4: Evaluasi Batch Otomatis & Ekspor CSV (04_batch_experiment.py)](#langkah-4-evaluasi-batch-otomatis--ekspor-csv-04_batch_experimentpy)
5. [Tabel Parameter Ambang Batas HSV (OpenCV 8-Bit)](#5-tabel-parameter-ambang-batas-hsv-opencv-8-bit)
6. [Panduan Troubleshooting & Masalah Umum](#6-panduan-troubleshooting--masalah-umum)
7. [Daftar Pustaka Rujukan](#7-daftar-pustaka-rujukan)

---

## 1. Struktur Direktori Proyek

Pastikan Anda berada di dalam folder `Experimen/` saat mengeksekusi skrip:

```text
Experimen/
├── venv/                           # Virtual environment Python terisolasi
├── dataset/
│   └── sample_images/              # Dataset foto pisang riil (.jpg, .jpeg, .png)
│       ├── pisang_mentah.jpeg       # Sampel pisang hijau (Unripe)
│       ├── pisang_mengkal.jpg      # Sampel pisang transisi hijau-kuning (Turning)
│       ├── pisang_matang.jpg       # Sampel pisang kuning cerah (Ripe)
│       └── pisang_terlalu_matang.jpeg # Sampel pisang cokelat senesen (Overripe)
├── results/
│   ├── calibration/                # Hasil ekspor snapshot dari trackbar kalibrasi
│   ├── segmentation/               # Panel visualisasi 6-subplot, mask, & foreground
│   ├── ripeness_classification/    # Visualisasi donat proporsi warna & mask multi-layer
│   └── experiment_summary.csv      # Rekapitulasi metrik numerik seluruh sampel citra
├── 01_calibrate_trackbar.py        # GUI slider OpenCV untuk pencarian threshold HSV real-time
├── 02_banana_segmentation.py       # Pipeline utama segmentasi, morfologi, & lokalisasi spasial
├── 03_ripeness_classifier.py       # Algoritma klasifikasi kematangan berbasis dominasi piksel
├── 04_batch_experiment.py          # Eksekusi batch otomatis seluruh dataset & ekspor CSV
└── README.md                       # Dokumentasi teknis dan panduan penggunaan
```

---

## 2. Prasyarat & Dependensi Sistem

Sistem telah diuji dan berjalan optimal pada lingkungan:
* **Sistem Operasi:** Linux (Arch Linux / Ubuntu / Debian), macOS, atau Windows 10/11
* **Versi Python:** Python 3.10 ke atas (teruji pada Python 3.14.x)
* **Pustaka Utama:**
  * `opencv-python` (>= 4.8.0 / 5.0.0)
  * `numpy` (>= 1.24.0 / 2.5.4)
  * `matplotlib` (>= 3.8.0 / 3.11.2)
  * `pillow` (>= 10.0.0)

---

## 3. Panduan Instalasi & Setup Environment

### A. Membuka Terminal pada Folder Proyek
Buka terminal dan arahkan ke direktori kerja `Experimen`:
```bash
cd "/home/alfian/Documents/CV Kel 7/Experimen"
```

### B. Mengaktifkan Virtual Environment
Virtual environment digunakan agar dependensi pustaka tidak bertabrakan dengan Python sistem operasi.

* **Pada Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```
  *(Indikator terminal akan menampilkan tanda `(venv)` di sebelah kiri prompt)*.

* **Pada Windows (Command Prompt / PowerShell):**
  ```cmd
  venv\Scripts\activate
  ```

> **Catatan Penting:**  
> Jika muncul error `ModuleNotFoundError: No module named 'cv2'`, hal tersebut menandakan virtual environment belum aktif. Jalankan perintah `source venv/bin/activate` terlebih dahulu, atau panggil interpreter langsung melalui `./venv/bin/python <nama_file.py>`.

### C. Instalasi Dependensi (Jika Membuat Environment Baru)
Jika virtual environment perlu dibangun ulang dari awal:
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install opencv-python numpy matplotlib pillow
```

---

## 4. Tutorial Langkah-demi-Langkah Menjalankan Program

### Langkah 1: Kalibrasi HSV Interaktif (`01_calibrate_trackbar.py`)
Skrip ini menyediakan antarmuka grafis (GUI) interaktif dengan 6 slider (*trackbars*) untuk menemukan batas bawah (*Lower*) dan batas atas (*Upper*) ruang warna HSV secara langsung (*real-time*).

```bash
# Menjalankan kalibrasi pada citra tertentu:
python 01_calibrate_trackbar.py dataset/sample_images/pisang_matang.jpg

# Atau jalankan tanpa argumen (otomatis memilih citra pertama di dataset):
python 01_calibrate_trackbar.py
```

* **Tata Letak Tampilan:**
  * **Panel Kiri:** Citra input asli (disertai indikator jumlah piksel aktif dan persentase luas area).
  * **Panel Tengah:** *Binary Mask* hasil thresholding dan operasi morfologi pembersihan.
  * **Panel Kanan:** Objek *foreground* hasil ekstraksi *Bitwise AND*.
* **Kontrol Keyboard:**
  * Geser slider `H Min`, `H Max`, `S Min`, `S Max`, `V Min`, `V Max` untuk mengatur ambang batas.
  * Tekan tombol **`s`** pada keyboard: Mencetak nilai threshold numerik ke terminal dan menyimpan citra snapshot mask serta segmented ke folder `results/calibration/`.
  * Tekan tombol **`q`** atau tombol **`ESC`**: Menutup jendela kalibrasi.

---

### Langkah 2: Pipeline Segmentasi & Morfologi (`02_banana_segmentation.py`)
Skrip ini menjalankan pipeline lengkap pemrosesan citra digital:
1. Denoising menggunakan filter Gaussian ($5 \times 5$).
2. Konversi ruang warna BGR ke HSV.
3. Dekomposisi 3 kanal mandiri: Hue ($H$), Saturation ($S$), dan Value ($V$).
4. Thresholding ambang batas `cv2.inRange()`.
5. Pemurnian morfologi: *Opening* (kernel elips $5 \times 5$, 1 iterasi) untuk membuang bintik noise, *Closing* (kernel elips $7 \times 7$, 2 iterasi) untuk menutup lubang pantulan cahaya, serta eliminasi komponen selain *Largest Connected Component*.
6. Ekstraksi objek *foreground* melalui operasi logika *Bitwise AND*.
7. Deteksi kontur eksternal, kalkulasi luas area kontur, dan penggambaran kotak pembatas (*Bounding Box*).
8. Ekspor panel komprehensif 6-subplot ($2 \times 3$) beresolusi tinggi (200 DPI).

**Sintaks Penggunaan:**
```bash
python 02_banana_segmentation.py <path_citra> <mode_target>
```

**Pilihan `<mode_target>`:**
* `yellow` : Segmentasi warna kuning (untuk sampel pisang matang).
* `green` : Segmentasi warna hijau (untuk sampel pisang mentah).
* `brown` atau `overripe` : Segmentasi warna cokelat/bercak senesen (untuk pisang terlalu matang).
* `all` : Segmentasi seluruh tubuh buah pisang (rentang spektrum luas $H: 5\text{--}85$).

**Contoh Perintah Eksekusi:**
```bash
# 1. Segmentasi pisang mentah (Target Hijau):
python 02_banana_segmentation.py dataset/sample_images/pisang_mentah.jpeg green

# 2. Segmentasi pisang mengkal (Target Seluruh Tubuh Pisang):
python 02_banana_segmentation.py dataset/sample_images/pisang_mengkal.jpg all

# 3. Segmentasi pisang matang (Target Kuning Cerah):
python 02_banana_segmentation.py dataset/sample_images/pisang_matang.jpg yellow

# 4. Segmentasi pisang terlalu matang (Target Cokelat Senesen):
python 02_banana_segmentation.py dataset/sample_images/pisang_terlalu_matang.jpeg overripe
```

**Berkas Luaran (`results/segmentation/`):**
* `<nama_file>_segmentation_panel.png` : Gambar panel multi-plot visualisasi ilmiah.
* `<nama_file>_mask.png` : Citra biner murni hasil pemurnian morfologi.
* `<nama_file>_segmented.png` : Citra objek buah terisolasi dengan latar belakang hitam.

---

### Langkah 3: Klasifikasi Kematangan Multi-Layer (`03_ripeness_classifier.py`)
Skrip ini menerapkan segmentasi multi-ambang secara simultan untuk menghitung sebaran persentase luas piksel dari 4 pigmen utama:
* **Hijau (Mentah):** Representasi klorofil dominan.
* **Kuning Kehijauan (Mengkal):** Fase transisi degradasi klorofil ke karotenoid.
* **Kuning Cerah (Matang):** Dominasi pigmen karotenoid dan lutein.
* **Cokelat (Senesen/Overripe):** Akumulasi melanin akibat aktivitas polifenol oksidase (PPO).

**Logika Pengambilan Keputusan (Rule-based Decision):**
* **Terlalu Matang (Overripe):** Persentase Cokelat $\ge 50\%$ atau Cokelat mendominasi Kuning dan Hijau.
* **Mentah (Unripe):** Persentase Hijau $\ge 40\%$ atau Hijau mendominasi Kuning dan Mengkal.
* **Mengkal / Setengah Matang (Turning):** Persentase Mengkal $\ge 20\%$ atau (Mengkal + Hijau) $>$ Kuning.
* **Matang Optimal (Ripe):** Persentase Kuning mendominasi secara merata.

**Contoh Perintah Eksekusi:**
```bash
# Menguji satu sampel citra spesifik:
python 03_ripeness_classifier.py dataset/sample_images/pisang_matang.jpg

# Menguji seluruh sampel citra yang ada di dataset/sample_images/ sekaligus:
python 03_ripeness_classifier.py
```

**Berkas Luaran (`results/ripeness_classification/`):**
* `<nama_file>_ripeness_result.png` : Grafik donat persentase pigmen warna, citra tersegmentasi, serta 3 panel mask individual.

---

### Langkah 4: Evaluasi Batch Otomatis & Ekspor CSV (`04_batch_experiment.py`)
Skrip ini memproses kumpulan citra secara otomatis tanpa memunculkan antarmuka grafis (*headless batch mode*), mengekstrak seluruh metrik kuantitatif, dan merekapitulasi hasilnya ke berkas CSV terstruktur.

```bash
# Menjalankan evaluasi batch pada folder default (dataset/sample_images):
python 04_batch_experiment.py

# Atau tentukan path folder tertentu secara kustom:
python 04_batch_experiment.py dataset/sample_images
```

**Berkas Luaran:**
* Berkas rekapitulasi tersimpan di: **`results/experiment_summary.csv`**.

**Cara Meninjau Hasil CSV di Terminal Linux:**
```bash
column -t -s, results/experiment_summary.csv
```

---

## 5. Tabel Parameter Ambang Batas HSV (OpenCV 8-Bit)

Berikut adalah parameter kalibrasi standar yang terbukti efektif membedakan fase kematangan pisang:

| Komponen Warna / Target | Lower Bound `[H, S, V]` | Upper Bound `[H, S, V]` | Karakteristik Fisik Buah |
| :--- | :---: | :---: | :--- |
| **Hijau (Mentah / Unripe)** | `[35, 50, 50]` / `[41, 50, 50]` | `[85, 255, 255]` | Kulit keras, pigmen klorofil tinggi |
| **Kuning Kehijauan (Mengkal)** | `[33, 50, 60]` | `[40, 255, 255]` | Fase transisi pematangan |
| **Kuning Cerah (Matang / Ripe)** | `[19, 70, 70]` / `[20, 70, 70]` | `[34, 255, 255]` | Kulit kuning cerah, aroma manis |
| **Cokelat (Bercak Senesen / Overripe)** | `[5, 40, 30]` / `[5, 50, 30]` | `[19, 255, 180]` | Bintik gula, pencokelatan enzimatik |
| **Total Tubuh Pisang (All Spectrum)** | `[5, 40, 30]` | `[85, 255, 255]` | Menangkap seluruh variasi warna pisang |

> **Konversi Skala:** OpenCV merepresentasikan $H \in [0, 179]$ (skala setengah dari $0^\circ\text{--}360^\circ$), $S \in [0, 255]$, dan $V \in [0, 255]$.

---

## 6. Panduan Troubleshooting & Masalah Umum

### 1. `ModuleNotFoundError: No module named 'cv2'`
* **Penyebab:** Perintah dijalankan menggunakan Python global sistem yang belum terinstal OpenCV, bukan virtual environment proyek.
* **Solusi:** Aktifkan venv dengan `source venv/bin/activate` atau panggil langsung via `./venv/bin/python <skrip.py>`.

### 2. `error: (-215:Assertion failed) !_src.empty()`
* **Penyebab:** Path citra salah ketik, berkas tidak ditemukan, atau format ekstensi tidak sesuai.
* **Solusi:** Pastikan path citra diawali dari folder kerja yang benar, contoh: `dataset/sample_images/pisang_matang.jpg`.

### 3. Jendela GUI Kalibrasi Terlalu Besar / Melebihi Resolusi Monitor
* **Penyebab:** Resolusi kamera foto asli sangat tinggi (misal $960 \times 1280$ atau lebih).
* **Solusi:** Skrip `01_calibrate_trackbar.py` telah dilengkapi fitur auto-scaling (`max_dim = 650`), citra akan otomatis disesuaikan secara proporsional ke dimensi layar.

### 4. Background Tersegmentasi (False Positive)
* **Penyebab:** Objek latar belakang (seperti keranjang anyaman, meja kayu kekuningan, atau taplak bermotif) memiliki rona warna (*Hue*) yang beririsan dengan objek pisang.
* **Solusi:**
  1. Naikkan batas bawah *Saturation* ($S \ge 70\text{--}80$) untuk memfilter warna pudar latar belakang.
  2. Gunakan latar belakang netral yang kontras (seperti kain putih/hitam polos) saat pengambilan foto.

---

## 7. Daftar Pustaka Rujukan

1. **Akbar, Z., Suwondo, A., Ramadhan, R., & Hasugian, A. H. (2025).** Deteksi warna dasar menggunakan metode thresholding HSV dengan OpenCV. *Neptunus: Jurnal Ilmu Komputer dan Teknologi Informasi*, 3(3), 157–165. https://doi.org/10.61132/neptunus.v3i3.1020
2. **Amrozi, M. A., Figo SW, D., & Wahyusari, R. (2024).** Perbandingan segmentasi ruang warna HSV dan YCbCr untuk deteksi objek. *INFOMATEK: Jurnal Informatika, Manajemen dan Teknologi*, 26(2). https://doi.org/10.23969/infomatek.v26i2.19025
3. **Dowarah, B., Borah, P., Laskar, R. A., Yasin, A., Kashyap, A., & Neog, B. (2026).** A MATLAB-based image processing protocol for quantitative differentiation of diseased and healthy plant tissue from digital leaf images. *Bio-protocol*, 5798.
4. **Herriyance, Panjaitan, Z., Dahria, M., & Andika, B. (2025).** Penerapan Hue, Saturation, Value (HSV) dengan Sobel dalam segmentasi citra digital untuk deteksi kematangan tomat. *Jurnal Sistem Informasi TGD*, 4(6), 1470–1480.
5. **Muratbekova, M., Toganas, N., Igali, A., Shagyrov, M., Kadyrgali, E., Yerkin, A., & Shamoi, P. (2026).** Color models in image processing: A review and experimental comparison. *Discover Applied Sciences*, 8, 494. https://doi.org/10.1007/s42452-025-08192-7
6. **Obite, E., Moko, A. B., & Elliot, K. N. (2026).** Texture-guided adaptive HSV segmentation for constrained-environment object detection: A classical pipeline with per-image saturation threshold modulation. *American Journal of Computer Science and Technology*, 9(2), 84–94. https://doi.org/10.11648/j.ajcst.20260902.15
7. **Perez-Udell, R. A., Udell, A. T., & Chang, S.-M. (2023).** An automated pipeline for supervised classification of petal color from citizen science photographs. *Applications in Plant Sciences*, 11(1), e11505. https://doi.org/10.1002/aps3.11505
8. **Putri, A. S. K., Mubin, F. N., Nugroho, R. F., & Yasin, M. (2026).** Identifikasi tingkat kematangan cabai berdasarkan analisis warna menggunakan metode HSV Color Threshold. *Jurnal Janitra Informatika dan Sistem Informasi*, 6(2), 77–85. https://doi.org/10.59395/exr6sa76
9. **Rizka, Nasution, S., Aulia, F., & Supiyandi. (2025).** Penerapan metode segmentasi warna HSV untuk deteksi objek berbasis warna pada citra digital. *Router: Jurnal Teknik Informatika dan Terapan*, 3(4), 11–24. https://doi.org/10.62951/router.v3i4.706
10. **Rui, J., You, M., Wei, H., Wang, Y., Yan, X., Zhou, L., Zhu, C., & Ren, H. (2026).** Image recognition of dyed fibers and component analysis of cigarette paper based on Hue, Saturation, and Value (HSV) threshold segmentation. *BioResources*, 21(2), 4408–4435. https://doi.org/10.15376/biores.21.2.4408-4435
