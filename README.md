# Eksperimen HSV Color Segmentation & Ripeness Classification Pisang
**Kelompok 7 - Pengolahan Citra Digital & Computer Vision**

Folder ini berisi implementasi eksperimental lengkap untuk membuktikan cara kerja **HSV Color Segmentation** dan klasifikasi tingkat kematangan buah pisang berdasarkan sintesis 10 artikel ilmiah (*literature review*).

---

## 1. Struktur Direktori Folder `Experimen`

```
Experimen/
├── venv/                       # Virtual environment Python (OpenCV, Matplotlib, NumPy)
├── dataset/
│   ├── sample_images/          # Simpan foto pisang asli Anda di sini (.jpg / .png)
│   └── synthetic_samples/      # 4 sampel citra pisang sintetis untuk uji coba instan
├── results/
│   ├── calibration/            # Hasil simpanan kalibrasi slider trackbar
│   ├── segmentation/           # Panel dekomposisi HSV, mask, dan bounding box
│   ├── ripeness_classification/# Laporan visual donat proporsi piksel & keputusan
│   └── experiment_summary.csv  # Rekapitulasi metrik numerik seluruh sampel uji
├── 01_calibrate_trackbar.py    # GUI slider interaktif untuk kalibrasi HSV real-time
├── 02_banana_segmentation.py   # Pipeline lengkap segmentasi morfologi & bounding box
├── 03_ripeness_classifier.py   # Klasifikasi kematangan multi-layer (Mentah, Mengkal, Matang, Overripe)
├── 04_batch_experiment.py      # Evaluasi otomatis batch & ekspor ke CSV
├── create_synthetic_banana.py  # Generator citra pisang sintetis
└── README.md                   # Dokumentasi teknis proyek
```

---

## 2. Cara Mengaktifkan Virtual Environment di VS Code

Buka folder **`Experimen`** di VS Code, lalu buka Terminal terintegrasi (`Ctrl + ~`) dan jalankan:
```bash
source venv/bin/activate
```
*(Seluruh pustaka seperti `opencv-python`, `matplotlib`, `numpy`, dan `pillow` sudah terpasang dan siap digunakan).*

---

## 3. Panduan Menjalankan Eksperimen

### Langkah 1: Kalibrasi Ambang Batas HSV (Interactive Trackbar)
Gunakan skrip ini untuk mencari rentang batas bawah (*Lower*) dan batas atas (*Upper*) pada foto pisang Anda secara langsung:
```bash
python 01_calibrate_trackbar.py dataset/sample_images/foto_pisang.jpg
```
* **Geser Slider:** $H, S, V$ untuk melihat bagian pisang yang tertangkap mask secara *real-time*.
* **Tekan 's':** Menyimpan parameter ke terminal dan menyimpan citra dokumentasi mask ke `results/calibration/`.
* **Tekan 'q' atau 'ESC':** Keluar dari jendela kalibrasi.

### Langkah 2: Menjalankan Segmentasi Detail & Dekomposisi Kanal
Mengekstrak pisang, membersihkan noise dengan operasi morfologi, dan membuat kotak pembatas (*Bounding Box*):
```bash
# Opsi target: yellow (kuning), green (hijau), atau all (seluruh tubuh pisang)
python 02_banana_segmentation.py dataset/synthetic_samples/pisang_3_matang.png yellow
```
*Hasil panel visual 6 subplot disimpan di:* `results/segmentation/`

### Langkah 3: Klasifikasi Tingkat Kematangan Pisang
Menganalisis proporsi piksel warna multi-layer dan menentukan fase kematangan:
```bash
python 03_ripeness_classifier.py dataset/synthetic_samples/pisang_3_matang.png
```
*Jika dijalankan tanpa argumen file, skrip akan menguji seluruh sampel di folder `dataset/synthetic_samples/` secara berurutan.*
*Hasil grafik donat dan visualisasi disimpan di:* `results/ripeness_classification/`

### Langkah 4: Evaluasi Batch Otomatis & Ekspor CSV
Mengevaluasi seluruh citra di dalam suatu folder dan mencatat persentase warna ke tabel:
```bash
python 04_batch_experiment.py dataset/synthetic_samples
```
*Hasil rekapitulasi numerik disimpan di:* `results/experiment_summary.csv`

---

## 4. Rentang Parameter Kalibrasi HSV Standar (OpenCV 8-bit)

Berdasarkan studi literatur (**Rizka et al., 2025**; **Putri et al., 2026**):
* **Pisang Mentah (Hijau):**
  * `Lower: [41, 50, 50]` | `Upper: [85, 255, 255]`
* **Pisang Mengkal (Kuning Kehijauan):**
  * `Lower: [33, 50, 60]` | `Upper: [40, 255, 255]`
* **Pisang Matang (Kuning Cerah):**
  * `Lower: [20, 70, 70]` | `Upper: [32, 255, 255]`
* **Bercak Senesen (Cokelat / Bintik Gula):**
  * `Lower: [5, 50, 30]` | `Upper: [19, 255, 160]`
