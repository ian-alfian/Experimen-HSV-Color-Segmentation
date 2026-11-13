"""
04_BATCH_EXPERIMENT.PY
----------------------
Skrip Evaluasi Otomatis Batch Experiment:
Memproses sekumpulan citra pisang, mengekstrak metrik segmentasi dan klasifikasi,
serta mengekspor seluruh hasil ke dalam berkas CSV (results/experiment_summary.csv).
"""

import cv2
import numpy as np
import os
import sys
import csv

def run_batch_experiment(input_dir="dataset/sample_images", output_csv="results/experiment_summary.csv"):
    os.makedirs("results", exist_ok=True)

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])

    if not files:
        print(f"Peringatan: Tidak ada berkas citra di '{input_dir}'.")
        return

    # Definisi rentang warna kematangan pisang
    ranges = {
        'green': (np.array([41, 50, 50]), np.array([85, 255, 255])),
        'turning': (np.array([33, 50, 60]), np.array([40, 255, 255])),
        'yellow': (np.array([20, 70, 70]), np.array([32, 255, 255])),
        'brown': (np.array([5, 50, 30]), np.array([19, 255, 160]))
    }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    results = []

    print("=" * 75)
    print(f"MENJALANKAN BATCH EKSPERIMEN PADA {len(files)} CITRA")
    print(f"Direktori Input: {input_dir}")
    print("=" * 75)

    for fpath in files:
        fname = os.path.basename(fpath)
        img = cv2.imread(fpath)
        if img is None:
            continue

        h, w = img.shape[:2]
        total_pixels = h * w

        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Hitung piksel untuk setiap komponen
        px_counts = {}
        for name, (low, up) in ranges.items():
            m = cv2.inRange(hsv, low, up)
            m_clean = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
            px_counts[name] = cv2.countNonZero(m_clean)

        total_banana_px = sum(px_counts.values())
        if total_banana_px == 0:
            total_banana_px = 1

        pct_green = (px_counts['green'] / total_banana_px) * 100.0
        pct_turning = (px_counts['turning'] / total_banana_px) * 100.0
        pct_yellow = (px_counts['yellow'] / total_banana_px) * 100.0
        pct_brown = (px_counts['brown'] / total_banana_px) * 100.0

        # Penentuan Kelas Berdasarkan Dominasi Warna
        if pct_brown >= 50.0 or (pct_brown > pct_yellow and pct_brown > pct_green):
            pred_class = "Terlalu Matang (Overripe)"
        elif pct_green >= 40.0 or (pct_green > pct_yellow and pct_green > pct_turning):
            pred_class = "Mentah (Unripe)"
        elif pct_turning >= 20.0 or (pct_turning + pct_green) > pct_yellow:
            pred_class = "Mengkal (Turning)"
        else:
            pred_class = "Matang (Ripe)"

        results.append({
            'Filename': fname,
            'Resolution': f"{w}x{h}",
            'Total_Banana_Pixels': total_banana_px,
            'Green_Pixels': px_counts['green'],
            'Green_Pct': round(pct_green, 2),
            'Turning_Pixels': px_counts['turning'],
            'Turning_Pct': round(pct_turning, 2),
            'Yellow_Pixels': px_counts['yellow'],
            'Yellow_Pct': round(pct_yellow, 2),
            'Brown_Pixels': px_counts['brown'],
            'Brown_Pct': round(pct_brown, 2),
            'Predicted_Ripeness': pred_class
        })

        print(f"[{fname}] -> Prediksi: {pred_class:<25} (Kuning: {pct_yellow:5.1f}%, Hijau: {pct_green:5.1f}%, Cokelat: {pct_brown:5.1f}%)")

    # Ekspor ke CSV
    fieldnames = [
        'Filename', 'Resolution', 'Total_Banana_Pixels',
        'Green_Pixels', 'Green_Pct', 'Turning_Pixels', 'Turning_Pct',
        'Yellow_Pixels', 'Yellow_Pct', 'Brown_Pixels', 'Brown_Pct',
        'Predicted_Ripeness'
    ]

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 75)
    print(f"Eksperimen batch selesai! Hasil tersimpan di: {output_csv}")
    print("=" * 75)

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/sample_images"
    run_batch_experiment(input_dir=target_dir)

if __name__ == "__main__":
    main()
