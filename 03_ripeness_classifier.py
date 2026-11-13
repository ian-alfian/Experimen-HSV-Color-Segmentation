"""
03_RIPENESS_CLASSIFIER.PY
-------------------------
Klasifikasi Tingkat Kematangan Pisang Otomatis Berdasarkan Analisis
Distribusi Piksel Warna Multi-Layer HSV.
Rujukan Metodologi: Putri et al. (2026), Adenugraha et al., & Rizka et al. (2025).

Kategori Kematangan:
1. Mentah (Unripe) - Dominan Hijau
2. Mengkal (Turning) - Transisi Kuning Kehijauan
3. Matang (Ripe) - Dominan Kuning Cerah
4. Terlalu Matang (Overripe) - Kuning dengan Bintik/Bercak Cokelat Senesen

Penggunaan:
    python 03_ripeness_classifier.py [path_ke_foto_pisang]
    Contoh: python 03_ripeness_classifier.py dataset/synthetic_samples/pisang_1_mentah.png
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

def classify_banana_ripeness(image_path, save_dir="results/ripeness_classification"):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"Error: Tidak dapat membaca citra dari '{image_path}'.")
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 1. Definisi Rentang HSV Multi-Layer (Sesuai Referensi Rizka et al. & Kalibrasi Pisang)
    ranges = {
        'Hijau (Mentah)': {
            'lower': np.array([41, 50, 50]),
            'upper': np.array([85, 255, 255]),
            'color': '#2ECC71'
        },
        'Kuning Kehijauan (Mengkal)': {
            'lower': np.array([33, 50, 60]),
            'upper': np.array([40, 255, 255]),
            'color': '#A2D200'
        },
        'Kuning Cerah (Matang)': {
            'lower': np.array([20, 70, 70]),
            'upper': np.array([32, 255, 255]),
            'color': '#F1C40F'
        },
        'Cokelat (Bercak Senesen)': {
            'lower': np.array([5, 50, 30]),
            'upper': np.array([19, 255, 160]),
            'color': '#795548'
        }
    }

    # 2. Ekstraksi Mask untuk Setiap Komponen Warna
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    counts = {}
    masks = {}

    for name, conf in ranges.items():
        m = cv2.inRange(hsv, conf['lower'], conf['upper'])
        # Morfologi ringan
        m_clean = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        masks[name] = m_clean
        counts[name] = cv2.countNonZero(m_clean)

    total_banana_px = sum(counts.values())
    if total_banana_px == 0:
        total_banana_px = 1  # Hindari pembagian dengan nol jika tidak ada pisang

    percentages = {k: (v / total_banana_px) * 100.0 for k, v in counts.items()}

    # 3. Logika Keputusan Tingkat Kematangan
    p_green = percentages['Hijau (Mentah)']
    p_turning = percentages['Kuning Kehijauan (Mengkal)']
    p_yellow = percentages['Kuning Cerah (Matang)']
    p_brown = percentages['Cokelat (Bercak Senesen)']

    # Aturan keputusan adaptif berdasarkan dominasi warna
    if p_brown >= 50.0 or (p_brown > p_yellow and p_brown > p_green):
        decision = "TERLALU MATANG (OVERRIPE)"
        desc = "Pisang telah sangat matang dengan dominasi warna cokelat/kehitaman (bintik senesen dan pembusukan)."
        badge_color = (0, 0, 180)  # Dark Red/Brown in BGR
    elif p_green >= 40.0 or (p_green > p_yellow and p_green > p_turning):
        decision = "MENTAH (UNRIPE)"
        desc = "Pisang masih mentah dengan dominasi pigmen klorofil hijau (> 40%)."
        badge_color = (0, 150, 0)  # Green in BGR
    elif p_turning >= 20.0 or (p_turning + p_green) > p_yellow:
        decision = "MENGKAL / SETENGAH MATANG (TURNING)"
        desc = "Pisang dalam fase transisi pematangan dengan perpaduan warna hijau dan kuning."
        badge_color = (0, 180, 200)  # Orange-Yellow in BGR
    else:
        decision = "MATANG OPTIMAL (RIPE)"
        desc = "Pisang matang sempurna didominasi oleh warna kuning cerah merata."
        badge_color = (0, 200, 240)  # Vivid Yellow in BGR

    print("=" * 65)
    print(f"HASIL KLASIFIKASI KEMATANGAN PISANG: {base_name}")
    print("=" * 65)
    print(f"Keputusan Akhir : {decision}")
    print(f"Keterangan      : {desc}")
    print("-" * 65)
    print("Distribusi Proporsi Piksel Warna:")
    for k in counts:
        print(f" - {k:<28}: {counts[k]:>8,} px ({percentages[k]:>6.2f}%)")
    print(f"Total Piksel Tubuh Pisang : {total_banana_px:>8,} px")
    print("=" * 65)

    # 4. Visualisasi Multi-Panel & Grafik Donut/Bar
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)

    # Citra Asli dengan Banner Keputusan
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(rgb)
    ax_orig.set_title(f"Input: {base_name}")
    ax_orig.axis('off')

    # Mask Gabungan Seluruh Tubuh Pisang
    combined_mask = np.zeros_like(list(masks.values())[0])
    for m in masks.values():
        combined_mask = cv2.bitwise_or(combined_mask, m)
    
    # Rekonstruksi Foreground Berwarna
    seg_banana = cv2.bitwise_and(bgr, bgr, mask=combined_mask)
    seg_banana_rgb = cv2.cvtColor(seg_banana, cv2.COLOR_BGR2RGB)

    ax_seg = fig.add_subplot(gs[0, 1])
    ax_seg.imshow(seg_banana_rgb)
    ax_seg.set_title("Ekstraksi Tubuh Pisang")
    ax_seg.axis('off')

    # Grafik Donut Distribusi Warna HSV
    ax_pie = fig.add_subplot(gs[0, 2])
    labels = list(percentages.keys())
    sizes = [percentages[k] for k in labels]
    colors = [ranges[k]['color'] for k in labels]
    # Filter hanya yang ada nilainya
    filtered_data = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0.5]
    if filtered_data:
        f_labels, f_sizes, f_colors = zip(*filtered_data)
        wedges, texts, autotexts = ax_pie.pie(
            f_sizes, labels=f_labels, autopct='%1.1f%%',
            colors=f_colors, startangle=140, pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor='w')
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_weight('bold')
    ax_pie.set_title("Proporsi Warna HSV (%)")

    # Baris Bawah: Panel Mask Individual 4 Komponen
    ax_m1 = fig.add_subplot(gs[1, 0])
    ax_m1.imshow(masks['Hijau (Mentah)'], cmap='gray')
    ax_m1.set_title(f"Mask Hijau: {percentages['Hijau (Mentah)']:.1f}%")
    ax_m1.axis('off')

    ax_m2 = fig.add_subplot(gs[1, 1])
    ax_m2.imshow(masks['Kuning Cerah (Matang)'], cmap='gray')
    ax_m2.set_title(f"Mask Kuning: {percentages['Kuning Cerah (Matang)']:.1f}%")
    ax_m2.axis('off')

    ax_m3 = fig.add_subplot(gs[1, 2])
    ax_m3.imshow(masks['Cokelat (Bercak Senesen)'], cmap='gray')
    ax_m3.set_title(f"Mask Cokelat: {percentages['Cokelat (Bercak Senesen)']:.1f}%")
    ax_m3.axis('off')

    plt.suptitle(f"KLASIFIKASI KEMATANGAN: {decision}\n{desc}", fontsize=13, weight='bold', y=0.98)
    plt.tight_layout()

    out_plot = os.path.join(save_dir, f"{base_name}_ripeness_result.png")
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print(f"Laporan visual disimpan di: {out_plot}\n")

    return {
        'decision': decision,
        'counts': counts,
        'percentages': percentages,
        'out_plot': out_plot
    }

def main():
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        classify_banana_ripeness(img_path)
    else:
        # Jalankan secara otomatis terhadap seluruh sampel pisang di dataset/sample_images
        sample_dir = "dataset/sample_images"
        files = sorted([os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        if not files:
            print("Silakan tentukan file citra uji: python 03_ripeness_classifier.py <path_citra>")
            return
        print(f"Menjalankan klasifikasi pada {len(files)} sampel uji...")
        for f in files:
            classify_banana_ripeness(f)

if __name__ == "__main__":
    main()
