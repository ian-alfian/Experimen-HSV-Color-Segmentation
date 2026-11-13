"""
02_BANANA_SEGMENTATION.PY
-------------------------
Pipeline Lengkap Segmentasi Citra Pisang Berbasis Ruang Warna HSV:
1. Pra-pemrosesan Denoising (Gaussian Filter)
2. Konversi BGR ke HSV
3. Dekomposisi 3 Kanal (Hue, Saturation, Value)
4. Pembentukan Mask Biner Berdasarkan Ambang Batas
5. Pemurnian Morfologi (Opening, Closing, dan Largest Connected Component)
6. Ekstraksi Objek Foreground (Bitwise AND)
7. Deteksi Kontur & Lokalisasi Spasial Bounding Box
8. Ekspor Visualisasi Multi-Panel ke Folder 'results/'
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Atur direktori konfigurasi Matplotlib ke /tmp agar tidak ada error permission
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

def segment_banana(image_path, target_stage="yellow", save_dir="results/segmentation"):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 1. Baca Citra
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"Error: Tidak dapat membaca citra dari {image_path}")
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2. Pra-pemrosesan: Gaussian Blur
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)

    # 3. Konversi Ruang Warna ke HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 4. Penentuan Rentang Thresholding Berdasarkan Target
    # Referensi rentang:
    # - Kuning (Matang): H ~ 19-32, S ~ 70-255, V ~ 70-255
    # - Hijau (Mentah): H ~ 38-85, S ~ 50-255, V ~ 50-255
    # - Cokelat (Terlalu Matang): H ~ 5-19, S ~ 50-255, V ~ 30-180
    # - Seluruh Tubuh Pisang: H ~ 5-85, S ~ 40-255, V ~ 30-255
    if target_stage in ["brown", "overripe"]:
        lower = np.array([5, 40, 30])
        upper = np.array([19, 255, 180])
        label = "Pisang Terlalu Matang (Cokelat)"
    elif target_stage == "yellow":
        lower = np.array([19, 70, 70])
        upper = np.array([34, 255, 255])
        label = "Pisang Kuning (Matang)"
    elif target_stage == "green":
        lower = np.array([35, 50, 50])
        upper = np.array([85, 255, 255])
        label = "Pisang Hijau (Mentah)"
    else:  # 'all' - seluruh tubuh buah pisang (termasuk hijau, kuning, dan cokelat)
        lower = np.array([5, 40, 30])
        upper = np.array([85, 255, 255])
        label = "Tubuh Pisang Total"

    # Mask mentah
    raw_mask = cv2.inRange(hsv, lower, upper)

    # 5. Operasi Morfologi Citra (Post-processing)
    # a. Opening: Menghilangkan salt-and-pepper noise pada latar belakang
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_open = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # b. Closing: Menutup lubang kilau cahaya (specular reflection) di dalam tubuh buah
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # c. Seleksi Largest Connected Component (mengabaikan bercak latar belakang terisolasi)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    if num_labels > 1:
        # Cari komponen dengan area terbesar (kecuali background label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_final = np.zeros_like(mask_clean)
        mask_final[labels == largest_label] = 255
    else:
        mask_final = mask_clean

    # 6. Ekstraksi Objek Foreground via Bitwise AND
    segmented_bgr = cv2.bitwise_and(bgr, bgr, mask=mask_final)
    segmented_rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)

    # 7. Deteksi Kontur & Bounding Box
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_img = rgb.copy()
    area = 0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(bbox_img, f"{label} ({int(area)} px)", (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    active_pixels = cv2.countNonZero(mask_final)
    total_pixels = bgr.shape[0] * bgr.shape[1]
    pct = (active_pixels / total_pixels) * 100.0

    print(f"\n[HASIL SEGMENTASI: {base_name}]")
    print(f"Target Label    : {label}")
    print(f"Lower Bound HSV : {lower}")
    print(f"Upper Bound HSV : {upper}")
    print(f"Piksel Terdeteksi: {active_pixels:,} px ({pct:.2f}% area citra)")
    print(f"Luas Kontur     : {area:,} px")

    # 8. Visualisasi Multi-Panel Terpadu
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("(a) Citra Asli (RGB)")
    axes[0, 0].axis('off')

    # Visualisasi Kanal HSV secara terpisah
    axes[0, 1].imshow(H, cmap='hsv')
    axes[0, 1].set_title("(b) Kanal Hue (H: 0-179)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(S, cmap='plasma')
    axes[0, 2].set_title("(c) Kanal Saturation (S: 0-255)")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(mask_final, cmap='gray')
    axes[1, 0].set_title(f"(d) Binary Mask ({active_pixels:,} px)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(segmented_rgb)
    axes[1, 1].set_title("(e) Foreground Tersegmentasi (Bitwise AND)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(bbox_img)
    axes[1, 2].set_title("(f) Lokalisasi Bounding Box")
    axes[1, 2].axis('off')

    plt.tight_layout()
    out_figure = os.path.join(save_dir, f"{base_name}_segmentation_panel.png")
    plt.savefig(out_figure, dpi=200)
    plt.close()
    print(f"Visualisasi berhasil disimpan: {out_figure}")

    # Simpan mask dan citra hasil segmentasi individual
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_mask.png"), mask_final)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_segmented.png", ), segmented_bgr)

    return {
        'active_pixels': active_pixels,
        'percentage': pct,
        'contour_area': area,
        'out_figure': out_figure
    }

def main():
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Default: uji pada sampel pisang matang
        target_path = "dataset/sample_images/pisang_matang.jpg"

    target_mode = sys.argv[2] if len(sys.argv) > 2 else "yellow"
    segment_banana(target_path, target_stage=target_mode)

if __name__ == "__main__":
    main()
