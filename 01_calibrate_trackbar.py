"""
01_CALIBRATE_TRACKBAR.PY
------------------------
Skrip kalibrasi interaktif untuk mencari rentang ambang batas (Threshold Bounds)
Hue, Saturation, dan Value (HSV) secara real-time pada citra pisang.
Rujukan: Putri et al. (2026) dan Akbar et al. (2025).

Cara Menjalankan:
    python 01_calibrate_trackbar.py [path_citra]
    Contoh: python 01_calibrate_trackbar.py dataset/sample_images/pisang_matang.jpg

Kontrol Keyboard:
    - Geser Slider: Menyesuaikan parameter H, S, V secara langsung
    - 's': Cetak rentang nilai saat ini ke terminal & simpan dokumentasi mask
    - 'q' atau ESC: Keluar dari program
"""

import cv2
import numpy as np
import sys
import os

def nothing(x):
    pass

def main():
    # 1. Penentuan citra input
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Cari file di dataset atau gunakan contoh sintetis
        default_dir = "dataset/sample_images"
        files = [os.path.join(default_dir, f) for f in os.listdir(default_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(default_dir) else []
        if files:
            img_path = files[0]
        else:
            print("Peringatan: Tidak ada argumen citra yang diberikan.")
            print("Penggunaan: python 01_calibrate_trackbar.py <path_ke_foto_pisang>")
            return

    if not os.path.exists(img_path):
        print(f"Error: Berkas citra '{img_path}' tidak ditemukan.")
        return

    # 2. Baca dan resize citra agar nyaman ditampilkan di layar
    src = cv2.imread(img_path)
    if src is None:
        print(f"Error: Gagal memuat citra dari '{img_path}'.")
        return

    h, w = src.shape[:2]
    max_dim = 650
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    img = cv2.resize(src, (int(w * scale), int(h * scale)))

    # Pra-pemrosesan: Gaussian Blur untuk mereduksi noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 3. Buat Jendela GUI Trackbar
    win_name = "HSV Banana Calibration Tool"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 900, 600)

    # Nilai default awal (perkiraan awal untuk warna kuning pisang)
    cv2.createTrackbar("H Min", win_name, 20, 179, nothing)
    cv2.createTrackbar("H Max", win_name, 35, 179, nothing)
    cv2.createTrackbar("S Min", win_name, 80, 255, nothing)
    cv2.createTrackbar("S Max", win_name, 255, 255, nothing)
    cv2.createTrackbar("V Min", win_name, 80, 255, nothing)
    cv2.createTrackbar("V Max", win_name, 255, 255, nothing)

    print("=" * 65)
    print("   KALIBRASI HSV INTERAKTIF - DETEKSI PISANG")
    print(f"   Citra Aktif: {img_path}")
    print("   Tekan 's' untuk mencetak & menyimpan parameter")
    print("   Tekan 'q' atau 'ESC' untuk keluar")
    print("=" * 65)

    save_count = 0
    os.makedirs("results/calibration", exist_ok=True)

    while True:
        # Ambil posisi slider
        h_min = cv2.getTrackbarPos("H Min", win_name)
        h_max = cv2.getTrackbarPos("H Max", win_name)
        s_min = cv2.getTrackbarPos("S Min", win_name)
        s_max = cv2.getTrackbarPos("S Max", win_name)
        v_min = cv2.getTrackbarPos("V Min", win_name)
        v_max = cv2.getTrackbarPos("V Max", win_name)

        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # Operasi Masking & Bitwise AND
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Pembersihan Morfologi (Opening untuk hapus bintik noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        result = cv2.bitwise_and(img, img, mask=mask_clean)

        # Hitung piksel aktif
        active_px = cv2.countNonZero(mask_clean)
        total_px = img.shape[0] * img.shape[1]
        pct = (active_px / total_px) * 100.0

        # Tampilkan status di panel citra
        display_img = img.copy()
        cv2.putText(display_img, f"Aktif: {active_px:,} px ({pct:.1f}%)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Tumpuk horizontal: Citra Asli (kiri), Mask Biner (tengah), Hasil Segmen (kanan)
        mask_3ch = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        preview = np.hstack((display_img, mask_3ch, result))

        cv2.imshow(win_name, preview)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            save_count += 1
            print(f"\n--- [PARAMETER TERSIMPAN #{save_count}] ---")
            print(f"Lower Bound : np.array([{h_min}, {s_min}, {v_min}])")
            print(f"Upper Bound : np.array([{h_max}, {s_max}, {v_max}])")
            print(f"Piksel Aktif: {active_px:,} ({pct:.2f}% dari total citra)")

            out_prefix = f"results/calibration/kalibrasi_{save_count}"
            cv2.imwrite(f"{out_prefix}_mask.png", mask_clean)
            cv2.imwrite(f"{out_prefix}_segmented.png", result)
            print(f"Dokumentasi disimpan ke {out_prefix}_*.png\n")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
