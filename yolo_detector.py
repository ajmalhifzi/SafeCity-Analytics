"""
================================================================================
KOD KURSUS   : DKA3223 ARTIFICIAL INTELLIGENCE FOR COMPUTER VISION
TUGASAN      : PRACTICAL TEST 2 (CLO 2)
SYARIKAT     : SAFECITY ANALYTICS - TRAFFIC MONITORING SYSTEM
FAIL         : yolo_detector.py
DESKRIPSI    : Melaksanakan pengesanan objek kenderaan lalu lintas (Object Detection)
               menggunakan model YOLOv8 (PyTorch/Ultralytics) berserta penalaan
               hiperparameter (conf=0.40, iou=0.50), melukis bounding boxes,
               dan menyimpan imej hasil sebagai result_traffic.jpg.
================================================================================
"""

import os
import sys
import cv2

# ==============================================================================
# KRITERIA 5.2 & 6.2: IMPORT PUSTAKA YOLO DARI ULTRALYTICS (BERASASKAN PYTORCH)
# ==============================================================================
try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"[RALAT] Gagal mengimport Ultralytics YOLO: {e}")
    sys.exit(1)


# ==============================================================================
# KRITERIA 5.8 & 6.7: EMPAT (4) KOMEN PERBANDINGAN TEKNIKAL (YOLO vs ANN)
# ==============================================================================

# KOMEN 1: PEMELIHARAAN STRUKTUR SPATIAL 2D (SPATIAL TOPOLOGY VS 1D FLATTENING)
# Seni bina ANN konvensional memerlukan matriks imej dua dimensi diratakan (flattened)
# menjadi vektor satu dimensi (1D), menyebabkan semua maklumat hubungan spatial dan
# kedudukan relatif piksel musnah. Sebaliknya, seni bina YOLO mengekalkan struktur
# tensor spatial (C x H x W) menggunakan lapisan konvolusi, membolehkan model
# mengekstrak ciri-ciri visual kenderaan serta memahami kedudukannya di atas jalan raya.

# KOMEN 2: REGRESI PENYETEMPATAN SATU LANGKAH (SINGLE-SHOT UNIFIED DETECTION)
# ANN tradisional hanya berfungsi untuk klasifikasi imej global (menentukan apa imej itu)
# dan tidak mampu mencari lokasi objek secara langsung melainkan digabungkan dengan teknik
# tetingkap gelongsor (sliding window) yang amat perlahan. YOLO memproses keseluruhan imej
# dalam SATU laluan ke hadapan (single forward pass), membahagikan imej kepada sel grid
# dan secara serentak meramal koordinat bounding box (x, y, w, h) serta kebarangkalian kelas.

# KOMEN 3: KETAKVARIANAN TRANSLASI DAN SKALA CIRI (TRANSLATION & SCALE INVARIANCE)
# Melalui perkongsian pemberat (weight sharing) penapis konvolusi (convolutional filters)
# dan Feature Pyramid Network (FPN), YOLO berupaya mengesan kenderaan tanpa mengira
# kedudukannya dalam bingkai kamera atau saiznya (jauh atau dekat). Bagi ANN berpautan penuh
# (Fully Connected), ia amat sensitif terhadap pergeseran piksel dan memerlukan latihan
# semula bagi setiap variasi posisi objek.

# KOMEN 4: KECEKAPAN PENGKOMPUTERAN UNTUK PERANTI EDGE (REAL-TIME EDGE DEPLOYMENT)
# Model YOLOv8n (nano) mempunyai bilangan parameter yang sangat padat (~3.2 juta parameter)
# dan direka khas untuk pemprosesan masa nyata (real-time FPS) dengan kependaman rendah
# (low latency) pada peranti Edge seperti NVIDIA Jetson. Sebaliknya, ANN yang menerima input
# imej beresolusi tinggi menghasilkan puluhan juta parameter pemberat linear, menjadikannya
# terlalu berat, lambat, dan tidak praktikal untuk sistem pemantauan trafik masa nyata.


def run_traffic_detection():
    print("=" * 75)
    print("   SAFECITY ANALYTICS - SISTEM PENGESANAN KENDERAAN LALU LINTAS")
    print("   PRAKTIKAL TEST 2: DKA3223 ARTIFICIAL INTELLIGENCE FOR COMPUTER VISION")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # KRITERIA 5.3 & 6.3: MEMUATKAN MODEL YOLOV8 ASAS (yolov8n.pt)
    # --------------------------------------------------------------------------
    model_path = "yolov8n.pt"
    print(f"\n[LANGKAH 1] Memuatkan model YOLOv8 asas: '{model_path}'...")
    try:
        model = YOLO(model_path)
        print("  -> Model YOLOv8n berjaya dimuatkan ke dalam persekitaran PyTorch.")
    except Exception as err:
        print(f"[RALAT] Gagal memuatkan model {model_path}: {err}")
        return

    # --------------------------------------------------------------------------
    # KRITERIA 5.4 & 6.3: MEMBACA SATU (1) KEPING FAIL IMEJ SUMBER UJIAN
    # --------------------------------------------------------------------------
    # Membenarkan laluan imej dinamik melalui baris arahan, atau lalai kepada 'traffic_test.jpg'
    image_input_path = sys.argv[1] if len(sys.argv) > 1 else "traffic_test.jpg"
    print(f"\n[LANGKAH 2] Memeriksa dan membaca fail imej sumber: '{image_input_path}'...")
    if not os.path.exists(image_input_path):
        print(f"[RALAT] Fail imej '{image_input_path}' tidak dijumpai di direktori semasa.")
        return

    # Membaca imej menggunakan OpenCV untuk mengesahkan integriti fail
    source_img = cv2.imread(image_input_path)
    if source_img is None:
        print(f"[RALAT] Gagal membaca data imej '{image_input_path}'.")
        return

    h, w, c = source_img.shape
    print(f"  -> Imej sumber berjaya dibaca. Dimensi: {w}x{h} piksel, Saluran: {c} (RGB/BGR).")

    # --------------------------------------------------------------------------
    # KRITERIA 5.5 & 6.4: PENALAAN HIPERPARAMETER (conf=0.40, iou=0.50)
    # --------------------------------------------------------------------------
    conf_threshold = 0.40  # Confidence Threshold (40%)
    iou_threshold = 0.50   # Intersection over Union / NMS Threshold (50%)

    print(f"\n[LANGKAH 3] Melaksanakan inferens dengan penalaan hiperparameter:")
    print(f"  * Confidence Threshold (conf) : {conf_threshold} (40%)")
    print(f"  * Intersection over Union (iou): {iou_threshold} (50%)")

    # Menjalankan inferens model YOLO
    results = model.predict(
        source=image_input_path,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )

    # --------------------------------------------------------------------------
    # KRITERIA 5.6 & 6.5: MELUKIS KOTAK SEMPADAN (BOUNDING BOXES), KELAS & SKOR
    # --------------------------------------------------------------------------
    print("\n[LANGKAH 4] Memproses hasil pengesanan objek...")
    result = results[0]
    detections = result.boxes

    total_detected = len(detections)
    print(f"  -> Jumlah objek kenderaan dikesan (conf >= {conf_threshold}): {total_detected}")

    if total_detected == 0:
        print("[AMARAN] Tiada objek dikesan melepasi ambang keyakinan.")
    else:
        print("\n" + "-" * 75)
        print(f"{'BIL':<5} | {'KELAS':<15} | {'KEYAKINAN (CONF)':<18} | {'BOUNDING BOX (x1, y1, x2, y2)'}")
        print("-" * 75)
        for idx, box in enumerate(detections, 1):
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            confidence_score = float(box.conf[0].item())
            coords = [round(float(c), 1) for c in box.xyxy[0].tolist()]

            print(f"{idx:<5} | {class_name:<15} | {confidence_score * 100:>6.2f}% ({confidence_score:.2f})  | {coords}")
        print("-" * 75)

    # Menjana visualisasi imej bertanda (bounding boxes, class label, confidence)
    annotated_image = result.plot()

    # --------------------------------------------------------------------------
    # KRITERIA 5.7 & 6.6: MENYIMPAN IMEJ HASIL DENGAN NAMA result_traffic.jpg
    # --------------------------------------------------------------------------
    output_filename = "result_traffic.jpg"
    print(f"\n[LANGKAH 5] Menyimpan imej hasil pengesanan...")
    save_success = cv2.imwrite(output_filename, annotated_image)

    if save_success and os.path.exists(output_filename):
        file_size_kb = os.path.getsize(output_filename) / 1024
        print(f"  -> Imej hasil berjaya disimpan: '{output_filename}' ({file_size_kb:.1f} KB)")
        print(f"  -> Lokasi direktori fail        : {os.path.abspath(output_filename)}")
    else:
        print(f"[RALAT] Gagal menyimpan imej output '{output_filename}'.")
        return

    # --------------------------------------------------------------------------
    # KRITERIA 6.8: PENGESAHAN EKSEKUSI BERSIH TANPA RALAT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("   [PENGESAHAN] OPERASI INFERENS SAFECITY ANALYTICS SELESAI DENGAN JAYANYA.")
    print("   STATUS LOG: BERSIH TANPA RALAT (ERROR-FREE)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_traffic_detection()
