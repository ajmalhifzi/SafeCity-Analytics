# SafeCity Analytics - Sistem Pemantauan Lalu Lintas Berasaskan AI

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8n-blue?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![Course](https://img.shields.io/badge/Course-DKA3223_AI_for_Computer_Vision-green)](https://moe.gov.my)

Repositori ini mengandungi kod sumber, konfigurasi inferens, dan laporan bagi **Ujian Amali 2 (DKA3223: Artificial Intelligence for Computer Vision)** yang dibangunkan untuk syarikat **SafeCity Analytics**.

---

## 📌 Pengenalan Projek

Syarikat **SafeCity Analytics** sedang membangunkan sistem pengesanan kenderaan masa nyata (*real-time traffic monitoring*) untuk kegunaan peranti Edge (seperti NVIDIA Jetson). Berbanding model ANN (*Artificial Neural Network*) konvensional yang tidak mempunyai pemeliharaan hubungan spatial piksel, projek ini mengaplikasikan model **YOLO (You Only Look Once)** berasaskan PyTorch bagi melaksanakan pengesanan objek secara serentak (*single-shot detection*).

---

## 📁 Struktur Direktori Repositori

```text
SafeCity-Analytics/
├── ann_cnn_compare.py          # Skrip 1: Perbandingan Seni Bina Model ANN & CNN
├── yolo_detector.py            # Skrip 2: Pengesanan Objek YOLOv8 & Penalaan Hiperparameter
├── traffic_test.jpg            # Imej input kenderaan lalu lintas (1280x852)
├── result_traffic.jpg          # Imej hasil inferens bertanda kotak sempadan (Bounding Boxes)
├── yolov8n.pt                  # Pemberat pra-terlatih model YOLOv8n (Nano)
├── README.md                   # Dokumentasi repositori GitHub
└── report/
    ├── Laporan_Amali_2_DKA3223.html # Templat laporan amali profesional (Print/PDF)
    └── Laporan_Amali_2_DKA3223.pdf  # Fail laporan rasmi untuk Google Classroom & cetakan
```

---

## ⚙️ Keperluan Sistem & Pemasangan

Pastikan Python 3.9+ telah dipasang. Pasang dependensi yang diperlukan:

```bash
pip install torch torchvision ultralytics opencv-python pillow
```

---

## 🚀 Arahan Menjalankan Skrip

### 1. Skrip Perbandingan Seni Bina (`ann_cnn_compare.py`)
Skrip ini memuatkan model pra-terlatih CNN (`ResNet-18`) dari `torchvision.models`, memaparkan struktur model, dan membandingkannya dengan asas ANN:
```bash
python ann_cnn_compare.py
```

### 2. Skrip Pengesanan YOLOv8 (`yolo_detector.py`)
Skrip ini memuatkan model `yolov8n.pt`, melaksanakan inferens pada imej lalu lintas dengan penalaan hiperparameter `conf=0.40` dan `iou=0.50`, melukis *bounding boxes*, serta menyimpan imej ke `result_traffic.jpg`:
```bash
# Menjalankan dengan imej lalai (traffic_test.jpg)
python yolo_detector.py

# Atau nyatakan imej pilihan secara dinamik:
python yolo_detector.py traffic_test.jpg
```

---

## 🎯 Penalaan Hiperparameter (*Hyperparameter Tuning*)

Inferens YOLO dikonfigurasikan dengan tepat mengikut spesifikasi penilaian:

| Hiperparameter | Nilai Tetapan | Fungsi & Justifikasi Teknikal |
| :--- | :---: | :--- |
| **Confidence Threshold (`conf`)** | `0.40` (40%) | Menapis pengesanan palsu (*false positives*) dan hanya membenarkan pengesanan objek dengan kebarangkalian keyakinan minimum 40%. |
| **Intersection over Union (`iou`)** | `0.50` (50%) | Ambang bagi algoritma *Non-Maximum Suppression* (NMS) untuk menghapuskan kotak pertindihan berganda bagi kenderaan yang sama. |

---

## 🧠 4 Komen Perbandingan Teknikal: Mengapa YOLO Lebih Baik Berbanding ANN?

1. **Pemeliharaan Struktur Spatial 2D (Spatial Topology vs 1D Flattening):**  
   ANN meratakan imej 2D menjadi vektor 1D, menghapuskan korelasi spatial kedudukan piksel. YOLO mengekalkan dimensi tensor menggunakan konvolusi 2D bagi mengekstrak ciri-ciri visual kenderaan.
2. **Regresi Penyetempatan Satu Langkah (Single-Shot Unified Detection):**  
   ANN hanya mampu mengklasifikasikan imej global tanpa lokasi tepat. YOLO membahagikan imej kepada sel grid dan meramal koordinat kotak sempadan `(x, y, w, h)` serta kelas secara serentak dalam *single forward pass*.
3. **Ketakvarianan Translasi & Skala Ciri (Translation & Scale Invariance):**  
   Penapis konvolusi YOLO berkongsi pemberat (*weight sharing*), membolehkan kenderaan dikesan walau di mana posisi atau skala saiznya di atas jalan raya. ANN memerlukan kedudukan piksel yang tegar.
4. **Kecekapan Pengkomputeran untuk Peranti Edge (Real-Time Edge Deployment):**  
   Model `yolov8n.pt` hanya mempunyai ~3.2 juta parameter dengan kadar inferens laju (>30 FPS), ideal untuk peranti Edge seperti NVIDIA Jetson. ANN untuk imej resolusi tinggi memerlukan puluhan juta parameter pemberat linear yang terlalu perlahan.

---

## 📊 Hasil Pengesanan Kenderaan

* **Imej Input:** `traffic_test.jpg`
* **Objek Dikesan:** `car` (Skor Keyakinan: `89.86%` / `0.90`)
* **Koordinat Kotak Sempadan (xyxy):** `[35.5, 372.8, 1190.0, 724.9]`
* **Fail Imej Output:** `result_traffic.jpg`

---

## 👨‍💻 Maklumat Calon & Kursus
* **Kursus:** DKA3223 Artificial Intelligence for Computer Vision
* **Program:** Diploma Teknologi Komputeran, Kolej Vokasional
* **Tahun / Semester:** 2026 / Semester 3
