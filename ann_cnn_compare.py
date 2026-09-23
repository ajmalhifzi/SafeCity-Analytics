"""
================================================================================
KOD KURSUS   : DKA3223 ARTIFICIAL INTELLIGENCE FOR COMPUTER VISION
TUGASAN      : PRACTICAL TEST 2 (CLO 2)
SYARIKAT     : SAFECITY ANALYTICS - TRAFFIC MONITORING SYSTEM
FAIL         : ann_cnn_compare.py
DESKRIPSI    : Memuatkan model pra-terlatih mudah (CNN ResNet-18) dari torchvision.models,
               memaparkan ringkasan seni bina (architecture summary), dan menganalisis
               perbandingan struktur asas ANN vs CNN untuk pemprosesan visual.
================================================================================
"""

import torch
import torch.nn as nn
import torchvision.models as models


def display_ann_structure():
    """
    Mendefinisikan dan memaparkan struktur model Asas ANN (Multilayer Perceptron / MLP).
    ANN memerlukan input imej 2D diratakan (flatten) kepada vektor 1D.
    """
    print("\n" + "=" * 70)
    print("1. CONTOH STRUKTUR MODEL ASAS ANN (ARTIFICIAL NEURAL NETWORK / MLP)")
    print("=" * 70)

    # Struktur ANN asas dengan lapisan Linear (Fully Connected)
    ann_model = nn.Sequential(
        nn.Flatten(),                     # Meratakan matriks imej (cth: 3x224x224 -> 150,528)
        nn.Linear(3 * 224 * 224, 512),    # Lapisan Tersembunyi 1 (Fully Connected)
        nn.ReLU(),
        nn.Linear(512, 128),              # Lapisan Tersembunyi 2
        nn.ReLU(),
        nn.Linear(128, 3)                 # Lapisan Output (cth: Car, Truck, Motorcycle)
    )

    print(ann_model)
    ann_params = sum(p.numel() for p in ann_model.parameters())
    print(f"\n[INFO ANN] Jumlah Parameter Asas ANN : {ann_params:,}")
    print("[KEKANGAN ANN] Memerlukan 1D vector; kehilangan korelasi spatial 2D piksel.")


def display_cnn_architecture():
    """
    Memuatkan model CNN pra-terlatih (ResNet-18) dari torchvision.models
    dan mencetak ringkasan seni bina (architecture summary) model tersebut.
    """
    print("\n" + "=" * 70)
    print("2. MEMUATKAN MODEL PRE-TRAINED CNN (torchvision.models.resnet18)")
    print("=" * 70)

    # Memuatkan model pra-terlatih ResNet-18 dengan pemberat ImageNet
    weights = models.ResNet18_Weights.DEFAULT
    cnn_model = models.resnet18(weights=weights)

    # Menetapkan model ke mod penilaian (evaluation mode)
    cnn_model.eval()

    print("\n>>> SENI BINA MODEL RESNET-18 (ARCHITECTURE SUMMARY) <<<")
    print(cnn_model)

    # Pengiraan parameter model
    total_params = sum(p.numel() for p in cnn_model.parameters())
    trainable_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)

    print("\n" + "-" * 70)
    print("RINGKASAN PARAMETER MODEL CNN (RESNET-18):")
    print("-" * 70)
    print(f"Total Parameters      : {total_params:,}")
    print(f"Trainable Parameters  : {trainable_params:,}")
    print(f"Model Backbone Type   : Deep Residual Convolutional Neural Network (CNN)")
    print(f"Pre-trained Weights   : ImageNet (ResNet18_Weights.DEFAULT)")
    print("-" * 70)

    return cnn_model


def display_architectural_comparison():
    """
    Memaparkan analisis perbandingan seni bina antara ANN, CNN, dan YOLO
    untuk sistem pemantauan lalu lintas SafeCity Analytics.
    """
    print("\n" + "=" * 70)
    print("3. JADUAL PERBANDINGAN SENI BINA TEKNIKAL: ANN vs CNN vs YOLO")
    print("=" * 70)
    comparison_text = """
+--------------------+-------------------------+-------------------------+-------------------------+
| Ciri / Metrik      | ANN (Asas)              | CNN (ResNet / Asas)     | YOLO (Object Detection) |
+--------------------+-------------------------+-------------------------+-------------------------+
| Format Input       | Vektor 1D (Diratakan)   | Tensor 2D/3D (C, H, W)  | Tensor 2D/3D (Grid)     |
| Sensitiviti Lokasi | Tiada (Hilang spatial)  | Tinggi (Feature maps)   | Sangat Tinggi (BBoxes)  |
| Keupayaan Tugas    | Klasifikasi Global      | Klasifikasi / Fitur     | Klasifikasi + Lokalisasi|
| Sifat Lapisan      | Fully Connected         | Konvolusi + Pooling     | Konvolusi Anchor/Anchorless
| Kesesuaian Edge    | Tidak efisien (Weights) | Sederhana               | Sangat Pantas (Real-Time)|
+--------------------+-------------------------+-------------------------+-------------------------+
"""
    print(comparison_text)
    print("[KESIMPULAN SAFECITY ANALYTICS]")
    print("Model konvolusi (CNN/YOLO) mengekalkan matriks spatial yang penting untuk")
    print("mengesan lokasi kenderaan secara tepat di atas jalan raya berbanding ANN biasa.\n")


def main():
    print("=" * 70)
    print("   SAFECITY ANALYTICS - PEMANTAUAN LALU LINTAS KECERDASAN BUATAN")
    print("   PRAKTIKAL TEST 2: DKA3223 ARTIFICIAL INTELLIGENCE FOR COMPUTER VISION")
    print("=" * 70)

    # 1. Paparan struktur asas ANN
    display_ann_structure()

    # 2. Pemuatan dan ringkasan model pra-terlatih CNN
    display_cnn_architecture()

    # 3. Analisis teknikal perbandingan
    display_architectural_comparison()

    print("[STATUS] Skrip ann_cnn_compare.py selesai dilaksanakan dengan jayanya tanpa ralat.")


if __name__ == "__main__":
    main()
