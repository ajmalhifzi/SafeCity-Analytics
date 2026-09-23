"""
================================================================================
KOD KURSUS   : DKA3223 ARTIFICIAL INTELLIGENCE FOR COMPUTER VISION
SYARIKAT     : SAFECITY ANALYTICS - TRAFFIC MONITORING SYSTEM
FAIL         : app_gui.py
DESKRIPSI    : Antara Muka Pengguna Grafik (GUI Desktop App) interaktif bagi
               Sistem Pengesanan Kenderaan SafeCity Analytics berasaskan YOLOv8.
================================================================================
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO


class SafeCityTrafficApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SafeCity Analytics - AI Traffic Monitoring Dashboard")
        self.root.geometry("1100x720")
        self.root.minsize(950, 650)
        self.root.configure(bg="#f0f4f8")

        # Keadaan data
        self.model = None
        self.current_image_path = "traffic_test.jpg" if os.path.exists("traffic_test.jpg") else ""
        self.annotated_image_cv = None
        self.display_photo = None

        # Muatkan model YOLOv8 asas
        self.load_yolo_model()

        # Bina Antara Muka (UI)
        self.build_ui()

        # Paparkan imej awal jika wujud
        if self.current_image_path:
            self.display_image(self.current_image_path)

    def load_yolo_model(self):
        try:
            model_path = "yolov8n.pt"
            self.model = YOLO(model_path)
        except Exception as e:
            messagebox.showerror("Ralat Pemuatan Model", f"Gagal memuatkan YOLOv8: {e}")

    def build_ui(self):
        # Header Utama
        header_frame = tk.Frame(self.root, bg="#1e3a8a", height=70)
        header_frame.pack(fill="x", side="top")

        title_label = tk.Label(
            header_frame,
            text="SAFECITY ANALYTICS - SISTEM PEMANTAUAN LALU LINTAS PINTAR",
            font=("Segoe UI", 16, "bold"),
            fg="#ffffff",
            bg="#1e3a8a"
        )
        title_label.pack(pady=(12, 2))

        subtitle_label = tk.Label(
            header_frame,
            text="Practical Test 2 (DKA3223) | Real-Time Edge Object Detection Dashboard (YOLOv8)",
            font=("Segoe UI", 10),
            fg="#93c5fd",
            bg="#1e3a8a"
        )
        subtitle_label.pack(pady=(0, 10))

        # Kandungan Utama (Kiri: Kawalan, Kanan: Paparan)
        main_content = tk.Frame(self.root, bg="#f0f4f8")
        main_content.pack(fill="both", expand=True, padx=20, pady=15)

        # Panel Kiri (Kawalan)
        control_panel = tk.LabelFrame(
            main_content,
            text=" Panel Kawalan & Hiperparameter ",
            font=("Segoe UI", 11, "bold"),
            fg="#1e3a8a",
            bg="#ffffff",
            padx=15,
            pady=15
        )
        control_panel.pack(side="left", fill="y", padx=(0, 15))

        # Butang Pilih Imej
        btn_browse = tk.Button(
            control_panel,
            text="📁 Pilih Imej Lalu Lintas",
            font=("Segoe UI", 10, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            padx=10,
            pady=8,
            cursor="hand2",
            command=self.browse_image
        )
        btn_browse.pack(fill="x", pady=(0, 12))

        # Maklumat Fail Imej
        self.lbl_file_info = tk.Label(
            control_panel,
            text="Imej Semasa:\ntraffic_test.jpg" if self.current_image_path else "Tiada imej dipilih",
            font=("Segoe UI", 9),
            fg="#4b5563",
            bg="#ffffff",
            wraplength=220,
            justify="left"
        )
        self.lbl_file_info.pack(fill="x", pady=(0, 15))

        # Garisan pemisah
        ttk.Separator(control_panel, orient="horizontal").pack(fill="x", pady=10)

        # Slider 1: Confidence Threshold (conf)
        lbl_conf = tk.Label(
            control_panel,
            text="Confidence Threshold (conf):",
            font=("Segoe UI", 10, "bold"),
            fg="#374151",
            bg="#ffffff"
        )
        lbl_conf.pack(anchor="w")

        self.conf_var = tk.DoubleVar(value=0.40)
        self.lbl_conf_val = tk.Label(control_panel, text="0.40 (40%)", font=("Segoe UI", 9, "bold"), fg="#2563eb", bg="#ffffff")
        self.lbl_conf_val.pack(anchor="e")

        scale_conf = tk.Scale(
            control_panel,
            from_=0.10,
            to=1.00,
            resolution=0.05,
            orient="horizontal",
            variable=self.conf_var,
            showvalue=False,
            bg="#ffffff",
            highlightthickness=0,
            troughcolor="#e2e8f0",
            activebackground="#2563eb",
            command=lambda v: self.lbl_conf_val.config(text=f"{float(v):.2f} ({int(float(v)*100)}%)")
        )
        scale_conf.pack(fill="x", pady=(0, 15))

        # Slider 2: IOU Threshold (iou)
        lbl_iou = tk.Label(
            control_panel,
            text="Intersection over Union (iou):",
            font=("Segoe UI", 10, "bold"),
            fg="#374151",
            bg="#ffffff"
        )
        lbl_iou.pack(anchor="w")

        self.iou_var = tk.DoubleVar(value=0.50)
        self.lbl_iou_val = tk.Label(control_panel, text="0.50 (50%)", font=("Segoe UI", 9, "bold"), fg="#2563eb", bg="#ffffff")
        self.lbl_iou_val.pack(anchor="e")

        scale_iou = tk.Scale(
            control_panel,
            from_=0.10,
            to=1.00,
            resolution=0.05,
            orient="horizontal",
            variable=self.iou_var,
            showvalue=False,
            bg="#ffffff",
            highlightthickness=0,
            troughcolor="#e2e8f0",
            activebackground="#2563eb",
            command=lambda v: self.lbl_iou_val.config(text=f"{float(v):.2f} ({int(float(v)*100)}%)")
        )
        scale_iou.pack(fill="x", pady=(0, 15))

        # Butang Jalankan Inferens
        btn_detect = tk.Button(
            control_panel,
            text="🚀 Jalankan Pengesanan (Run)",
            font=("Segoe UI", 11, "bold"),
            bg="#16a34a",
            fg="#ffffff",
            activebackground="#15803d",
            activeforeground="#ffffff",
            relief="flat",
            padx=10,
            pady=10,
            cursor="hand2",
            command=self.run_detection
        )
        btn_detect.pack(fill="x", pady=(10, 10))

        # Butang Buka Laporan PDF
        btn_report = tk.Button(
            control_panel,
            text="📄 Buka Laporan Rasmi (PDF)",
            font=("Segoe UI", 9),
            bg="#475569",
            fg="#ffffff",
            relief="flat",
            pady=6,
            cursor="hand2",
            command=self.open_pdf_report
        )
        btn_report.pack(fill="x", pady=(0, 10))

        # Panel Kanan (Paparan Imej & Keputusan)
        display_frame = tk.Frame(main_content, bg="#f0f4f8")
        display_frame.pack(side="right", fill="both", expand=True)

        # Ruang Kanvas Imej
        image_panel = tk.LabelFrame(
            display_frame,
            text=" Paparan Visualisasi Hasil Pengesanan Kenderaan ",
            font=("Segoe UI", 11, "bold"),
            fg="#1e3a8a",
            bg="#ffffff",
            padx=10,
            pady=10
        )
        image_panel.pack(fill="both", expand=True, pady=(0, 10))

        self.canvas_label = tk.Label(image_panel, bg="#111827", text="Sila tekan 'Jalankan Pengesanan' untuk mula.", fg="#9ca3af")
        self.canvas_label.pack(fill="both", expand=True)

        # Panel Statistik Log di Bawah
        stats_frame = tk.LabelFrame(
            display_frame,
            text=" Ringkasan Pengesanan Masa Nyata ",
            font=("Segoe UI", 10, "bold"),
            fg="#1e3a8a",
            bg="#ffffff",
            padx=10,
            pady=8
        )
        stats_frame.pack(fill="x")

        self.lbl_stats = tk.Label(
            stats_frame,
            text="Model: YOLOv8n (Edge) | Status: Bersedia | Jumlah Objek: 0",
            font=("Segoe UI", 9.5),
            fg="#1f2937",
            bg="#ffffff",
            anchor="w",
            justify="left"
        )
        self.lbl_stats.pack(fill="x")

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Imej Ujian Kenderaan",
            filetypes=[("Imej JPEG/PNG", "*.jpg *.jpeg *.png"), ("Semua Fail", "*.*")]
        )
        if file_path:
            self.current_image_path = file_path
            self.lbl_file_info.config(text=f"Imej Semasa:\n{os.path.basename(file_path)}")
            self.display_image(file_path)

    def display_image(self, img_path_or_array):
        try:
            if isinstance(img_path_or_array, str):
                pil_img = Image.open(img_path_or_array)
            else:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb = cv2.cvtColor(img_path_or_array, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

            # Resize dengan nisbah aspek yang tepat
            max_w, max_h = 750, 430
            pil_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            self.display_photo = ImageTk.PhotoImage(pil_img)
            self.canvas_label.config(image=self.display_photo, text="")
        except Exception as e:
            self.canvas_label.config(text=f"Ralat memaparkan imej: {e}", image="")

    def run_detection(self):
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            messagebox.showwarning("Peringatan", "Sila pilih fail imej ujian terlebih dahulu.")
            return

        conf = self.conf_var.get()
        iou = self.iou_var.get()

        try:
            results = self.model.predict(
                source=self.current_image_path,
                conf=conf,
                iou=iou,
                verbose=False
            )
            res = results[0]
            boxes = res.boxes
            total = len(boxes)

            # Lukis hasil kotak sempadan
            self.annotated_image_cv = res.plot()

            # Simpan fail result_traffic.jpg secara automatik
            cv2.imwrite("result_traffic.jpg", self.annotated_image_cv)

            # Paparkan imej beranotasi pada kanvas
            self.display_image(self.annotated_image_cv)

            # Ringkasan teks
            detected_classes = [self.model.names[int(b.cls[0])] for b in boxes]
            class_counts = {c: detected_classes.count(c) for c in set(detected_classes)}
            summary_str = ", ".join([f"{k}: {v}" for k, v in class_counts.items()]) if class_counts else "Tiada objek"

            self.lbl_stats.config(
                text=f"Status: Berjaya Dikesan ✔ | Jumlah: {total} | Objek: {summary_str} | Disimpan ke: result_traffic.jpg",
                fg="#15803d"
            )

        except Exception as err:
            messagebox.showerror("Ralat Semasa Inferens", f"Ralat: {err}")

    def open_pdf_report(self):
        pdf_path = os.path.abspath(os.path.join("report", "Laporan_Amali_2_DKA3223.pdf"))
        if os.path.exists(pdf_path):
            try:
                os.startfile(pdf_path)
            except Exception as e:
                messagebox.showinfo("Laporan PDF", f"Lokasi PDF: {pdf_path}")
        else:
            messagebox.showwarning("Fail Tidak Dijumpai", f"Fail laporan tidak wujud di: {pdf_path}")


def main():
    root = tk.Tk()
    app = SafeCityTrafficApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
