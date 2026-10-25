# 🛸 AI-Based Ocean Waste Plastic Detection Using Drone Imagery

Autonomous Aerial Surveillance and Real-Time Marine Plastic Debris Localization using Fine-Tuned YOLOv8 Nano.

---

## 📌 Project Overview
Marine plastic pollution poses a severe environmental threat to global marine ecosystems. Traditional boat surveys and satellite imaging suffer from high operational costs and low spatial resolution. 

This project implements an end-to-end aerial deep learning pipeline using **unmanned aerial vehicles (UAVs / drones)** and **YOLOv8** to autonomously detect, count, and localize floating plastic waste (bottles, surgical masks, containers, and debris) in real-time.

---

## 🏆 Key Performance Benchmark (Unseen Test Set)

The model was trained on **5,519 aerial drone images** and evaluated on an independent, unseen test set of **263 drone images** at $640 \times 640$ resolution:

| Metric | Score | Description |
|---|---|---|
| **mAP@50** | **90.41%** (0.9041) | Mean Average Precision at 0.50 IoU |
| **Recall** | **90.57%** (0.9057) | Proportion of true plastic debris detected |
| **Precision** | **84.13%** (0.8413) | Accuracy of positive detections |
| **F1-Score** | **87.23%** (0.8723) | Harmonic balance between Precision & Recall |
| **mAP@50-95** | **69.06%** (0.6906) | Stringent COCO-standard metric across multiple IoUs |
| **Model Size** | **3.01 Million Parameters** | Optimized for lightweight edge drone inference |

---

## 🏗️ Architecture & Pipeline Flow

```text
[Raw Aerial Drone Footage / Roboflow Images]
                 │
                 ▼
[Data Engineering & CV Auto-Labeling Pipeline]
(Top-Hat Morphology, CIELAB Saliency, Floodfill Water Masking)
                 │
                 ▼
[Processed YOLO Dataset: 6,354 Matched Images & Labels]
                 │
                 ▼
[YOLOv8n Fine-Tuning with Drone-Specific Augmentations]
(Mosaic 1.0, Vertical Flip 0.5, Horizontal Flip 0.5, 15 Epochs)
                 │
                 ▼
[Trained Weights Checkpoint: best.pt]
                 │
        ┌────────┴──────────────────┐
        ▼                           ▼
[Interactive Web Dashboard]    [CLI Video & Batch Inference]
(Streamlit UI: app.py)         (predict.py & detect_video.py)
```

---

## 📁 Repository Structure

```text
AI_based_Ocean_waste_plastic_detection_using_drone_image/
│
├── configs/
│   └── data.yaml                   # YOLO dataset configuration
│
├── data/
│   ├── raw/                        # Original Roboflow export
│   └── processed/
│       └── yolo_dataset/           # Train (5,519), Valid (572), Test (263) splits
│
├── outputs/
│   ├── eda/                        # Exploratory data analysis charts & previews
│   ├── models/
│   │   └── ocean_plastic_yolov8n/  # Checkpoints (best.pt, last.pt) & training curves
│   ├── evaluation/                 # Metrics JSON & comparison plots
│   └── predictions/                # Inference outputs
│
├── src/
│   ├── check_environment.py        # System & PyTorch CUDA validator
│   ├── inspect_dataset.py          # Dataset auditor
│   ├── eda.py                      # Exploratory Data Analysis
│   ├── preview_autolabel.py        # Saliency-based bounding box generator
│   ├── generate_yolo_dataset.py    # Batch dataset generator
│   ├── verify_annotations.py       # Quality assurance checker
│   ├── train_model.py              # YOLOv8n fine-tuning script
│   ├── evaluate_model.py           # Evaluation benchmark script
│   ├── predict.py                  # Single & batch image inference CLI
│   └── detect_video.py             # Video surveillance tracking CLI
│
├── app.py                          # Interactive Streamlit Web Application
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## ⚡ Quick Start & Usage

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies (with CUDA 12.1 PyTorch)
pip install -r requirements.txt
```

### 2. Launch Streamlit Web Application
```powershell
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501` to:
- Test sample images or upload new drone aerial photos.
- Adjust confidence thresholds live.
- Process drone `.mp4` video files.
- Inspect model evaluation graphs and confusion matrices.

### 3. Run Command-Line Inference on Test Images
```powershell
python src/predict.py --num-samples 5 --conf 0.30
```
Annotated prediction images will be saved in `outputs/predictions/`.

### 4. Run Real-Time Drone Video Detection
```powershell
python src/detect_video.py --video "path_to_drone_footage.mp4" --conf 0.30
```

---

## 💡 Technical Innovations
1. **Automated Bounding-Box Generation**: Solved the challenge of unannotated raw classification data by developing a CIELAB color-opponent chromaticity filter combined with morphological Top-Hat filters to detect floating debris against dynamic ocean ripple backgrounds.
2. **Nadir Drone Data Augmentations**: Applied rotational and vertical flip invariants since aerial drones observe waste from a 90-degree nadir top-down viewpoint.
3. **Edge Optimization**: Chose YOLOv8 Nano (`yolov8n.pt`) with only 3.01 million parameters, enabling real-time on-board inference on embedded drone hardware (e.g., NVIDIA Jetson).
