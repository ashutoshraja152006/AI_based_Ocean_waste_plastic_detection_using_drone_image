from pathlib import Path
import torch
from ultralytics import YOLO

DATA_YAML = Path("configs/data.yaml")
PROJECT_DIR = Path("outputs/models")
RUN_NAME = "ocean_plastic_yolov8n"


def main():
    print("=" * 65)
    print("STAGE 8-11: YOLOv8 MODEL INITIALIZATION & GPU TRAINING")
    print("=" * 65)

    # 1. Verify GPU availability before starting
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[GPU READY] Training on Device 0: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        device = "cpu"
        print("[WARNING] CUDA GPU not detected. Falling back to CPU.")

    # 2. Load Pre-trained YOLOv8n (Transfer Learning)
    print("\nLoading pre-trained YOLOv8n model ('yolov8n.pt')...")
    model = YOLO("yolov8n.pt")

    # 3. Start reproducible training with drone-specific augmentations
    print("\nStarting training... (Monitor Box Loss, Class Loss, and mAP50 per epoch)")
    results = model.train(
        data=str(DATA_YAML.resolve()),
        epochs=15,
        imgsz=640,
        batch=16,
        device=device,
        workers=2,           # Safe worker count for Windows
        seed=42,             # Reproducible training
        patience=5,          # Early stopping if no improvement for 5 epochs
        lr0=0.01,            # Initial learning rate
        lrf=0.01,            # Final learning rate factor
        flipud=0.5,          # Drone top-down vertical flip augmentation
        fliplr=0.5,          # Drone top-down horizontal flip augmentation
        mosaic=1.0,          # 4-image mosaic augmentation for small plastic debris
        project=str(PROJECT_DIR.resolve()),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
    )

    best_weights = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    last_weights = PROJECT_DIR / RUN_NAME / "weights" / "last.pt"

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE!")
    print("=" * 65)
    print(f"Best Model Weights Saved : {best_weights}")
    print(f"Last Model Weights Saved : {last_weights}")
    print(f"Training Charts & Logs   : {PROJECT_DIR / RUN_NAME}")
    print("=" * 65)


if __name__ == "__main__":
    main()