import argparse
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path("outputs/models/ocean_plastic_yolov8n/weights/best.pt")
DEFAULT_SOURCE = Path("data/processed/yolo_dataset/test/images")
DEFAULT_OUTPUT = Path("outputs/predictions")


def run_prediction(
    weights_path: Path,
    source_path: Path,
    output_dir: Path,
    conf_thresh: float = 0.30,
    iou_thresh: float = 0.45,
    max_samples: int = None,
):
    """
    Runs plastic debris detection on a single image, video, or folder of images.
    Saves visual predictions to output_dir and prints summary statistics.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Input source path not found at: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("OCEAN WASTE PLASTIC DETECTION - INFERENCE PIPELINE")
    print("=" * 65)
    print(f"Model Weights       : {weights_path}")
    print(f"Device              : {device} ({'GPU' if device == 0 else 'CPU'})")
    print(f"Confidence Threshold: {conf_thresh}")
    print(f"Source              : {source_path}")
    print(f"Output Directory    : {output_dir}")
    print("-" * 65)

    model = YOLO(str(weights_path))

    # Collect images if directory, or single file
    if source_path.is_dir():
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted([f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions])
        if max_samples:
            image_files = image_files[:max_samples]
    else:
        image_files = [source_path]

    total_images = len(image_files)
    total_detected_objects = 0
    images_with_debris = 0

    print(f"Processing {total_images} image(s)...")

    for idx, img_file in enumerate(image_files, start=1):
        # Run inference
        results = model.predict(
            source=str(img_file),
            conf=conf_thresh,
            iou=iou_thresh,
            device=device,
            verbose=False,
        )

        res = results[0]
        boxes = res.boxes
        num_boxes = len(boxes)
        total_detected_objects += num_boxes

        if num_boxes > 0:
            images_with_debris += 1

        # Plot annotated image with custom labels and bounding boxes
        annotated_bgr = res.plot(line_width=2, font_size=0.6)

        # Add total count overlay banner at top of image
        banner_text = f"Plastic Waste Detected: {num_boxes}"
        cv2.putText(
            annotated_bgr,
            banner_text,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if num_boxes > 0 else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        save_path = output_dir / f"pred_{img_file.name}"
        cv2.imwrite(str(save_path), annotated_bgr)

        # Print per-image report
        conf_scores = [round(float(b.conf[0]), 3) for b in boxes] if num_boxes > 0 else []
        print(f"[{idx:2d}/{total_images}] {img_file.name[:25]:<25} -> {num_boxes:2d} plastic items | Conf: {conf_scores[:4]}")

    print("\n" + "=" * 65)
    print("INFERENCE COMPLETE SUMMARY")
    print("=" * 65)
    print(f"Total Images Processed        : {total_images}")
    print(f"Images with Plastic Debris    : {images_with_debris}")
    print(f"Total Plastic Objects Detected: {total_detected_objects}")
    if total_images > 0:
        print(f"Avg Objects per Image         : {total_detected_objects / total_images:.2f}")
    print(f"Annotated Output Saved at     : {output_dir.resolve()}")
    print("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser(description="Ocean Plastic Waste Detection Inference")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to YOLOv8 model weights (.pt)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to input image, video, or directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory to save annotated predictions",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--max-samples",
        "--num-samples",
        dest="max_samples",
        type=int,
        default=10,
        help="Max number of images to process when source is a directory (default: 10)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(
        weights_path=args.weights,
        source_path=args.source,
        output_dir=args.output,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        max_samples=args.max_samples,
    )
