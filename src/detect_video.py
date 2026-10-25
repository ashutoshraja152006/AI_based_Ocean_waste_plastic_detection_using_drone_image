import argparse
import time
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path("outputs/models/ocean_plastic_yolov8n/weights/best.pt")
DEFAULT_OUTPUT = Path("outputs/predictions/detected_drone_video.mp4")


def process_video(
    video_source: str,
    weights_path: Path,
    output_path: Path,
    conf_thresh: float = 0.30,
    iou_thresh: float = 0.45,
    show_live: bool = False,
):
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at: {weights_path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("OCEAN WASTE PLASTIC DETECTION - VIDEO SURVEILLANCE PIPELINE")
    print("=" * 65)
    print(f"Model Weights       : {weights_path}")
    print(f"Device              : {device} ({'GPU' if device == 0 else 'CPU'})")
    print(f"Confidence Threshold: {conf_thresh}")
    print(f"Video Source        : {video_source}")
    print(f"Output Video Path   : {output_path}")
    print("-" * 65)

    model = YOLO(str(weights_path))

    # Open video capture
    is_webcam = video_source.isdigit()
    cap = cv2.VideoCapture(int(video_source) if is_webcam else video_source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()
    total_plastic_boxes = 0

    print(f"Processing Video ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t_frame_start = time.time()

        # Run inference on current frame
        results = model.predict(
            source=frame,
            conf=conf_thresh,
            iou=iou_thresh,
            device=device,
            verbose=False,
        )

        res = results[0]
        num_boxes = len(res.boxes)
        total_plastic_boxes += num_boxes

        annotated = res.plot(line_width=2, font_size=0.6)

        # Calculate instantaneous FPS
        fps_current = 1.0 / max(time.time() - t_frame_start, 1e-4)

        # Draw banner & status on frame
        banner_text = f"Plastic Debris Detected: {num_boxes} | FPS: {fps_current:.1f}"
        cv2.putText(
            annotated,
            banner_text,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255) if num_boxes > 0 else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        out_writer.write(annotated)

        if show_live:
            cv2.imshow("Drone Ocean Waste Surveillance", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Interrupted by user.")
                break

        if frame_count % 30 == 0 or frame_count == total_frames:
            pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
            print(f"  Frame {frame_count:4d}/{total_frames:4d} ({pct:5.1f}%) -> {num_boxes:2d} plastic items | {fps_current:.1f} FPS")

    cap.release()
    out_writer.release()
    if show_live:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 65)
    print("VIDEO PROCESSING COMPLETE SUMMARY")
    print("=" * 65)
    print(f"Total Frames Processed : {frame_count}")
    print(f"Total Time Elapsed     : {elapsed:.2f} seconds")
    print(f"Average Inference Speed: {avg_fps:.1f} FPS")
    print(f"Annotated Video Saved  : {output_path.resolve()}")
    print("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser(description="Drone Video Ocean Plastic Detection")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file (.mp4, .avi) or '0' for live webcam/drone feed",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to trained YOLOv8 model weights (.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output annotated video file (.mp4)",
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
        "--show",
        action="store_true",
        help="Show live OpenCV display window while processing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(
        video_source=args.video,
        weights_path=args.weights,
        output_path=args.output,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        show_live=args.show,
    )
