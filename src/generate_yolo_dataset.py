import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
from tqdm import tqdm

# Import our verified detector from preview_autolabel.py
from src.preview_autolabel import detect_plastic_boxes

RAW_DATASET_DIR = Path("data/raw/marine debris from drone.v1 robo flow")
PROCESSED_DATASET_DIR = Path("data/processed/yolo_dataset")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_single_image(args):
    """
    Worker function that reads one raw image, detects plastic bounding boxes,
    copies the image to data/processed/yolo_dataset/<split>/images/,
    and writes the normalized YOLO .txt label file to <split>/labels/.
    """
    img_path, dst_img_dir, dst_lbl_dir = args
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return 0, False

    h, w = img_bgr.shape[:2]
    boxes = detect_plastic_boxes(img_bgr)

    # Copy image safely to processed folder (raw image remains untouched)
    dst_img_path = dst_img_dir / img_path.name
    if not dst_img_path.exists():
        shutil.copy2(img_path, dst_img_path)

    # Convert pixel boxes [x1, y1, x2, y2, score] -> YOLO normalized [0 x_c y_c bw bh]
    yolo_lines = []
    for (x1, y1, x2, y2, _) in boxes:
        x_center = ((x1 + x2) / 2.0) / w
        y_center = ((y1 + y2) / 2.0) / h
        box_w = (x2 - x1) / w
        box_h = (y2 - y1) / h

        # Clamp strictly to [0.0, 1.0]
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        box_w = min(max(box_w, 0.0), 1.0)
        box_h = min(max(box_h, 0.0), 1.0)

        if box_w > 0.002 and box_h > 0.002:
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    dst_lbl_path = dst_lbl_dir / f"{img_path.stem}.txt"
    dst_lbl_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

    return len(yolo_lines), len(yolo_lines) == 0


def process_split(split_name: str):
    src_split_dir = RAW_DATASET_DIR / split_name
    dst_img_dir = PROCESSED_DATASET_DIR / split_name / "images"
    dst_lbl_dir = PROCESSED_DATASET_DIR / split_name / "labels"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in src_split_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    tasks = [(img_path, dst_img_dir, dst_lbl_dir) for img_path in image_files]

    total_boxes = 0
    empty_images = 0

    print(f"\nProcessing '{split_name}/' ({len(image_files)} images)...")
    with ProcessPoolExecutor(max_workers=8) as executor:
        for num_boxes, is_empty in tqdm(executor.map(process_single_image, tasks, chunksize=16), total=len(tasks), ncols=80):
            total_boxes += num_boxes
            if is_empty:
                empty_images += 1

    avg_boxes = total_boxes / len(image_files) if image_files else 0.0
    print(f"  -> Done '{split_name}/':")
    print(f"     Images copied        : {len(image_files)}")
    print(f"     Label .txt files     : {len(image_files)}")
    print(f"     Total plastic boxes  : {total_boxes}")
    print(f"     Avg boxes per image  : {avg_boxes:.2f}")
    print(f"     Images with 0 boxes  : {empty_images} (background ocean samples)")

    return len(image_files), total_boxes, empty_images


def main():
    print("=" * 65)
    print("STAGE 5: GENERATING FULL YOLO BOUNDING-BOX DATASET (6,354 IMAGES)")
    print("=" * 65)
    print(f"Source (Read-Only) : {RAW_DATASET_DIR.resolve()}")
    print(f"Destination        : {PROCESSED_DATASET_DIR.resolve()}")

    grand_images = 0
    grand_boxes = 0
    grand_empty = 0

    for split in ["train", "valid", "test"]:
        n_img, n_box, n_emp = process_split(split)
        grand_images += n_img
        grand_boxes += n_box
        grand_empty += n_emp

    print("\n" + "=" * 65)
    print("FULL YOLO DATASET GENERATION COMPLETE!")
    print("=" * 65)
    print(f"  Total Images Processed : {grand_images}")
    print(f"  Total Bounding Boxes   : {grand_boxes}")
    print(f"  Background Images (0)  : {grand_empty}")
    print("=" * 65)


if __name__ == "__main__":
    main()