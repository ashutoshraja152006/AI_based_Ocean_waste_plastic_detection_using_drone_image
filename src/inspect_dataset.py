import csv
from pathlib import Path
from collections import Counter

# Path to your dataset inside data/raw
DATASET_DIR = Path("data/raw/marine debris from drone.v1 robo flow")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_readme_files(dataset_dir: Path):
    print("=" * 65)
    print("1. READING DATASET README FILES")
    print("=" * 65)
    for readme_name in ["README.dataset.txt", "README.roboflow.txt"]:
        readme_path = dataset_dir / readme_name
        if readme_path.exists():
            print(f"\n--- {readme_name} ---")
            print(readme_path.read_text(encoding="utf-8", errors="ignore").strip())
            print("-" * 45)


def inspect_csv_in_split(dataset_dir: Path, split_name: str):
    split_dir = dataset_dir / split_name
    print("\n" + "=" * 65)
    print(f"2. INSPECTING CSV ANNOTATIONS IN: '{split_name}/'")
    print("=" * 65)

    if not split_dir.exists():
        print(f"  [WARNING] '{split_name}/' folder not found!")
        return Counter()

    image_files = {f.name for f in split_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS}
    csv_files = [f for f in split_dir.iterdir() if f.is_file() and f.suffix.lower() == ".csv"]

    print(f"  Total image files on disk : {len(image_files)}")
    print(f"  CSV files found           : {[f.name for f in csv_files]}")

    if not csv_files:
        print("  [WARNING] No CSV file found in this split!")
        return Counter()

    csv_path = csv_files[0]
    class_counts = Counter()
    images_in_csv = set()
    total_rows = 0
    sample_rows = []
    missing_images_on_disk = set()

    with open(csv_path, mode="r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        print(f"  CSV Column Headers        : {header}")

        # Determine column indices if standard object detection CSV
        col_map = {col.strip().lower(): idx for idx, col in enumerate(header)} if header else {}
        filename_idx = col_map.get("filename", 0)
        class_idx = col_map.get("class", None)

        has_bbox_cols = all(k in col_map for k in ["xmin", "ymin", "xmax", "ymax"])
        print(f"  Has Bounding Box Columns? : {has_bbox_cols} (xmin, ymin, xmax, ymax)")

        for row in reader:
            if not row:
                continue
            total_rows += 1
            if len(sample_rows) < 5:
                sample_rows.append(row)

            fname = row[filename_idx].strip()
            images_in_csv.add(fname)
            if fname not in image_files:
                missing_images_on_disk.add(fname)

            if class_idx is not None and class_idx < len(row):
                class_counts[row[class_idx].strip()] += 1

    unannotated_images = image_files - images_in_csv

    print(f"  Total annotation rows     : {total_rows} (total bounding boxes in '{split_name}')")
    print(f"  Unique images in CSV      : {len(images_in_csv)}")
    print(f"  Images with 0 annotations : {len(unannotated_images)} (background/unlabeled images)")
    print(f"  CSV images missing on disk: {len(missing_images_on_disk)}")
    print(f"  Classes & Box Counts      : {dict(class_counts)}")

    print("\n  Sample first 5 rows from CSV:")
    print(f"    HEADER -> {header}")
    for i, srow in enumerate(sample_rows, start=1):
        print(f"    Row {i}  -> {srow}")

    return class_counts


def main():
    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset folder not found: {DATASET_DIR.resolve()}")
        return

    read_readme_files(DATASET_DIR)

    overall_classes = Counter()
    for split in ["train", "valid", "test"]:
        split_counts = inspect_csv_in_split(DATASET_DIR, split)
        overall_classes.update(split_counts)

    print("\n" + "=" * 65)
    print("3. OVERALL DATASET CLASS SUMMARY (ACROSS ALL SPLITS)")
    print("=" * 65)
    print(f"  Total Number of Classes : {len(overall_classes)}")
    print(f"  Class Names             : {list(overall_classes.keys())}")
    print(f"  Total Boxes per Class   : {dict(overall_classes)}")
    print("=" * 65)


if __name__ == "__main__":
    main()