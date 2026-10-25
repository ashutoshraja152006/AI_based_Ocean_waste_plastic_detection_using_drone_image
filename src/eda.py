import csv
from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

# Paths
DATASET_DIR = Path("data/raw/marine debris from drone.v1 robo flow")
OUTPUT_DIR = Path("outputs/eda")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_images_and_csv(dataset_dir: Path, split_name: str):
    split_dir = dataset_dir / split_name
    image_files = sorted([f for f in split_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])

    resolutions = Counter()
    color_modes = Counter()
    corrupted_files = []
    zero_byte_files = []

    for img_path in image_files:
        if img_path.stat().st_size == 0:
            zero_byte_files.append(img_path.name)
            continue
        try:
            with Image.open(img_path) as img:
                resolutions[img.size] += 1  # (width, height)
                color_modes[img.mode] += 1  # e.g., 'RGB'
        except Exception as e:
            corrupted_files.append((img_path.name, str(e)))

    # Analyze the 5 binary columns in _classes.csv
    csv_path = split_dir / "_classes.csv"
    col_positive_counts = Counter()
    row_sum_counts = Counter()  # How many active '1' flags per image
    sample_by_col = {i: [] for i in range(1, 6)}

    if csv_path.exists():
        with open(csv_path, mode="r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # skip corrupted header
            for row in reader:
                if len(row) < 6:
                    continue
                fname = row[0].strip()
                flags = []
                for col_idx in range(1, 6):
                    val = int(row[col_idx].strip()) if row[col_idx].strip().isdigit() else 0
                    flags.append(val)
                    if val == 1:
                        col_positive_counts[f"Col_{col_idx}"] += 1
                        if len(sample_by_col[col_idx]) < 3:
                            sample_by_col[col_idx].append((fname, flags))
                row_sum_counts[sum(flags)] += 1

    print(f"\n--- Split: '{split_name}/' ---")
    print(f"  Total Images Checked   : {len(image_files)}")
    print(f"  Zero-Byte Images       : {len(zero_byte_files)}")
    print(f"  Corrupted Images       : {len(corrupted_files)}")
    print(f"  Image Resolutions (WxH): {dict(resolutions)}")
    print(f"  Color Modes (Channels) : {dict(color_modes)}")
    print(f"  CSV Positive Counts    : {dict(sorted(col_positive_counts.items()))}")
    print(f"  Active Flags per Image : {dict(sorted(row_sum_counts.items()))} (key = number of '1's in row)")

    return {
        "count": len(image_files),
        "corrupted": corrupted_files,
        "resolutions": resolutions,
        "col_counts": col_positive_counts,
        "sample_by_col": sample_by_col,
    }


def save_split_chart(split_stats: dict, output_dir: Path):
    splits = ["train", "valid", "test"]
    counts = [split_stats[s]["count"] for s in splits]
    total = sum(counts)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(splits, counts, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black")
    plt.title("Dataset Split Distribution (Total: 6,354 Drone Images)", fontsize=13, fontweight="bold")
    plt.xlabel("Dataset Split", fontsize=11)
    plt.ylabel("Number of Images", fontsize=11)
    plt.ylim(0, max(counts) * 1.15)

    for bar, count in zip(bars, counts):
        pct = (count / total) * 100 if total > 0 else 0
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart_path = output_dir / "dataset_split_distribution.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\n[SAVED] Split distribution chart -> {chart_path}")


def save_sample_grid(train_dir: Path, sample_by_col: dict, output_dir: Path):
    # Pick up to 2 samples from each of the 5 CSV columns (total 10 images) in a 2x5 grid
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle("Sample Drone Images from 'train/' Across the 5 CSV Columns", fontsize=14, fontweight="bold")

    for col_idx in range(1, 6):
        samples = sample_by_col.get(col_idx, [])
        for row_idx in range(2):
            ax = axes[row_idx, col_idx - 1]
            if row_idx < len(samples):
                fname, flags = samples[row_idx]
                img_path = train_dir / fname
                if img_path.exists():
                    with Image.open(img_path) as img:
                        ax.imshow(img)
                    short_name = fname.split(".rf.")[0][:18]
                    ax.set_title(f"Col_{col_idx}=1 | {short_name}\nFlags: {flags}", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    grid_path = output_dir / "sample_drone_images.png"
    plt.savefig(grid_path, dpi=200)
    plt.close()
    print(f"[SAVED] Sample drone images grid -> {grid_path}")


def main():
    print("=" * 65)
    print("STAGE 4: EXPLORATORY DATA ANALYSIS (EDA) & INTEGRITY CHECK")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_stats = {}

    for split in ["train", "valid", "test"]:
        split_stats[split] = check_images_and_csv(DATASET_DIR, split)

    save_split_chart(split_stats, OUTPUT_DIR)
    save_sample_grid(DATASET_DIR / "train", split_stats["train"]["sample_by_col"], OUTPUT_DIR)

    print("\n" + "=" * 65)
    print("EDA COMPLETE! Check the 'outputs/eda/' folder in VS Code.")
    print("=" * 65)


if __name__ == "__main__":
    main()