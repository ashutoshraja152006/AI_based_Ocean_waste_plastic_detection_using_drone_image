from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed/yolo_dataset")
OUTPUT_DIR = Path("outputs/eda")


def verify_split(split_name: str):
    img_dir = PROCESSED_DIR / split_name / "images"
    lbl_dir = PROCESSED_DIR / split_name / "labels"

    img_files = {f.stem: f for f in img_dir.glob("*.jpg")}
    lbl_files = {f.stem: f for f in lbl_dir.glob("*.txt")}

    missing_labels = set(img_files.keys()) - set(lbl_files.keys())
    orphan_labels = set(lbl_files.keys()) - set(img_files.keys())

    invalid_lines = []
    out_of_bounds_boxes = []
    box_counts_per_img = []
    box_widths_px = []
    box_heights_px = []

    for stem, lbl_path in sorted(lbl_files.items()):
        lines = [line.strip() for line in lbl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        box_counts_per_img.append(len(lines))

        for line_num, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                invalid_lines.append((lbl_path.name, line_num, "Not 5 values"))
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
            except ValueError:
                invalid_lines.append((lbl_path.name, line_num, "Non-numeric value"))
                continue

            if cls_id != 0:
                invalid_lines.append((lbl_path.name, line_num, f"Unexpected class_id {cls_id}"))

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                out_of_bounds_boxes.append((lbl_path.name, line_num, (xc, yc, bw, bh)))
            else:
                box_widths_px.append(bw * 640.0)
                box_heights_px.append(bh * 640.0)

    print(f"\n--- Verification Report for '{split_name}/' ---")
    print(f"  Images found             : {len(img_files)}")
    print(f"  Label .txt files found   : {len(lbl_files)}")
    print(f"  Missing label files      : {len(missing_labels)}")
    print(f"  Orphan label files       : {len(orphan_labels)}")
    print(f"  Invalid format lines     : {len(invalid_lines)}")
    print(f"  Out-of-bounds boxes      : {len(out_of_bounds_boxes)}")
    print(f"  Valid boxes verified     : {len(box_widths_px)}")
    print(f"  Min / Max boxes in image : {min(box_counts_per_img)} / {max(box_counts_per_img)}")

    passed = (
        len(img_files) > 0
        and len(missing_labels) == 0
        and len(orphan_labels) == 0
        and len(invalid_lines) == 0
        and len(out_of_bounds_boxes) == 0
    )
    print(f"  STATUS                   : {'PASSED (100% VALID)' if passed else 'FAILED'}")

    return box_counts_per_img, box_widths_px, box_heights_px


def save_visual_verification_grid():
    """Reads .txt files from train, valid, and test and draws boxes on 9 sample images."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    fig.suptitle("Stage 6: Visual Verification of Saved YOLO .txt Labels Across Splits", fontsize=14, fontweight="bold")

    splits = ["train", "valid", "test"]
    for row_idx, split in enumerate(splits):
        img_dir = PROCESSED_DIR / split / "images"
        lbl_dir = PROCESSED_DIR / split / "labels"
        sample_imgs = sorted(img_dir.glob("*.jpg"))[:3]

        for col_idx, img_path in enumerate(sample_imgs):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            img_bgr = cv2.imread(str(img_path))
            vis_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = vis_rgb.shape[:2]

            lines = [l.strip() for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            for line in lines:
                _, xc, yc, bw, bh = map(float, line.split())
                x1 = int((xc - bw / 2.0) * w)
                y1 = int((yc - bh / 2.0) * h)
                x2 = int((xc + bw / 2.0) * w)
                y2 = int((yc + bh / 2.0) * h)
                cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_rgb, "plastic", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)

            ax = axes[row_idx, col_idx]
            ax.imshow(vis_rgb)
            short_name = img_path.name.split(".rf.")[0][:18]
            ax.set_title(f"[{split.upper()}] {short_name}\n({len(lines)} boxes from .txt)", fontsize=10, fontweight="bold")
            ax.axis("off")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "verified_yolo_labels.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\n[SAVED] Visual label verification grid -> {out_path}")


def save_bbox_statistics_chart(all_counts, all_widths, all_heights):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Histogram of plastic objects per image
    bins = range(0, max(all_counts) + 2)
    axes[0].hist(all_counts, bins=bins, color="#2b83ba", edgecolor="black", align="left")
    axes[0].set_title("Distribution of Plastic Objects per Drone Image", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Plastic Waste Objects Detected in Image", fontsize=10)
    axes[0].set_ylabel("Number of Images", fontsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 2: Bounding box width vs height in pixels
    axes[1].hist(all_widths, bins=40, alpha=0.65, label="Box Width (px)", color="#fdae61", edgecolor="black")
    axes[1].hist(all_heights, bins=40, alpha=0.65, label="Box Height (px)", color="#abdda4", edgecolor="black")
    axes[1].set_title("Bounding Box Size Distribution (in Pixels on 640x640 Image)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Size in Pixels", fontsize=10)
    axes[1].set_ylabel("Number of Bounding Boxes", fontsize=10)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    stat_path = OUTPUT_DIR / "bbox_statistics.png"
    plt.savefig(stat_path, dpi=200)
    plt.close()
    print(f"[SAVED] Bounding box statistics chart  -> {stat_path}")


def main():
    print("=" * 65)
    print("STAGE 6: YOLO ANNOTATION VERIFICATION & QUALITY CHECK")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_counts, all_widths, all_heights = [], [], []

    for split in ["train", "valid", "test"]:
        c, w, h = verify_split(split)
        all_counts.extend(c)
        all_widths.extend(w)
        all_heights.extend(h)

    save_visual_verification_grid()
    save_bbox_statistics_chart(all_counts, all_widths, all_heights)

    print("\n" + "=" * 65)
    print("ANNOTATION VERIFICATION COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    main()