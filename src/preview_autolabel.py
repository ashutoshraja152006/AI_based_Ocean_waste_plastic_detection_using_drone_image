import csv
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = Path("data/raw/marine debris from drone.v1 robo flow")
OUTPUT_DIR = Path("outputs/eda")


def get_valid_ocean_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Uses FLOODFILL_FIXED_RANGE from the 4 corners to mask out solid gray/black
    rotation border triangles and erodes the mask to create a safe interior zone.
    """
    h, w = img_bgr.shape[:2]
    valid_mask = np.ones((h, w), dtype=np.uint8) * 255
    corners = [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)]

    flags = cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)

    for cx, cy in corners:
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(
            img_bgr.copy(),
            flood_mask,
            seedPoint=(cx, cy),
            newVal=(255, 255, 255),
            loDiff=(10, 10, 10),
            upDiff=(10, 10, 10),
            flags=flags,
        )
        region = flood_mask[1:-1, 1:-1]
        area_ratio = np.count_nonzero(region) / float(h * w)
        touches_center = np.any(region[int(h * 0.32):int(h * 0.68), int(w * 0.32):int(w * 0.68)] > 0)

        if 0.0005 < area_ratio < 0.28 and not touches_center:
            valid_mask[region > 0] = 0

    # Erode valid region by 20 pixels along any detected rotation border
    if np.count_nonzero(valid_mask == 0) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        valid_mask = cv2.erode(valid_mask, kernel, iterations=1)

    # Always ignore a 10-pixel safety margin around the outer frame edge
    margin = 10
    valid_mask[:margin, :] = 0
    valid_mask[-margin:, :] = 0
    valid_mask[:, :margin] = 0
    valid_mask[:, -margin:] = 0

    return valid_mask


def merge_overlapping_boxes(boxes: list, gap_px: float = 4.0) -> list:
    """
    Merges any overlapping, contained, or nearly-touching boxes into a single
    clean bounding box [xmin, ymin, xmax, ymax, max_score] so large containers
    or bottles are never split into multiple pieces.
    """
    if not boxes:
        return []

    merged = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            bx1, by1, bx2, by2, bscore = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                ox1, oy1, ox2, oy2, oscore = merged[j]
                # Check if boxes overlap or are within gap_px pixels of each other
                if not (bx2 + gap_px < ox1 or ox2 + gap_px < bx1 or by2 + gap_px < oy1 or oy2 + gap_px < by1):
                    bx1 = min(bx1, ox1)
                    by1 = min(by1, oy1)
                    bx2 = max(bx2, ox2)
                    by2 = max(by2, oy2)
                    bscore = max(bscore, oscore)
                    used[j] = True
                    changed = True
            new_merged.append([bx1, by1, bx2, by2, bscore])
        merged = new_merged

    return merged


def detect_plastic_boxes(img_bgr: np.ndarray) -> list:
    """
    Detects floating plastic waste while suppressing ocean wave ripples and border edges.
    Returns list of [xmin, ymin, xmax, ymax, score].
    """
    h, w = img_bgr.shape[:2]
    valid_mask = get_valid_ocean_mask(img_bgr)
    valid_bool = valid_mask > 0

    if np.count_nonzero(valid_bool) < 0.15 * h * w:
        valid_bool = np.ones((h, w), dtype=bool)

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    med_lab = np.median(img_lab[valid_bool], axis=0)
    lab_filled = img_lab.copy()
    lab_filled[~valid_bool] = med_lab

    # Local ocean background smoothing
    bg_lab = cv2.GaussianBlur(lab_filled, (51, 51), 18)
    diff_lab = lab_filled - bg_lab

    # 1. Chromaticity contrast (colored plastic)
    chroma_dist = np.sqrt(diff_lab[:, :, 1] ** 2 + diff_lab[:, :, 2] ** 2)

    # 2. Morphological Top-Hat (27x27) for bright plastic bottles, masks, containers, caps
    se_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    tophat_l = cv2.morphologyEx(lab_filled[:, :, 0], cv2.MORPH_TOPHAT, se_tophat)

    # 3. Dark-object contrast on medium/bright water
    blackhat_l = np.zeros_like(tophat_l)
    if med_lab[0] > 60:
        blackhat_l = cv2.morphologyEx(lab_filled[:, :, 0], cv2.MORPH_BLACKHAT, se_tophat) * 0.75

    saliency = 1.8 * chroma_dist + 1.25 * tophat_l + blackhat_l
    saliency[~valid_bool] = 0.0

    s_valid = saliency[valid_bool]
    med_s = float(np.median(s_valid))
    mad_s = float(np.median(np.abs(s_valid - med_s))) + 1e-5
    robust_std = 1.4826 * mad_s

    thresh_val = max(med_s + 6.5 * robust_std, 28.0)
    binary = ((saliency >= thresh_val) & valid_bool).astype(np.uint8) * 255

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 14 or area > 0.12 * (h * w):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / float(min(bw, bh) + 1e-5)
        if aspect > 5.2:
            continue

        # Reject any box that touches the invalid border mask
        box_mask_crop = valid_mask[y:y + bh, x:x + bw]
        if np.mean(box_mask_crop > 0) < 0.95:
            continue

        roi_sal = saliency[y:y + bh, x:x + bw]
        peak_score = float(np.max(roi_sal))
        mean_score = float(np.mean(roi_sal))

        if peak_score < thresh_val * 1.25:
            continue

        pad = 3
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w - 1, x + bw + pad)
        y2 = min(h - 1, y + bh + pad)
        candidate_boxes.append([x1, y1, x2, y2, peak_score * 0.7 + mean_score * 0.3])

    merged = merge_overlapping_boxes(candidate_boxes, gap_px=4.0)

    if merged:
        max_score = max(b[4] for b in merged)
        merged = [b for b in merged if b[4] >= max(max_score * 0.32, 32.0)]
        merged = sorted(merged, key=lambda b: b[4], reverse=True)[:18]

    return merged


def select_10_unique_samples(train_dir: Path) -> list:
    csv_path = train_dir / "_classes.csv"
    sample_by_col = {i: [] for i in range(1, 6)}
    seen_files = set()

    with open(csv_path, mode="r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            fname = row[0].strip()
            if fname in seen_files:
                continue
            for col_idx in range(1, 6):
                val = int(row[col_idx].strip()) if row[col_idx].strip().isdigit() else 0
                if val == 1 and len(sample_by_col[col_idx]) < 2:
                    sample_by_col[col_idx].append(fname)
                    seen_files.add(fname)
                    break

    selected = []
    for row_idx in range(2):
        for col_idx in range(1, 6):
            if row_idx < len(sample_by_col[col_idx]):
                selected.append((col_idx, sample_by_col[col_idx][row_idx]))
    return selected


def main():
    print("=" * 65)
    print("STAGE 5 (FINAL PILOT): TESTING CLEAN AUTO-LABELER ON 10 IMAGES")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_dir = DATASET_DIR / "train"
    samples = select_10_unique_samples(train_dir)

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle("Final Verified YOLO Bounding Boxes (Green = plastic_waste)", fontsize=14, fontweight="bold")

    total_boxes = 0
    for idx, (col_idx, fname) in enumerate(samples):
        img_path = train_dir / fname
        img_bgr = cv2.imread(str(img_path))
        boxes = detect_plastic_boxes(img_bgr)
        total_boxes += len(boxes)

        vis_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        for (x1, y1, x2, y2, _) in boxes:
            cv2.rectangle(vis_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        r, c = divmod(idx, 5)
        ax = axes[r, c]
        ax.imshow(vis_rgb)
        short_name = fname.split(".rf.")[0][:16]
        ax.set_title(f"{short_name}\nDetected: {len(boxes)} plastic items", fontsize=9, fontweight="bold")
        ax.axis("off")
        print(f"  [{idx + 1:2d}/10] {short_name:<18} -> Detected {len(boxes):2d} plastic bounding boxes")

    plt.tight_layout()
    preview_path = OUTPUT_DIR / "autolabel_preview.png"
    plt.savefig(preview_path, dpi=200)
    plt.close()

    print("-" * 65)
    print(f"Total plastic objects detected across 10 unique sample images: {total_boxes}")
    print(f"[SAVED] Final Verified Preview -> {preview_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()