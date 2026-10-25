import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

BEST_WEIGHTS = Path("outputs/models/ocean_plastic_yolov8n/weights/best.pt")
DATA_YAML = Path("configs/data.yaml")
EVAL_DIR = Path("outputs/evaluation")


def extract_metrics(val_results) -> dict:
    p = float(val_results.box.mp)
    r = float(val_results.box.mr)
    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {
        "Precision": round(p, 4),
        "Recall": round(r, 4),
        "F1_Score": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
    }


def save_comparison_chart(val_metrics: dict, test_metrics: dict, out_dir: Path):
    labels = ["Precision", "Recall", "F1_Score", "mAP50", "mAP50_95"]
    display_labels = ["Precision", "Recall", "F1-Score", "mAP@50", "mAP@50-95"]
    val_vals = [val_metrics[k] * 100 for k in labels]
    test_vals = [test_metrics[k] * 100 for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width / 2, val_vals, width, label="Validation Split (572 images)", color="#1f77b4", edgecolor="black")
    rects2 = ax.bar(x + width / 2, test_vals, width, label="Unseen Test Split (263 images)", color="#2ca02c", edgecolor="black")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("YOLOv8n Model Evaluation: Validation vs. Unseen Test Set", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            h + 1.2,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout()
    chart_path = out_dir / "metrics_comparison.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"[SAVED] Validation vs. Test metrics chart -> {chart_path}")


def main():
    print("=" * 65)
    print("STAGE 11: FORMAL MODEL EVALUATION ON VALIDATION & UNSEEN TEST SET")
    print("=" * 65)

    if not BEST_WEIGHTS.exists():
        print(f"[ERROR] Trained weights not found at: {BEST_WEIGHTS}")
        return

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BEST_WEIGHTS))

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n1. Evaluating on VALIDATION split (572 images) using device: {device}...")
    val_res = model.val(
        data=str(DATA_YAML.resolve()),
        split="val",
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        project=str(EVAL_DIR.resolve()),
        name="val_split",
        exist_ok=True,
        verbose=False,
    )
    val_metrics = extract_metrics(val_res)

    print(f"\n2. Evaluating on UNSEEN TEST split (263 images) using device: {device}...")
    test_res = model.val(
        data=str(DATA_YAML.resolve()),
        split="test",
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        project=str(EVAL_DIR.resolve()),
        name="test_split",
        exist_ok=True,
        verbose=False,
    )
    test_metrics = extract_metrics(test_res)

    summary = {
        "model": str(BEST_WEIGHTS),
        "validation_split": val_metrics,
        "unseen_test_split": test_metrics,
    }

    json_path = EVAL_DIR / "evaluation_metrics.json"
    json_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print("\n" + "=" * 65)
    print("FINAL EVALUATION SUMMARY (VALIDATION vs. UNSEEN TEST SET)")
    print("=" * 65)
    print(f"{'Metric':<15} | {'Validation (572 imgs)':<22} | {'Unseen Test (263 imgs)':<22}")
    print("-" * 65)
    for k in ["Precision", "Recall", "F1_Score", "mAP50", "mAP50_95"]:
        v_str = f"{val_metrics[k] * 100:.2f}% ({val_metrics[k]:.4f})"
        t_str = f"{test_metrics[k] * 100:.2f}% ({test_metrics[k]:.4f})"
        print(f"{k:<15} | {v_str:<22} | {t_str:<22}")
    print("=" * 65)

    save_comparison_chart(val_metrics, test_metrics, EVAL_DIR)
    print(f"[SAVED] Metrics JSON report               -> {json_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()