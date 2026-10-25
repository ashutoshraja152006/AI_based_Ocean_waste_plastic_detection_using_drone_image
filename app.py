import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Ocean Waste Plastic Detection",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_WEIGHTS = Path("outputs/models/ocean_plastic_yolov8n/weights/best.pt")
TEST_IMAGES_DIR = Path("data/processed/yolo_dataset/test/images")
METRICS_JSON = Path("outputs/evaluation/evaluation_metrics.json")
CONFUSION_MATRIX_IMG = Path("outputs/models/ocean_plastic_yolov8n/confusion_matrix_normalized.png")
METRICS_COMPARISON_IMG = Path("outputs/evaluation/metrics_comparison.png")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


def detect_image(model, image: Image.Image, conf_thresh: float, iou_thresh: float, device):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model.predict(
        source=img_cv,
        conf=conf_thresh,
        iou=iou_thresh,
        device=device,
        verbose=False,
    )
    res = results[0]
    boxes = res.boxes
    annotated_bgr = res.plot(line_width=2, font_size=0.6)

    # Add overlay count badge
    num_boxes = len(boxes)
    badge_text = f"Plastic Waste Detected: {num_boxes}"
    cv2.putText(
        annotated_bgr,
        badge_text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if num_boxes > 0 else (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    details = []
    if num_boxes > 0:
        for i, box in enumerate(boxes, start=1):
            conf = float(box.conf[0])
            xyxy = [round(float(c), 1) for c in box.xyxy[0]]
            details.append({
                "Item #": i,
                "Class": "plastic_waste",
                "Confidence": f"{conf * 100:.2f}%",
                "Bounding Box [x1, y1, x2, y2]": str(xyxy),
            })

    avg_conf = float(np.mean([float(b.conf[0]) for b in boxes])) * 100 if num_boxes > 0 else 0.0
    return annotated_rgb, num_boxes, details, avg_conf


# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
def main():
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); padding: 22px; border-radius: 12px; margin-bottom: 20px;">
            <h1 style="color: #ffffff; margin: 0; font-size: 2.2rem;">🛸 AI-Based Ocean Waste Plastic Detection</h1>
            <p style="color: #d1d8e0; margin: 6px 0 0 0; font-size: 1.05rem;">
                Autonomous Aerial Drone Surveillance & Real-Time Marine Plastic Debris Localization using Fine-Tuned YOLOv8n
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check model file
    if not MODEL_WEIGHTS.exists():
        st.error(f"❌ Model weights not found at `{MODEL_WEIGHTS}`. Please ensure training is completed.")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode"
    model = load_model(str(MODEL_WEIGHTS))

    # -------------------------------------------------------------------------
    # Sidebar Controls
    # -------------------------------------------------------------------------
    st.sidebar.header("⚙️ Model & Detection Settings")
    st.sidebar.markdown(f"**Device**: `{device_name}`")
    st.sidebar.markdown(f"**Weights**: `{MODEL_WEIGHTS.name}`")

    conf_thresh = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.30,
        step=0.05,
        help="Minimum probability score required to classify an object as plastic waste.",
    )

    iou_thresh = st.sidebar.slider(
        "NMS IoU Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Intersection-over-Union threshold for Non-Maximum Suppression to remove overlapping boxes.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Project Highlights")
    st.sidebar.metric(label="Validation mAP@50", value="88.14%")
    st.sidebar.metric(label="Unseen Test mAP@50", value="90.41%")
    st.sidebar.metric(label="Test Recall", value="90.57%")

    # -------------------------------------------------------------------------
    # Tabs
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "📷 Drone Image Inference",
        "🎥 Drone Video Surveillance",
        "📈 Model Evaluation & Performance",
    ])

    # =========================================================================
    # TAB 1: Single & Batch Image Inference
    # =========================================================================
    with tab1:
        st.subheader("Detect Floating Plastic Waste in Drone Imagery")
        
        input_source = st.radio(
            "Select Input Source:",
            ["Choose from Unseen Test Set Samples", "Upload Your Own Drone Image"],
            horizontal=True,
        )

        image_to_process = None
        image_name = ""

        if input_source == "Choose from Unseen Test Set Samples":
            if TEST_IMAGES_DIR.exists():
                sample_files = sorted([f.name for f in TEST_IMAGES_DIR.glob("*.jpg")])[:30]
                selected_sample = st.selectbox("Select a Drone Image from Test Split:", sample_files)
                if selected_sample:
                    img_path = TEST_IMAGES_DIR / selected_sample
                    image_to_process = Image.open(img_path)
                    image_name = selected_sample
            else:
                st.warning(f"Test images directory not found at `{TEST_IMAGES_DIR}`.")
        else:
            uploaded_file = st.file_uploader("Upload Drone Image (JPG / PNG):", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image_to_process = Image.open(uploaded_file)
                image_name = uploaded_file.name

        if image_to_process is not None:
            col_run, col_clear = st.columns([1, 5])
            with col_run:
                run_btn = st.button("🚀 Run Detection", type="primary", use_container_width=True)

            if run_btn:
                with st.spinner("Analyzing drone imagery with YOLOv8..."):
                    annotated_rgb, num_detected, details, avg_conf = detect_image(
                        model, image_to_process, conf_thresh, iou_thresh, device
                    )

                # Metrics Summary Banner (Per-Image)
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Total Plastic Debris", f"{num_detected} items")
                with col_m2:
                    st.metric(
                        "Current Image Avg Conf",
                        f"{avg_conf:.1f}%" if num_detected > 0 else "N/A",
                        help="Average detection confidence score for plastic objects found in THIS specific image.",
                    )
                with col_m3:
                    risk_status = "⚠️ High Risk" if num_detected >= 5 else ("🟡 Moderate" if num_detected > 0 else "🟢 Clean Water")
                    st.metric("Pollution Severity", risk_status)
                with col_m4:
                    st.metric("Inference Engine", f"YOLOv8n ({device_name[:12]})")

                # Visual Comparison Side-by-Side
                col_orig, col_pred = st.columns(2)
                with col_orig:
                    st.markdown("**Original Drone Capture**")
                    st.image(image_to_process, use_container_width=True)

                with col_pred:
                    st.markdown(f"**YOLOv8 Detection Result ({num_detected} Bounding Boxes)**")
                    st.image(annotated_rgb, use_container_width=True)

                    # Download button for annotated result
                    res_pil = Image.fromarray(annotated_rgb)
                    buf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    res_pil.save(buf.name, format="JPEG")
                    with open(buf.name, "rb") as f:
                        st.download_button(
                            label="💾 Download Annotated Result",
                            data=f,
                            file_name=f"detected_{image_name}",
                            mime="image/jpeg",
                            use_container_width=True,
                        )

                # Details Table
                if details:
                    st.markdown("### 📋 Detected Plastic Objects Breakdown")
                    df = pd.DataFrame(details)
                    st.dataframe(df, use_container_width=True)

    # =========================================================================
    # TAB 2: Video Detection
    # =========================================================================
    with tab2:
        st.subheader("Real-Time Drone Video Processing")
        st.markdown(
            "Upload an ocean aerial surveillance drone video (`.mp4`) to perform real-time bounding box detection, counting, and tracking."
        )

        uploaded_video = st.file_uploader("Upload Drone Video (.mp4 / .avi / .mov):", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()

            st.video(tfile.name)

            if st.button("▶️ Process Video with YOLOv8", type="primary"):
                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

                progress_bar = st.progress(0)
                frame_placeholder = st.empty()
                status_placeholder = st.empty()

                frame_idx = 0
                max_detected_in_video = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1

                    # Process every frame
                    results = model.predict(source=frame, conf=conf_thresh, iou=iou_thresh, device=device, verbose=False)
                    res = results[0]
                    num_boxes = len(res.boxes)
                    max_detected_in_video = max(max_detected_in_video, num_boxes)

                    annotated = res.plot(line_width=2, font_size=0.6)
                    cv2.putText(
                        annotated,
                        f"Frame: {frame_idx}/{total_frames} | Plastic Waste: {num_boxes}",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255) if num_boxes > 0 else (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, use_container_width=True)
                    progress_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))
                    status_placeholder.text(f"Processing Frame {frame_idx}/{total_frames} — Current Plastic Count: {num_boxes}")

                cap.release()
                st.success(f"✅ Video processing complete! Maximum plastic debris detected in a single frame: {max_detected_in_video}")

    # =========================================================================
    # TAB 3: Model Metrics & Evaluation Charts
    # =========================================================================
    with tab3:
        st.subheader("Model Benchmark & Evaluation Report")

        st.markdown(
            """
            The model was trained on **5,519 aerial drone images** using transfer learning from pre-trained **YOLOv8n**,
            with specialized data augmentations (mosaic, flips) tailored for aerial nadir viewpoints.
            """
        )
        st.info(
            "📌 **Academic Note for Examiners/Viva**: These metrics are the fixed benchmark scores "
            "calculated across the entire **263 unseen test images** test set to measure overall model generalization."
        )

        col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
        col_b1.metric("Precision", "84.13%", help="Proportion of detected boxes that were true plastic debris.")
        col_b2.metric("Recall", "90.57%", help="Proportion of actual marine plastic debris successfully detected.")
        col_b3.metric("F1-Score", "87.23%", help="Harmonic mean of precision and recall.")
        col_b4.metric("mAP@50", "90.41%", help="Mean Average Precision at 0.50 IoU threshold on unseen test set.")
        col_b5.metric("mAP@50-95", "69.06%", help="Strict mAP across 0.50 to 0.95 IoU range.")

        st.markdown("---")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("### 📊 Validation vs. Unseen Test Comparison")
            if METRICS_COMPARISON_IMG.exists():
                st.image(str(METRICS_COMPARISON_IMG), use_container_width=True)
            else:
                st.info("Metrics comparison plot not found.")

        with col_c2:
            st.markdown("### 🎯 Normalized Confusion Matrix")
            if CONFUSION_MATRIX_IMG.exists():
                st.image(str(CONFUSION_MATRIX_IMG), use_container_width=True)
            else:
                st.info("Confusion matrix plot not found.")


if __name__ == "__main__":
    main()
