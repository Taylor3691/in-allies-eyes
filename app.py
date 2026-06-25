import sys
import os
import cv2
import gradio as gr
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from Kalman_filter import KalmanTracker

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global Kalman Tracker and State variables
tracker = KalmanTracker()
latest_frame = None
missing_frames_counter = 0
camera_active = False
use_kalman_global = False

def update_kalman_state(val):
    global use_kalman_global
    use_kalman_global = val

# Stub callbacks for Re-ID Tab searches and Comparisons
def search_gallery(query_img, use_caj, top_k):
    return [], []

def run_comparison(raw_img, kalman_img):
    return [], [], [], []

def reset_tracker():
    global tracker, missing_frames_counter
    tracker = KalmanTracker()
    missing_frames_counter = 0
    return "Missing", "Predict only"

def toggle_camera(use_kf):
    global camera_active, latest_frame, missing_frames_counter, use_kalman_global
    
    if camera_active:
        # Stop the camera loop
        camera_active = False
        yield None, "Start Camera", "Missing", "Predict only"
        return
        
    camera_active = True
    use_kalman_global = use_kf
    # Open direct connection to system webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        camera_active = False
        yield None, "Start Camera", "Camera Offline", "Predict only"
        return
        
    # Configure webcam stream properties for fast processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Stream frames directly in a loop using Python generators
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for a mirrored preview
        frame = cv2.flip(frame, 1)
        
        # Store latest frame as RGB for cropping
        latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detection preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        raw_bbox = None
        
        if len(faces) > 0:
            # Pick largest detection
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            raw_bbox = (int(x), int(y), int(w), int(h))
            missing_frames_counter = 0
        else:
            missing_frames_counter += 1
            
        # Run Kalman state matrices predictions and corrections
        if use_kalman_global:
            tracker.predict()
            if raw_bbox is not None:
                tracker.update(raw_bbox)
            else:
                if missing_frames_counter > 30:
                    tracker.state = None
                    tracker.status = "Missing"
                    tracker.mode = "Predict only"
        else:
            if raw_bbox is not None:
                tracker.state = None
                tracker.update(raw_bbox)
                tracker.status = "Found"
                tracker.mode = "Predict only"
            else:
                tracker.state = None
                tracker.status = "Missing"
                tracker.mode = "Predict only"
                
        # Draw target overlays
        output_frame = frame.copy()
        
        if raw_bbox is not None:
            rx, ry, rw, rh = raw_bbox
            cv2.rectangle(output_frame, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)  # Red raw box
            
        if use_kalman_global and tracker.state is not None:
            kf_rect = tracker.get_rect()
            if kf_rect is not None:
                kx, ky, kw, kh = kf_rect
                fh, fw_img, _ = frame.shape
                kx = max(0, min(kx, fw_img))
                ky = max(0, min(ky, fh))
                kw = max(1, min(kw, fw_img - kx))
                kh = max(1, min(kh, fh - ky))
                cv2.rectangle(output_frame, (kx, ky), (kx + kw, ky + kh), (0, 255, 0), 2)  # Green Kalman box
                
        # Convert output to RGB for display in Gradio UI
        output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        yield output_frame_rgb, "Stop Camera", tracker.status, tracker.mode
        
        # Yield at ~30 FPS to ensure smooth display and direct thread control
        time.sleep(0.03)
        
    cap.release()
    yield None, "Start Camera", "Missing", "Predict only"

def capture_query(use_kf, camera_id):
    global latest_frame
    if latest_frame is None:
        return None, "No active frame captured", "N/A", "N/A"

    # Convert RGB frame back to BGR for OpenCV processing
    latest_bgr = cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(latest_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    raw_bbox = None

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        raw_bbox = faces[0]

    crop = None
    source = "None"

    if use_kf:
        kf_rect = tracker.get_rect()
        if kf_rect is not None:
            kx, ky, kw, kh = kf_rect
            fh, fw_img, _ = latest_bgr.shape
            kx = max(0, min(kx, fw_img))
            ky = max(0, min(ky, fh))
            kw = max(1, min(kw, fw_img - kx))
            kh = max(1, min(kh, fh - ky))
            crop = latest_bgr[ky:ky+kh, kx:kx+kw]
            source = "Kalman bbox"
    else:
        if raw_bbox is not None:
            x, y, w, h = raw_bbox
            crop = latest_bgr[y:y+h, x:x+w]
            source = "Raw bbox"

    if crop is None or crop.size == 0:
        return None, "No target found to crop", "N/A", "N/A"

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    quality = f"Contrast: {int(np.std(crop_rgb))}"

    # Save to disk
    save_dir = "captured/kalman" if use_kf else "captured/original"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"crop_{int(time.time())}.png"
    cv2.imwrite(os.path.join(save_dir, filename), crop)

    return crop_rgb, source, camera_id, quality

# Gradio Blocks layout initialization
with gr.Blocks(title="Person Re-ID Demo App") as demo:
    gr.Markdown("# Person Re-Identification (Re-ID) Demo Interface")
    gr.Markdown("### Track targets with Kalman Filter & Optimize retrieval with CA-Jaccard Re-ranking")
    
    # -------------------------------------------------------------
    # Tab 1: Realtime Capture / Kalman Demo
    # -------------------------------------------------------------
    with gr.Tab("Realtime Capture / Kalman Demo"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("#### Webcam Video Stream")
                # Single display for output; direct camera streaming loop
                webcam_placeholder = gr.Image(label="Live Video Feed (Webcam Stream)")
                
            with gr.Column(scale=1):
                gr.Markdown("#### Bounding Box & Tracking Controls")
                # Custom toggle button outside the frame
                btn_toggle_cam = gr.Button("Start Camera", variant="secondary")
                
                use_kalman = gr.Checkbox(label="Use Kalman Filter", value=False)
                det_status = gr.Label(value="Missing", label="Detector Status")
                kf_mode = gr.Label(value="Predict only", label="Kalman Mode")
                cam_id = gr.Textbox(label="Manual Camera ID Input", value="0")
                
                btn_reset = gr.Button("Reset Tracker")
                btn_capture = gr.Button("Capture Query", variant="primary")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Bottom Section - Capture Information")
                with gr.Row():
                    capture_preview = gr.Image(label="Query Crop Preview", width=250, height=250)
                    with gr.Column():
                        capture_source = gr.Label(label="Capture Source", value="None")
                        capture_cam_id = gr.Label(label="Camera ID", value="0")
                        frame_quality = gr.Label(label="Optional Frame Quality", value="N/A")

    # -------------------------------------------------------------
    # Tab 2: Gallery Search / CA-Jaccard Demo
    # -------------------------------------------------------------
    with gr.Tab("Gallery Search / CA-Jaccard Demo"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Query Input & Management")
                query_upload = gr.Image(label="Query Image")
                dataset_selector = gr.Radio(
                    choices=["Market-1501 (Default)", "Custom"], 
                    value="Market-1501 (Default)", 
                    label="Dataset Selector"
                )
                query_camera_id = gr.Textbox(label="Query Camera ID", value="0")
                
            with gr.Column(scale=1):
                gr.Markdown("#### Search & Re-ranking Controls")
                use_caj_ranking = gr.Checkbox(label="Use CA-Jaccard Re-ranking", value=True)
                top_k_selector = gr.Dropdown(choices=[5, 10], value=5, label="Top-K Results")
                
                btn_search = gr.Button("Search Gallery", variant="primary")
                btn_clear = gr.Button("Clear Result")
                
        with gr.Column():
            gr.Markdown("#### Results Panel")
            baseline_gallery = gr.Gallery(label="Baseline Top-K Results (Cosine/Euclidean Similarity)")
            final_gallery = gr.Gallery(label="Final Top-K Results (Adjusted Rank Matrix)")

    # -------------------------------------------------------------
    # Tab 3: Four-Mode Top-5 Comparison
    # -------------------------------------------------------------
    with gr.Tab("Four-Mode Top-5 Comparison"):
        gr.Markdown("#### Query Matrix Setup")
        with gr.Row():
            raw_query_img = gr.Image(label="Raw Query Image Component (Red Box Crop)")
            kalman_query_img = gr.Image(label="Kalman-smoothed Query Image Component (Green Box Crop)")
            
        btn_run_compare = gr.Button("Run 4-Mode Comparison", variant="primary")
        
        gr.Markdown("#### 4-Column/Grid Comparison Engine (Top-5 Results)")
        with gr.Row():
            mode1_out = gr.Gallery(label="Mode 1: Kalman OFF | CA-Jaccard OFF")
            mode2_out = gr.Gallery(label="Mode 2: Kalman ON  | CA-Jaccard OFF")
        with gr.Row():
            mode3_out = gr.Gallery(label="Mode 3: Kalman OFF | CA-Jaccard ON")
            mode4_out = gr.Gallery(label="Mode 4: Kalman ON  | CA-Jaccard ON")

    # Connect event handlers
    use_kalman.change(fn=update_kalman_state, inputs=[use_kalman], outputs=[])
    btn_toggle_cam.click(
        fn=toggle_camera,
        inputs=[use_kalman],
        outputs=[webcam_placeholder, btn_toggle_cam, det_status, kf_mode],
        concurrency_limit=None,
        trigger_mode="multiple"
    )
    btn_reset.click(fn=reset_tracker, inputs=[], outputs=[det_status, kf_mode])
    btn_capture.click(fn=capture_query, inputs=[use_kalman, cam_id], outputs=[capture_preview, capture_source, capture_cam_id, frame_quality])
    btn_search.click(fn=search_gallery, inputs=[query_upload, use_caj_ranking, top_k_selector], outputs=[baseline_gallery, final_gallery])
    btn_run_compare.click(fn=run_comparison, inputs=[raw_query_img, kalman_query_img], outputs=[mode1_out, mode2_out, mode3_out, mode4_out])

if __name__ == "__main__":
    demo.launch()
