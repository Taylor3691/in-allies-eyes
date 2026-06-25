import sys
import os
import gradio as gr
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from Kalman_filter import KalmanTracker

# Stub callbacks for initial visual testing
def search_gallery(query_img, use_caj, top_k):
    # Placeholder: Returns an empty list of images for baseline and final results
    return [], []

def run_comparison(raw_img, kalman_img):
    # Placeholder: Returns empty lists for the 4 comparison modes
    return [], [], [], []

def reset_tracker():
    return "Missing", "Predict only"

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
                # Placeholder image representing a live camera feed
                webcam_placeholder = gr.Image(label="Live Video Feed (Webcam Stream Placeholder)")
                
            with gr.Column(scale=1):
                gr.Markdown("#### Bounding Box & Tracking Controls")
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

    # Connect stub event handlers
    btn_reset.click(fn=reset_tracker, inputs=[], outputs=[det_status, kf_mode])
    btn_search.click(fn=search_gallery, inputs=[query_upload, use_caj_ranking, top_k_selector], outputs=[baseline_gallery, final_gallery])
    btn_run_compare.click(fn=run_comparison, inputs=[raw_query_img, kalman_query_img], outputs=[mode1_out, mode2_out, mode3_out, mode4_out])

if __name__ == "__main__":
    demo.launch()
