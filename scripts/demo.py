import sys
import os

import gradio as gr

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.app.webcam_handler import toggle_camera, capture_query, reset_tracker, update_kalman_state
from src.app.gallery_search import search_gallery, run_comparison

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
                    choices=["Market1501", "CUHK03", "Custom"],
                    value="Market1501",
                    label="Dataset Selector"
                )
                query_camera_id = gr.Number(label="Query Camera ID", value=1, minimum=1, maximum=6, precision=0)

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
    btn_capture.click(
        fn=capture_query,
        inputs=[use_kalman, cam_id],
        outputs=[
            capture_preview, capture_source, capture_cam_id, frame_quality,
            query_upload, query_camera_id,
            raw_query_img, kalman_query_img
        ]
    )

    dataset_selector.change(
        fn=(lambda dataset:
            gr.update(minimum=1, maximum=2, value=1)
            if dataset == "CUHK03"
            else gr.update(minimum=1, maximum=6, value=1)
        ),
        inputs=[dataset_selector],
        outputs=[query_camera_id]
    )

    btn_search.click(
        fn=search_gallery,
        inputs=[query_upload, use_caj_ranking, top_k_selector, query_camera_id, dataset_selector],
        outputs=[baseline_gallery, final_gallery, query_upload],
        show_progress="full",
    )
    btn_clear.click(
        fn=lambda: [None, None],
        outputs=[baseline_gallery, final_gallery],
    )
    btn_run_compare.click(
        fn=run_comparison,
        inputs=[raw_query_img, kalman_query_img],
        outputs=[mode1_out, mode2_out, mode3_out, mode4_out, raw_query_img, kalman_query_img],
        show_progress="full",
    )

if __name__ == "__main__":
    demo.launch()
