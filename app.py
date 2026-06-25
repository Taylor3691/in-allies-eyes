import sys
import os
import cv2
import gradio as gr
import numpy as np
import time
import threading

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

# Global Re-ID Model variables for lazy loading
reid_model = None
reid_model_lock = threading.Lock()

def get_reid_model():
    global reid_model
    if reid_model is None:
        with reid_model_lock:
            if reid_model is None:
                print("==> Initializing Re-ID model lazily...")
                checkpoint_path = os.path.join(os.path.dirname(__file__), 'pretrained_models', 'market_resnet50_model_120_rank1_945.pth')
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please run scripts/download_pretrained_models.py demo.")

                import torch
                import torch.nn as nn
                import torch.nn.functional as F
                from caj.utils.serialization import load_checkpoint, copy_state_dict

                class NormalizedFeatureModel(nn.Module):
                    def __init__(self, model, normalize=True):
                        super(NormalizedFeatureModel, self).__init__()
                        self.model = model
                        self.normalize = normalize

                    def forward(self, inputs):
                        outputs = self.model(inputs)
                        if isinstance(outputs, (tuple, list)):
                            outputs = outputs[0]
                        if self.normalize:
                            outputs = F.normalize(outputs, dim=1, p=2)
                        return outputs

                bot_root = os.path.join(os.path.dirname(__file__), 'src', 'thirdparty', 'bot')
                if bot_root not in sys.path:
                    sys.path.insert(0, bot_root)
                from modeling.baseline import Baseline

                # Market-1501 dataset has 751 train pids
                num_classes = 751
                raw_model = Baseline(
                    num_classes=num_classes,
                    last_stride=1,
                    model_path='',
                    neck='bnneck',
                    neck_feat='after',
                    model_name='resnet50',
                    pretrain_choice='none',
                )
                model = NormalizedFeatureModel(raw_model, normalize=True)

                checkpoint = load_checkpoint(checkpoint_path)
                raw_state_dict = checkpoint['state_dict']

                from collections import OrderedDict
                state_dict = OrderedDict()
                layer_map = {
                    'base.conv1.': 'base.0.',
                    'base.bn1.': 'base.1.',
                    'base.layer1.': 'base.4.',
                    'base.layer2.': 'base.5.',
                    'base.layer3.': 'base.6.',
                    'base.layer4.': 'base.7.',
                    'bottleneck.': 'feat_bn.',
                }
                for name, param in raw_state_dict.items():
                    if name.startswith('classifier.'):
                        continue
                    new_name = name
                    for old_prefix, new_prefix in layer_map.items():
                        if name.startswith(old_prefix):
                            new_name = new_prefix + name[len(old_prefix):]
                            break
                    state_dict[new_name] = param

                copy_state_dict(state_dict, model.model, strip='module.')

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                model.eval()

                reid_model = (model, device)
                print("==> Re-ID model loaded successfully.")

    return reid_model

def update_kalman_state(val):
    global use_kalman_global
    use_kalman_global = val

# Global cached gallery data
cached_gallery_data = None

def search_gallery(query_img, use_caj, top_k, query_cam_id="0"):
    global cached_gallery_data
    if query_img is None:
        return [], []

    try:
        # 1. Lazy load model
        model, device = get_reid_model()
        
        # 2. Extract query feature
        import torch
        from PIL import Image
        import torchvision.transforms as T_vision
        
        pil_img = Image.fromarray(query_img)
        normalizer = T_vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformer = T_vision.Compose([
            T_vision.Resize((256, 128), interpolation=T_vision.InterpolationMode.BICUBIC),
            T_vision.ToTensor(),
            normalizer
        ])
        img_tensor = transformer(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            query_feat = model(img_tensor)
            if isinstance(query_feat, (tuple, list)):
                query_feat = query_feat[0]
            query_feat = query_feat.cpu().numpy() # shape (1, D)

        # 3. Lazy load gallery features cache
        if cached_gallery_data is None:
            cache_path = os.path.join(os.path.dirname(__file__), 'market1501_gallery_features.npy')
            if not os.path.exists(cache_path):
                return [("https://via.placeholder.com/150?text=Run+Cache+Script+First", "Cache not found")], []
            cached_gallery_data = np.load(cache_path, allow_pickle=True).item()

        g_features = cached_gallery_data['features']
        g_paths = cached_gallery_data['image_paths']
        g_camids = cached_gallery_data['camids']

        # 4. Compute Cosine similarity & Baseline Ranking
        q_f = query_feat / np.linalg.norm(query_feat, axis=1, keepdims=True)
        g_f = g_features / np.linalg.norm(g_features, axis=1, keepdims=True)
        
        dist_matrix = 1.0 - np.dot(q_f, g_f.T)
        dist = dist_matrix[0]
        
        initial_indices = np.argsort(dist)
        top_200_indices = initial_indices[:200]
        
        # 5. Localized CA-Jaccard Re-ranking
        if use_caj:
            from caj.utils.rerank import re_ranking
            
            # Setup inputs for re_ranking
            q_g_dist = dist[top_200_indices].reshape(1, 200)
            q_q_dist = np.zeros((1, 1))
            
            top_200_features = g_f[top_200_indices]
            g_g_dist = 1.0 - np.dot(top_200_features, top_200_features.T)
            
            query_cam = int(query_cam_id)
            gallery_cams = g_camids[top_200_indices]
            cids = np.concatenate([np.array([query_cam]), gallery_cams])
            
            # Setup dummy args
            class CAJArgs:
                def __init__(self):
                    self.k1 = 20
                    self.k2 = 6
                    self.ckrnns = True
                    self.k1_intra = 5
                    self.k1_inter = 20
                    self.clqe = True
                    self.k2_intra = 2
                    self.k2_inter = 4
                    
            args = CAJArgs()
            final_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, cids, args)
            
            reranked_sub_indices = np.argsort(final_dist[0])
            final_top_indices = top_200_indices[reranked_sub_indices]
        else:
            final_top_indices = initial_indices

        # 6. Build Baseline Results
        baseline_results = []
        for rank_idx, idx in enumerate(initial_indices[:top_k]):
            img_path = g_paths[idx]
            cam_id = g_camids[idx]
            
            # Load and convert to RGB
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Same-camera highlight
                if int(cam_id) == int(query_cam_id):
                    # Yellow border
                    img = cv2.copyMakeBorder(img, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=[255, 220, 0])
                    caption = f"Rank {rank_idx+1} | Cam {cam_id} (Same Cam)"
                else:
                    caption = f"Rank {rank_idx+1} | Cam {cam_id}"
                baseline_results.append((img, caption))
            else:
                baseline_results.append((np.zeros((128, 64, 3), dtype=np.uint8), f"Missing: {os.path.basename(img_path)}"))

        # 7. Build Final / Re-ranked Results
        final_results = []
        for rank_idx, idx in enumerate(final_top_indices[:top_k]):
            img_path = g_paths[idx]
            cam_id = g_camids[idx]
            
            # Find baseline position
            baseline_pos = np.where(initial_indices == idx)[0][0]
            
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Check for rank improvement
                improved = use_caj and (rank_idx < baseline_pos)
                is_same_cam = int(cam_id) == int(query_cam_id)
                
                if improved:
                    # Green border
                    img = cv2.copyMakeBorder(img, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=[46, 204, 113])
                    caption = f"Rank {rank_idx+1} | Cam {cam_id} (Improved from {baseline_pos+1}!)"
                elif is_same_cam:
                    # Yellow border
                    img = cv2.copyMakeBorder(img, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=[255, 220, 0])
                    caption = f"Rank {rank_idx+1} | Cam {cam_id} (Same Cam)"
                else:
                    caption = f"Rank {rank_idx+1} | Cam {cam_id}"
                
                final_results.append((img, caption))
            else:
                final_results.append((np.zeros((128, 64, 3), dtype=np.uint8), f"Missing: {os.path.basename(img_path)}"))

        return baseline_results, final_results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return [("https://via.placeholder.com/150", f"Error: {e}")], []

def run_comparison(raw_img, kalman_img):
    if raw_img is None or kalman_img is None:
        return [], [], [], []
    
    # Mode 1 & 3 use raw query, Mode 2 & 4 use Kalman-smoothed query
    m1_out, _ = search_gallery(raw_img, use_caj=False, top_k=5, query_cam_id="0")
    m2_out, _ = search_gallery(kalman_img, use_caj=False, top_k=5, query_cam_id="0")
    _, m3_out = search_gallery(raw_img, use_caj=True, top_k=5, query_cam_id="0")
    _, m4_out = search_gallery(kalman_img, use_caj=True, top_k=5, query_cam_id="0")
    
    return m1_out, m2_out, m3_out, m4_out

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

    # Lazy-load model and run feature extraction to verify pipeline works
    try:
        model, device = get_reid_model()
        import torch
        from PIL import Image
        import torchvision.transforms as T_vision
        pil_img = Image.fromarray(crop_rgb)
        normalizer = T_vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformer = T_vision.Compose([
            T_vision.Resize((256, 128), interpolation=T_vision.InterpolationMode.BICUBIC),
            T_vision.ToTensor(),
            normalizer
        ])
        img_tensor = transformer(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            _ = feat.cpu().numpy()
        print("==> Feature extracted successfully for captured query.")
    except Exception as e:
        print(f"Error during on-demand feature extraction: {e}")

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
    btn_search.click(fn=search_gallery, inputs=[query_upload, use_caj_ranking, top_k_selector, query_camera_id], outputs=[baseline_gallery, final_gallery], show_progress="full")
    btn_run_compare.click(fn=run_comparison, inputs=[raw_query_img, kalman_query_img], outputs=[mode1_out, mode2_out, mode3_out, mode4_out], show_progress="full")

if __name__ == "__main__":
    demo.launch()
