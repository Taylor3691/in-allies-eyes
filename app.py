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
_model_loading_thread = None
_model_loading_error = None

def _bg_load_model():
    global reid_model, _model_loading_error
    try:
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

        checkpoint_path = os.path.join(os.path.dirname(__file__), 'pretrained_models', 'market_resnet50_model_120_rank1_945.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please run scripts/download_pretrained_models.py demo.")

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
        print("==> Re-ID model loaded successfully in background thread.")
    except Exception as e:
        _model_loading_error = e
        import traceback
        traceback.print_exc()

def get_reid_model():
    global reid_model, _model_loading_thread, _model_loading_error
    if reid_model is not None:
        return reid_model

    with reid_model_lock:
        if reid_model is None:
            if _model_loading_thread is None:
                print("==> Starting Re-ID model lazy load in background thread...")
                _model_loading_error = None
                _model_loading_thread = threading.Thread(target=_bg_load_model)
                _model_loading_thread.start()

            # Yield control periodically to allow event loops (Uvicorn heartbeat, OpenCV webcam stream) to run
            while _model_loading_thread.is_alive():
                time.sleep(0.05)

            _model_loading_thread = None
            if _model_loading_error is not None:
                raise _model_loading_error

    return reid_model

def update_kalman_state(val):
    global use_kalman_global
    use_kalman_global = val

# Global cached gallery data
cached_gallery_data = None

def capture_live_fallback(use_kf=False):
    global latest_frame, camera_active, tracker
    
    # Case 1: Camera is already running
    if camera_active and latest_frame is not None:
        latest_bgr = cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(latest_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        raw_bbox = None
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            raw_bbox = faces[0]

        crop_bgr = None
        if use_kf:
            kf_rect = tracker.get_rect()
            if kf_rect is not None:
                kx, ky, kw, kh = kf_rect
                fh, fw_img, _ = latest_bgr.shape
                kx = max(0, min(kx, fw_img))
                ky = max(0, min(ky, fh))
                kw = max(1, min(kw, fw_img - kx))
                kh = max(1, min(kh, fh - ky))
                crop_bgr = latest_bgr[ky:ky+kh, kx:kx+kw]
            elif raw_bbox is not None:
                rx, ry, rw, rh = raw_bbox
                fh, fw_img, _ = latest_bgr.shape
                rx = max(0, min(rx, fw_img))
                ry = max(0, min(ry, fh))
                rw = max(1, min(rw, fw_img - rx))
                rh = max(1, min(rh, fh - ry))
                crop_bgr = latest_bgr[ry:ry+rh, rx:rx+rw]
        else:
            if raw_bbox is not None:
                rx, ry, rw, rh = raw_bbox
                fh, fw_img, _ = latest_bgr.shape
                rx = max(0, min(rx, fw_img))
                ry = max(0, min(ry, fh))
                rw = max(1, min(rw, fw_img - rx))
                rh = max(1, min(rh, fh - ry))
                crop_bgr = latest_bgr[ry:ry+rh, rx:rx+rw]
                
        if crop_bgr is not None and crop_bgr.size > 0:
            return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), "Active Webcam"
        return None, "Active Webcam (No face detected)"

    # Case 2: Camera is not running, open it temporarily
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "Webcam Offline"
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    crop_bgr = None
    crop_source = "None"
    
    # Read up to 15 frames to let auto-exposure adjust and look for a face
    for _ in range(15):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue
            
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            rx, ry, rw, rh = faces[0]
            
            fh, fw_img, _ = frame.shape
            rx = max(0, min(rx, fw_img))
            ry = max(0, min(ry, fh))
            rw = max(1, min(rw, fw_img - rx))
            rh = max(1, min(rh, fh - ry))
            crop_bgr = frame[ry:ry+rh, rx:rx+rw]
            crop_source = "Raw bbox (webcam capture)"
            break
            
        time.sleep(0.03)
        
    # If no face was detected in any of the 15 frames, just capture the full center of the last frame as fallback
    if crop_bgr is None:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            fh, fw_img, _ = frame.shape
            cx, cy = fw_img // 2, fh // 2
            size = min(fh, fw_img) // 2
            crop_bgr = frame[cy-size:cy+size, cx-size:cx+size]
            crop_source = "Center crop fallback"
            
    cap.release()
    
    if crop_bgr is not None and crop_bgr.size > 0:
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), crop_source
    return None, "Failed to capture frame"

def capture_live_both():
    global latest_frame, camera_active, tracker
    
    # Case 1: Camera is already running
    if camera_active and latest_frame is not None:
        latest_bgr = cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(latest_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        raw_bbox = None
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            raw_bbox = faces[0]

        raw_crop_rgb = None
        if raw_bbox is not None:
            rx, ry, rw, rh = raw_bbox
            fh, fw_img, _ = latest_bgr.shape
            rx = max(0, min(rx, fw_img))
            ry = max(0, min(ry, fh))
            rw = max(1, min(rw, fw_img - rx))
            rh = max(1, min(rh, fh - ry))
            crop_bgr = latest_bgr[ry:ry+rh, rx:rx+rw]
            if crop_bgr.size > 0:
                raw_crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        kalman_crop_rgb = None
        kf_rect = tracker.get_rect()
        if kf_rect is not None:
            kx, ky, kw, kh = kf_rect
            fh, fw_img, _ = latest_bgr.shape
            kx = max(0, min(kx, fw_img))
            ky = max(0, min(ky, fh))
            kw = max(1, min(kw, fw_img - kx))
            kh = max(1, min(kh, fh - ky))
            crop_bgr = latest_bgr[ky:ky+kh, kx:kx+kw]
            if crop_bgr.size > 0:
                kalman_crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                
        return raw_crop_rgb, kalman_crop_rgb, "Active Webcam"

    # Case 2: Camera is not running, open it temporarily
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, None, "Webcam Offline"
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    raw_crop_rgb = None
    kalman_crop_rgb = None
    crop_source = "None"
    
    for _ in range(15):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue
            
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            rx, ry, rw, rh = faces[0]
            
            fh, fw_img, _ = frame.shape
            rx = max(0, min(rx, fw_img))
            ry = max(0, min(ry, fh))
            rw = max(1, min(rw, fw_img - rx))
            rh = max(1, min(rh, fh - ry))
            crop_bgr = frame[ry:ry+rh, rx:rx+rw]
            if crop_bgr.size > 0:
                raw_crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                kalman_crop_rgb = raw_crop_rgb
                crop_source = "Raw bbox (webcam capture)"
            break
            
        time.sleep(0.03)
        
    if raw_crop_rgb is None:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            fh, fw_img, _ = frame.shape
            cx, cy = fw_img // 2, fh // 2
            size = min(fh, fw_img) // 2
            crop_bgr = frame[cy-size:cy+size, cx-size:cx+size]
            if crop_bgr.size > 0:
                raw_crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                kalman_crop_rgb = raw_crop_rgb
                crop_source = "Center crop fallback"
            
    cap.release()
    return raw_crop_rgb, kalman_crop_rgb, crop_source

def search_gallery(query_img, use_caj, top_k, query_cam_id="0"):
    global cached_gallery_data, use_kalman_global
    print(f"\n==> search_gallery called! query_img_type={type(query_img)}, use_caj={use_caj}, top_k={top_k}, query_cam_id={query_cam_id}")
    
    # Fallback to live webcam crop if no query image is uploaded/snapshotted
    if query_img is None:
        print("==> search_gallery: query_img is None. Checking live webcam...")
        crop_rgb, crop_source = capture_live_fallback(use_kf=use_kalman_global)
        if crop_rgb is not None:
            query_img = crop_rgb
            print(f"==> search_gallery: successfully captured crop from {crop_source}!")
        else:
            print(f"==> search_gallery: live fallback failed: {crop_source}")

    if query_img is None:
        print("==> search_gallery: query_img is None!")
        return [], [], None

    try:
        # 1. Lazy load model
        model, device = get_reid_model()
        print("==> get_reid_model returned successfully.")
        
        # 2. Extract query feature
        import torch
        from PIL import Image
        import torchvision.transforms as T_vision
        
        if isinstance(query_img, str):
            pil_img = Image.open(query_img).convert('RGB')
        elif isinstance(query_img, np.ndarray):
            pil_img = Image.fromarray(query_img)
        elif isinstance(query_img, dict):
            img_val = query_img.get("composite", query_img.get("background", None))
            if img_val is None:
                return [], [], None
            if isinstance(img_val, str):
                pil_img = Image.open(img_val).convert('RGB')
            elif isinstance(img_val, np.ndarray):
                pil_img = Image.fromarray(img_val)
            else:
                pil_img = img_val
        elif isinstance(query_img, Image.Image):
            pil_img = query_img
        else:
            pil_img = Image.fromarray(np.uint8(query_img))

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
                return [("https://via.placeholder.com/150?text=Run+Cache+Script+First", "Cache not found")], [], None
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

        return baseline_results, final_results, query_img

    except Exception as e:
        import traceback
        traceback.print_exc()
        return [("https://via.placeholder.com/150", f"Error: {e}")], [], None

def run_comparison(raw_img, kalman_img):
    # Fallback to live webcam crops if query images are missing
    if raw_img is None or kalman_img is None:
        print("==> run_comparison: Query images are None. Checking live webcam...")
        raw_crop, kalman_crop, crop_source = capture_live_both()
        if raw_crop is not None:
            raw_img = raw_crop
        if kalman_crop is not None:
            kalman_img = kalman_crop
        print(f"==> run_comparison: capture result: {crop_source}")

    if raw_img is None or kalman_img is None:
        print("==> run_comparison: Query images are still None!")
        return [], [], [], [], raw_img, kalman_img
    
    # Mode 1 & 3 use raw query, Mode 2 & 4 use Kalman-smoothed query
    m1_out, _, _ = search_gallery(raw_img, use_caj=False, top_k=5, query_cam_id="0")
    m2_out, _, _ = search_gallery(kalman_img, use_caj=False, top_k=5, query_cam_id="0")
    _, m3_out, _ = search_gallery(raw_img, use_caj=True, top_k=5, query_cam_id="0")
    _, m4_out, _ = search_gallery(kalman_img, use_caj=True, top_k=5, query_cam_id="0")
    
    return m1_out, m2_out, m3_out, m4_out, raw_img, kalman_img

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
        return None, "No active frame captured", "N/A", "N/A", None, camera_id, None, None

    # Convert RGB frame back to BGR for OpenCV processing
    latest_bgr = cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(latest_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    raw_bbox = None

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        raw_bbox = faces[0]

    # Extract raw crop
    raw_crop_rgb = None
    raw_crop_bgr = None
    if raw_bbox is not None:
        rx, ry, rw, rh = raw_bbox
        fh, fw_img, _ = latest_bgr.shape
        rx = max(0, min(rx, fw_img))
        ry = max(0, min(ry, fh))
        rw = max(1, min(rw, fw_img - rx))
        rh = max(1, min(rh, fh - ry))
        raw_crop_bgr = latest_bgr[ry:ry+rh, rx:rx+rw]
        if raw_crop_bgr.size > 0:
            raw_crop_rgb = cv2.cvtColor(raw_crop_bgr, cv2.COLOR_BGR2RGB)

    # Extract Kalman crop
    kalman_crop_rgb = None
    kalman_crop_bgr = None
    kf_rect = tracker.get_rect()
    if kf_rect is not None:
        kx, ky, kw, kh = kf_rect
        fh, fw_img, _ = latest_bgr.shape
        kx = max(0, min(kx, fw_img))
        ky = max(0, min(ky, fh))
        kw = max(1, min(kw, fw_img - kx))
        kh = max(1, min(kh, fh - ky))
        kalman_crop_bgr = latest_bgr[ky:ky+kh, kx:kx+kw]
        if kalman_crop_bgr.size > 0:
            kalman_crop_rgb = cv2.cvtColor(kalman_crop_bgr, cv2.COLOR_BGR2RGB)

    # Determine preview and active crop
    crop_rgb = None
    crop_bgr = None
    source = "None"
    if use_kf:
        if kalman_crop_rgb is not None:
            crop_rgb = kalman_crop_rgb
            crop_bgr = kalman_crop_bgr
            source = "Kalman bbox"
        elif raw_crop_rgb is not None:
            crop_rgb = raw_crop_rgb
            crop_bgr = raw_crop_bgr
            source = "Raw bbox (Kalman N/A)"
    else:
        if raw_crop_rgb is not None:
            crop_rgb = raw_crop_rgb
            crop_bgr = raw_crop_bgr
            source = "Raw bbox"

    if crop_rgb is None:
        return None, "No target found to crop", "N/A", "N/A", None, camera_id, None, None

    quality = f"Contrast: {int(np.std(crop_rgb))}"

    # Save to disk
    save_dir = "captured/kalman" if (use_kf and source.startswith("Kalman")) else "captured/original"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"crop_{int(time.time())}.png"
    cv2.imwrite(os.path.join(save_dir, filename), crop_bgr)

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

    return crop_rgb, source, camera_id, quality, crop_rgb, camera_id, raw_crop_rgb, kalman_crop_rgb

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
    btn_capture.click(
        fn=capture_query,
        inputs=[use_kalman, cam_id],
        outputs=[
            capture_preview, capture_source, capture_cam_id, frame_quality,
            query_upload, query_camera_id,
            raw_query_img, kalman_query_img
        ]
    )
    btn_search.click(fn=search_gallery, inputs=[query_upload, use_caj_ranking, top_k_selector, query_camera_id], outputs=[baseline_gallery, final_gallery, query_upload], show_progress="full")
    btn_run_compare.click(fn=run_comparison, inputs=[raw_query_img, kalman_query_img], outputs=[mode1_out, mode2_out, mode3_out, mode4_out, raw_query_img, kalman_query_img], show_progress="full")

if __name__ == "__main__":
    demo.launch()
