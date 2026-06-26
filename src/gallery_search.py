import os
import cv2
import numpy as np
from reid_engine import get_reid_model
from webcam_handler import capture_live_fallback, capture_live_both
import webcam_handler


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, ".cache")
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Global cached gallery data
cached_gallery_data = None

def search_gallery(query_img, use_caj, top_k, query_cam_id="0"):
    global cached_gallery_data
    print(f"\n==> search_gallery called! query_img_type={type(query_img)}, use_caj={use_caj}, top_k={top_k}, query_cam_id={query_cam_id}")
    
    # Fallback to live webcam crop if no query image is uploaded/snapshotted
    if query_img is None:
        print("==> search_gallery: query_img is None. Checking live webcam...")
        crop_rgb, crop_source = capture_live_fallback(use_kf=webcam_handler.use_kalman_global)
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
            cache_path = os.path.join(CACHE_DIR, 'market1501_gallery_features.npy')
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
