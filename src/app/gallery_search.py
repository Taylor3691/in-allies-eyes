import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T_vision

from .reid_engine import get_reid_model
from .webcam_handler import capture_live_fallback, capture_live_both
from . import webcam_handler


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(REPO_ROOT, ".cache")
DATA_DIR = os.path.join(REPO_ROOT, "data")

BORDER_RATIO = 0.04

# Global cached gallery data dict mapping dataset name to its cached data dict
cached_gallery_data = {}


def build_and_cache_gallery(cache_path, model, device, dataset_name='market1501'):
    print(f"==> Cache not found at {cache_path}. Extracting gallery features dynamically for {dataset_name}...")
    from ..caj import datasets
    from ..caj.utils.data.preprocessor import Preprocessor

    dataset_root = os.path.join(DATA_DIR, dataset_name)

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"{dataset_name} dataset not found at {dataset_root}. Please run download_datasets.py first.")

    dataset = datasets.create(dataset_name, dataset_root)

    normalizer = T_vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformer = T_vision.Compose([
        T_vision.Resize((256, 128), interpolation=T_vision.InterpolationMode.BICUBIC),
        T_vision.ToTensor(),
        normalizer
    ])

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=dataset.images_dir, transform=transformer),
        batch_size=256, num_workers=4,
        shuffle=False, pin_memory=True
    )

    gallery_features = []
    gallery_paths = []
    gallery_pids = []
    gallery_camids = []

    with torch.no_grad():
        for i, (imgs, fnames, pids, camids, _) in enumerate(gallery_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            gallery_features.append(outputs.cpu().numpy())
            gallery_paths.extend(fnames)
            gallery_pids.extend(pids.numpy())
            gallery_camids.extend(camids.numpy())

            if (i + 1) % 10 == 0:
                print(f"Extraction progress: [{i+1}/{len(gallery_loader)}]")

    gallery_features = np.concatenate(gallery_features, axis=0)

    data_dict = {
        'features': gallery_features,
        'image_paths': np.array(gallery_paths),
        'pids': np.array(gallery_pids),
        'camids': np.array(gallery_camids)
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, **data_dict)
    print(f"==> Successfully cached {len(gallery_paths)} gallery embeddings to {cache_path}")
    return data_dict


def search_gallery(query_img, use_caj, top_k, query_cam_id=1, dataset_name="Market-1501 (Default)"):
    global cached_gallery_data
    print(f"\n==> search_gallery called! query_img_type={type(query_img)}, use_caj={use_caj}, top_k={top_k}, query_cam_id={query_cam_id}, dataset_name={dataset_name}")

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

        # Shift camera ID for Market-1501/Custom
        query_cam = int(query_cam_id)
        if dataset_name != "CUHK03":
            query_cam = query_cam - 1

        # 2. Extract query feature
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
        cache_filename = 'cuhk03_gallery_features.npz' if dataset_name == "CUHK03" else 'market1501_gallery_features.npz'
        internal_dataset_name = 'cuhk03' if dataset_name == "CUHK03" else 'market1501'
        cache_path = os.path.join(CACHE_DIR, cache_filename)

        if dataset_name not in cached_gallery_data:
            if not os.path.exists(cache_path):
                cached_gallery_data[dataset_name] = build_and_cache_gallery(cache_path, model, device, internal_dataset_name)
            else:
                with np.load(cache_path) as data:
                    cached_gallery_data[dataset_name] = {
                        'features': data['features'],
                        'image_paths': data['image_paths'],
                        'pids': data['pids'],
                        'camids': data['camids']
                    }

        g_data = cached_gallery_data[dataset_name]
        g_features = g_data['features']
        g_paths = g_data['image_paths']
        g_camids = g_data['camids']

        # 4. Compute Cosine similarity & Baseline Ranking
        q_f = query_feat / np.linalg.norm(query_feat, axis=1, keepdims=True)
        g_f = g_features / np.linalg.norm(g_features, axis=1, keepdims=True)

        dist_matrix = 1.0 - np.dot(q_f, g_f.T)
        dist = dist_matrix[0]

        initial_indices = np.argsort(dist)
        top_200_indices = initial_indices[:200]

        # 5. Localized CA-Jaccard Re-ranking
        if use_caj:
            from ..caj.utils.rerank import re_ranking

            # Setup inputs for re_ranking
            q_g_dist = dist[top_200_indices].reshape(1, 200)
            q_q_dist = np.zeros((1, 1))

            top_200_features = g_f[top_200_indices]
            g_g_dist = 1.0 - np.dot(top_200_features, top_200_features.T)

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
                disp_cam_id = int(cam_id) + (1 if dataset_name != "CUHK03" else 0)
                if int(cam_id) == query_cam:
                    # Yellow border relative to image size
                    border_w = max(1, int(img.shape[1] * BORDER_RATIO))
                    img = cv2.copyMakeBorder(img, border_w, border_w, border_w, border_w, cv2.BORDER_CONSTANT, value=[255, 220, 0])
                    caption = f"Rank {rank_idx+1} | Cam {disp_cam_id} (Same Cam)"
                else:
                    caption = f"Rank {rank_idx+1} | Cam {disp_cam_id}"
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
                disp_cam_id = int(cam_id) + (1 if dataset_name != "CUHK03" else 0)
                improved = use_caj and (rank_idx < baseline_pos)
                is_same_cam = int(cam_id) == query_cam

                if improved or is_same_cam:
                    border_w = max(1, int(img.shape[1] * BORDER_RATIO))
                    if improved:
                        # Green border relative to image size
                        img = cv2.copyMakeBorder(img, border_w, border_w, border_w, border_w, cv2.BORDER_CONSTANT, value=[46, 204, 113])
                        caption = f"Rank {rank_idx+1} | Cam {disp_cam_id} (Improved from {baseline_pos+1}!)"
                    else:
                        # Yellow border relative to image size
                        img = cv2.copyMakeBorder(img, border_w, border_w, border_w, border_w, cv2.BORDER_CONSTANT, value=[255, 220, 0])
                        caption = f"Rank {rank_idx+1} | Cam {disp_cam_id} (Same Cam)"
                else:
                    caption = f"Rank {rank_idx+1} | Cam {disp_cam_id}"

                final_results.append((img, caption))
            else:
                final_results.append((np.zeros((128, 64, 3), dtype=np.uint8), f"Missing: {os.path.basename(img_path)}"))

        import gradio as gr
        return (
            gr.update(value=baseline_results, selected_index=0),
            gr.update(value=final_results, selected_index=0),
            gr.update(value=query_img)
        )

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

    import gradio as gr
    return (
        gr.update(value=m1_out['value'] if isinstance(m1_out, dict) else m1_out, selected_index=0),
        gr.update(value=m2_out['value'] if isinstance(m2_out, dict) else m2_out, selected_index=0),
        gr.update(value=m3_out['value'] if isinstance(m3_out, dict) else m3_out, selected_index=0),
        gr.update(value=m4_out['value'] if isinstance(m4_out, dict) else m4_out, selected_index=0),
        gr.update(value=raw_img),
        gr.update(value=kalman_img)
    )
