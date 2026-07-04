import os
import cv2
import numpy as np
import time

from ..kalman_filter import KalmanTracker


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(REPO_ROOT, ".cache")

# OpenCV face detector cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Shared state variables
latest_frame = None
camera_active = False
use_kalman_global = False
missing_frames_counter = 0
tracker = KalmanTracker()


def update_kalman_state(val):
    global use_kalman_global
    use_kalman_global = val

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
    try:
        camera_id_int = int(camera_id)
        if camera_id_int < 1:
            camera_id_int = 1
    except (ValueError, TypeError):
        camera_id_int = 1

    if latest_frame is None:
        return None, "No active frame captured", "N/A", "N/A", None, camera_id_int, None, None

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
        return None, "No target found to crop", "N/A", "N/A", None, camera_id_int, None, None

    quality = f"Contrast: {int(np.std(crop_rgb))}"

    # Save to disk
    save_dir = (
        os.path.join(CACHE_DIR, "captured", "kalman")
        if (use_kf and source.startswith("Kalman"))
        else os.path.join(CACHE_DIR, "captured", "original")
    )
    os.makedirs(save_dir, exist_ok=True)
    filename = f"crop_{int(time.time())}.png"
    cv2.imwrite(os.path.join(save_dir, filename), crop_bgr)

    return crop_rgb, source, camera_id, quality, crop_rgb, camera_id_int, raw_crop_rgb, kalman_crop_rgb

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
