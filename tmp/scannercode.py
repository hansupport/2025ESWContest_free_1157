# barcodecheck.py — show BARCODE info only for ~3 seconds after detection
import os
import cv2
import json
import re
import time
import numpy as np

# ==== SETTINGS ====
DISPLAY_SECONDS = 3.0           # show overlay for N seconds after a detection
DEFAULT_CAM_INDEX = 2           # your confirmed RGB node (e.g., /dev/video2)
PREFERRED_RES = (640, 480)      # stable on Jetson Nano
# ===================

# Try OpenCV BarcodeDetector (opencv-contrib) first
_HAS_OCV_BARCODE = hasattr(cv2, "barcode_BarcodeDetector")
_BARCODE_DET = cv2.barcode_BarcodeDetector() if _HAS_OCV_BARCODE else None

# Try pyzbar as a fallback
try:
    from pyzbar import pyzbar
    _HAS_PYZBAR = True
except Exception:
    _HAS_PYZBAR = False

def parse_payload(s: str):
    """Barcode payload string -> list[(key, value)] (same UI as before)"""
    s = s.strip()
    # JSON first (some barcodes may carry JSON)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return [(str(k), str(v)) for k, v in obj.items()]
        return [("data", str(obj))]
    except Exception:
        pass
    # key=value;key=value
    parts = re.split(r'[;,\n]+', s)
    kv = []
    for p in parts:
        if '=' in p:
            k, v = p.split('=', 1)
            kv.append((k.strip(), v.strip()))
    return kv if kv else [("data", s)]

def draw_overlay(frame, symbology, text_lines, box_pts=None, fps=None):
    """Draw info panel and FPS on frame"""
    h, w = frame.shape[:2]

    # Draw polygon (if available)
    if box_pts is not None and len(box_pts) >= 4:
        try:
            pts = np.array(box_pts, dtype=int).reshape(-1, 2)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        except Exception:
            pass

    # Semi-transparent panel
    panel_w, panel_h = int(w * 0.60), int(h * 0.50)
    x0, y0 = 10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    # Title
    title = f"BARCODE ({symbology})"
    cv2.putText(frame, title, (x0 + 18, y0 + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    # Body
    y = y0 + 80
    for k, v in text_lines[:10]:
        line = f"{k}: {v}"
        cv2.putText(frame, line, (x0 + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        y += 36

    # FPS
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2, cv2.LINE_AA)

def try_open_uvc():
    """Open the specific UVC camera (with optional env override)."""
    idx = DEFAULT_CAM_INDEX
    env = os.getenv("QR_CAM_INDEX")
    if env and env.isdigit():
        idx = int(env)

    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open /dev/video{idx}")
        return None

    # Prefer MJPG for Jetson stability, if available
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREFERRED_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREFERRED_RES[1])

    ok, _ = cap.read()
    if not ok:
        print(f"[ERROR] /dev/video{idx} opened but failed to read a frame")
        cap.release()
        return None

    print(f"[INFO] Using UVC camera: /dev/video{idx}")
    return cap

def detect_barcodes(frame):
    """
    Detect barcodes and return a list of dicts:
      [{'type': 'CODE128', 'data': 'ABC123', 'polygon': [(x,y), ...]}, ...]
    """
    results = []

    # First: OpenCV BarcodeDetector (fast, no extra deps)
    if _HAS_OCV_BARCODE and _BARCODE_DET is not None:
        ok, decoded_infos, decoded_types, corners = _BARCODE_DET.detectAndDecode(frame)
        if ok and decoded_infos:
            for data, typ, pts in zip(decoded_infos, decoded_types, corners):
                if not data:
                    continue
                poly = None
                try:
                    # corners shape: (N, 4, 2)
                    poly = [(int(p[0]), int(p[1])) for p in pts.reshape(-1, 2)]
                except Exception:
                    poly = None
                results.append({
                    'type': typ if typ else 'UNKNOWN',
                    'data': data,
                    'polygon': poly
                })
            return results  # if found, done

    # Fallback: pyzbar
    if _HAS_PYZBAR:
        decoded = pyzbar.decode(frame)  # works on BGR/GRAY
        for obj in decoded:
            data = obj.data.decode('utf-8', errors='replace')
            typ = obj.type or 'UNKNOWN'
            poly = None
            if obj.polygon:
                poly = [(p.x, p.y) for p in obj.polygon]
            results.append({'type': typ, 'data': data, 'polygon': poly})

    return results

def main():
    cap = try_open_uvc()
    if cap is None:
        return

    last_t = time.time()
    fps = 0.0

    last_payload = None
    last_type = None
    parsed = []
    box_pts = None

    overlay_until = 0.0
    had_any_prev = False

    win_name = "Jetson Nano - D435 BARCODE Viewer"

    # 🔑 전체화면 설정
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame")
            break

        detections = detect_barcodes(frame)
        now = time.time()

        if detections:
            det = detections[0]
            data = det['data']
            sym = det['type']
            poly = det['polygon']

            if (not had_any_prev) or (data != last_payload):
                last_payload = data
                last_type = sym
                parsed = parse_payload(data)
                overlay_until = now + DISPLAY_SECONDS
                print(f"[BARCODE][{sym}] {data}")

            had_any_prev = True
            box_pts = poly
        else:
            had_any_prev = False
            box_pts = None

        dt = now - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_t = now

        if now < overlay_until and parsed:
            draw_overlay(frame, last_type or "UNKNOWN", parsed, box_pts, fps=fps)
        else:
            cv2.putText(frame, "Show a BARCODE to the camera",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


