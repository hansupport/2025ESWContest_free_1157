# qrcheck.py — show QR info only for ~3 seconds after detection
import os
import cv2
import json
import re
import time

# ==== SETTINGS ====
DISPLAY_SECONDS = 3.0           # show overlay for N seconds after a detection
DEFAULT_CAM_INDEX = 2           # your confirmed RGB node (e.g., /dev/video2)
PREFERRED_RES = (640, 480)      # stable on Jetson Nano
# ===================

def parse_payload(s: str):
    """QR payload string -> list[(key, value)]"""
    s = s.strip()
    # JSON first
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

def draw_overlay(frame, text_lines, box_pts=None, fps=None):
    """Draw info panel and FPS on frame"""
    h, w = frame.shape[:2]

    # Draw QR polygon
    if box_pts is not None and len(box_pts) == 4:
        pts = box_pts.reshape(-1, 2).astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # Semi-transparent panel
    panel_w, panel_h = int(w * 0.55), int(h * 0.48)
    x0, y0 = 10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    # Title
    cv2.putText(frame, "QR INFO", (x0 + 18, y0 + 42),
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

def main():
    qr = cv2.QRCodeDetector()
    cap = try_open_uvc()
    if cap is None:
        return

    last_t = time.time()
    fps = 0.0

    last_payload = None
    parsed = []
    box_pts = None

    # show overlay until this timestamp; 0 means hidden
    overlay_until = 0.0

    # to avoid re-triggering while the same QR stays in view:
    had_data_prev = False

    win_name = "Jetson Nano - D435 QR Viewer"
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame")
            break

        data, pts, _ = qr.detectAndDecode(frame)
        now = time.time()

        if data:
            # Trigger only when QR just appeared OR payload changed
            if (not had_data_prev) or (data != last_payload):
                last_payload = data
                parsed = parse_payload(data)
                overlay_until = now + DISPLAY_SECONDS
                print("[QR]", data)
            had_data_prev = True
            box_pts = pts
        else:
            had_data_prev = False
            box_pts = None

        # FPS
        dt = now - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_t = now

        # Draw overlay only while timer is active
        if now < overlay_until and parsed:
            draw_overlay(frame, parsed, box_pts, fps=fps)
        else:
            # (Optional) small hint; comment out if not needed
            cv2.putText(frame, "Show a QR code to the camera",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
