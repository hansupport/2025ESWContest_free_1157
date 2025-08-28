# capture.py
# Jetson Nano + RealSense D435f (RGB @ /dev/video2)
# - 1920x1080 @ 15fps
# - Center ROI: (w=530, h=530), offset(dx=90, dy=-130)
# - 화면엔 ROI만 표시(프레임 나머지 영역은 시각화하지 않음)
# - 터미널 입력: 'd'+Enter → ROI 저장, 'f'+Enter → 전체화면 토글, 'q'+Enter → 종료

import sys
import time
import select
from pathlib import Path
import cv2

# ===== Window & Paths =====
WINDOW_TITLE = "D435f ROI Only"
ROOT = Path(__file__).resolve().parents[1]     # .../model
SAVE_DIR = ROOT / "pretrain" / "img_data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ===== Camera & ROI =====
DEVICE = "/dev/video2"
WIDTH, HEIGHT, FPS = 1920, 1080, 15

ROI_WH  = (530, 530)     # (w, h)
ROI_OFF = (90, -130)     # (dx, dy) from center

# 표시용 스케일: ROI를 화면 높이에 맞춰 키워서 보기 좋게
DISPLAY_HEIGHT = HEIGHT  # 1080으로 맞춤
FNAME_TPL = "capture_{ts}_roi.jpg"


def build_gst_pipeline(dev: str, w: int, h: int, fps: int, mjpg: bool) -> str:
    dev = str(dev)
    if mjpg:
        return (
            f"v4l2src device={dev} io-mode=2 ! "
            f"image/jpeg, width={w}, height={h}, framerate={fps}/1 ! "
            f"jpegdec ! videoconvert ! appsink"
        )
    else:
        return (
            f"v4l2src device={dev} io-mode=2 ! "
            f"video/x-raw, format=YUY2, width={w}, height={h}, framerate={fps}/1 ! "
            f"videoconvert ! appsink"
        )


def open_camera(device: str, w: int, h: int, fps: int):
    cam_id = device
    if isinstance(device, str) and device.startswith("/dev/video"):
        try:
            cam_id = int(device.replace("/dev/video", ""))
        except Exception:
            pass

    # V4L2 먼저
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()

    # GStreamer (MJPG) → (YUY2) 순서로 폴백
    gst = build_gst_pipeline(device, w, h, fps, mjpg=True)
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()

    gst = build_gst_pipeline(device, w, h, fps, mjpg=False)
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()

    return None


def compute_center_roi_xy(img_shape, roi_wh, roi_off):
    H, W = img_shape[:2]
    rw, rh = map(int, roi_wh)
    dx, dy = map(int, roi_off)
    rw = max(1, min(rw, W))
    rh = max(1, min(rh, H))
    cx, cy = W // 2 + dx, H // 2 + dy
    x = max(0, min(W - rw, cx - rw // 2))
    y = max(0, min(H - rh, cy - rh // 2))
    return x, y, rw, rh


def stdin_readline_nonblock(timeout_sec=0.001):
    """터미널에서 줄 입력을 논블로킹으로 읽는다. (Linux 전용)"""
    r, _, _ = select.select([sys.stdin], [], [], timeout_sec)
    if r:
        return sys.stdin.readline().strip()
    return None


def main():
    print("[open] device=", DEVICE, "res=", (WIDTH, HEIGHT), "fps=", FPS)
    print("[hint] 터미널에서 'd'+Enter=저장 | 'f'+Enter=전체화면 토글 | 'q'+Enter=종료")
    cap = open_camera(DEVICE, WIDTH, HEIGHT, FPS)
    if not cap or not cap.isOpened():
        print("[fatal] failed to open camera", DEVICE)
        return

    # 전체화면 가능한 윈도우 생성
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    is_fullscreen = True

    saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            # ROI만 잘라서 표시
            x, y, rw, rh = compute_center_roi_xy(frame.shape, ROI_WH, ROI_OFF)
            roi = frame[y:y+rh, x:x+rw]

            # 보기 편하게 ROI를 화면 높이에 맞춰 확대
            if rh > 0 and DISPLAY_HEIGHT > 0 and rh != DISPLAY_HEIGHT:
                scale = float(DISPLAY_HEIGHT) / float(rh)
                new_w = max(1, int(rw * scale))
                roi_to_show = cv2.resize(roi, (new_w, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
            else:
                roi_to_show = roi

            cv2.imshow(WINDOW_TITLE, roi_to_show)
            # 윈도우 이벤트 처리를 위해 waitKey는 유지(키 입력은 터미널로 받음)
            cv2.waitKey(1)

            # ---- 터미널 입력 처리 ----
            cmd = stdin_readline_nonblock(0.001)
            if not cmd:
                continue
            c = cmd.strip().lower()
            if c == 'q':
                break
            elif c == 'f':
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(
                    WINDOW_TITLE,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                )
                print(f"[ui] fullscreen={'ON' if is_fullscreen else 'OFF'}")
            elif c == 'd':
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = SAVE_DIR / FNAME_TPL.format(ts=ts)
                ok = cv2.imwrite(str(out_path), roi)  # 저장은 원본 크기 ROI
                if ok:
                    saved += 1
                    print(f"[save#{saved}] {out_path}")
                else:
                    print("[warn] failed to save image")

    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[exit] bye")


if __name__ == "__main__":
    main()
