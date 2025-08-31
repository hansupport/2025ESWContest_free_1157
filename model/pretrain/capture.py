# capture.py (3 ROI 시각화: lm/rm/c를 모두 포함하는 최소 사각형 표시 + 각 ROI 박스/라벨)
# Jetson Nano + RealSense D435f (RGB @ /dev/video2)
# - 1920x1080 @ 15fps
# - ROI 3개: LEFT_MIRROR(lm), RIGHT_MIRROR(rm), CENTER(c)
# - 화면엔 세 ROI를 모두 포함하는 최소 바운딩 박스를 잘라서 확대 표시
# - 각 ROI에 구분선(네모)과 라벨을 표시
# - 터미널 입력: 'd'+Enter → 센터 ROI 저장, 'f'+Enter → 전체화면 토글, 'q'+Enter → 종료

import sys
import time
import select
from pathlib import Path
import cv2

# ===== Window & Paths =====
WINDOW_TITLE = "D435f 3-ROI View"
ROOT = Path(__file__).resolve().parents[1]     # .../model
SAVE_DIR = ROOT / "pretrain" / "img_data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ===== Camera & ROI =====
DEVICE = "/dev/video2"
WIDTH, HEIGHT, FPS = 1920, 1080, 15

# --- ROI 정의 (config.yaml의 datamatrix.rois에 맞춤) ---
# 좌/우 거울뷰는 hflip=True로 캡처 시 뒤집어 쓰지만, 여기 시각화는 위치 네모만 그리므로 hflip은 무관
ROI_LM_WH  = (270, 470)    # left mirror
ROI_LM_OFF = (-450, -160)

ROI_RM_WH  = (260, 460)    # right mirror
ROI_RM_OFF = (670, -170)

ROI_C_WH   = (520, 550)    # center
ROI_C_OFF  = (90, -130)

# 표시용 스케일: 바운딩 박스를 화면 높이에 맞춰 키워서 보기 좋게
DISPLAY_HEIGHT = HEIGHT  # 1080으로 맞춤
FNAME_TPL = "capture_{ts}_roi.jpg"   # 저장은 기존대로 '센터 ROI'만

# === 색상(BGR)
COL_BOX_LM = (255, 128, 0)   # 주황
COL_BOX_RM = (0, 128, 255)   # 노랑-파랑 계열
COL_BOX_C  = (0, 255, 0)     # 초록
COL_BBOX   = (255, 255, 255) # 바깥 바운딩 박스 테두리(흰색)


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


def union_rects(rects, img_shape):
    """여러 (x,y,w,h)를 포함하는 최소 바운딩 박스 반환, 이미지 경계로 클램프"""
    H, W = img_shape[:2]
    if not rects:
        return (0, 0, W, H)
    x1 = min([r[0] for r in rects])
    y1 = min([r[1] for r in rects])
    x2 = max([r[0] + r[2] for r in rects])
    y2 = max([r[1] + r[3] for r in rects])
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(1, min(W, x2))
    y2 = max(1, min(H, y2))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def draw_roi_rect(dst_img, rect, color, label=None, thickness=2):
    """dst_img 내 로컬 좌표 rect(x,y,w,h)에 박스 및 라벨"""
    x, y, w, h = rect
    cv2.rectangle(dst_img, (x, y), (x + w, y + h), color, thickness)
    if label:
        # 라벨 박스
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx, ty = x + 5, max(0, y - 8)
        # 배경 반투명 없이 간단 박스
        cv2.rectangle(dst_img, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), color, -1)
        cv2.putText(dst_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def stdin_readline_nonblock(timeout_sec=0.001):
    """터미널에서 줄 입력을 논블로킹으로 읽는다. (Linux 전용)"""
    r, _, _ = select.select([sys.stdin], [], [], timeout_sec)
    if r:
        return sys.stdin.readline().strip()
    return None


def main():
    print("[open] device=", DEVICE, "res=", (WIDTH, HEIGHT), "fps=", FPS)
    print("[hint] 터미널에서 'd'+Enter=센터 ROI 저장 | 'f'+Enter=전체화면 토글 | 'q'+Enter=종료")
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

            # --- 각 ROI 절대좌표 계산 ---
            x_lm, y_lm, w_lm, h_lm = compute_center_roi_xy(frame.shape, ROI_LM_WH, ROI_LM_OFF)
            x_rm, y_rm, w_rm, h_rm = compute_center_roi_xy(frame.shape, ROI_RM_WH, ROI_RM_OFF)
            x_c,  y_c,  w_c,  h_c  = compute_center_roi_xy(frame.shape, ROI_C_WH,  ROI_C_OFF)

            # --- 세 ROI를 모두 포함하는 최소 바운딩 박스 ---
            bbx, bby, bbw, bbh = union_rects(
                [(x_lm, y_lm, w_lm, h_lm), (x_rm, y_rm, w_rm, h_rm), (x_c, y_c, w_c, h_c)],
                frame.shape
            )

            # --- 바운딩 박스 영역만 잘라서 표시 ---
            crop = frame[bby:bby+bbh, bbx:bbx+bbw].copy()

            # 로컬 좌표로 ROI 변환
            lm_local = (x_lm - bbx, y_lm - bby, w_lm, h_lm)
            rm_local = (x_rm - bbx, y_rm - bby, w_rm, h_rm)
            c_local  = (x_c  - bbx, y_c  - bby, w_c,  h_c)

            # --- ROI 사각형 그리기 ---
            draw_roi_rect(crop,  c_local,  COL_BOX_C,  "CENTER (c)", thickness=2)
            draw_roi_rect(crop,  lm_local, COL_BOX_LM, "LEFT MIRROR (lm)", thickness=2)
            draw_roi_rect(crop,  rm_local, COL_BOX_RM, "RIGHT MIRROR (rm)", thickness=2)

            # 바깥 바운딩 박스 테두리(시각화용) - crop 전체에 테두리
            cv2.rectangle(crop, (1, 1), (crop.shape[1]-2, crop.shape[0]-2), COL_BBOX, 1)

            # 보기 편하게 화면 높이에 맞춰 확대
            if bbh > 0 and DISPLAY_HEIGHT > 0 and bbh != DISPLAY_HEIGHT:
                scale = float(DISPLAY_HEIGHT) / float(bbh)
                new_w = max(1, int(bbw * scale))
                to_show = cv2.resize(crop, (new_w, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
            else:
                to_show = crop

            cv2.imshow(WINDOW_TITLE, to_show)
            # 윈도우 이벤트 처리를 위해 waitKey는 유지(키 입력은 터미널로 받음)
            cv2.waitKey(1)

            # ---- 터미널 입력 처리 ----
            cmd = stdin_readline_nonblock(0.001)
            if not cmd:
                continue
            ckey = cmd.strip().lower()
            if ckey == 'q':
                break
            elif ckey == 'f':
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(
                    WINDOW_TITLE,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                )
                print("[ui] fullscreen={}".format('ON' if is_fullscreen else 'OFF'))
            elif ckey == 'd':
                # 기존 동작 유지: 센터 ROI만 저장
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = SAVE_DIR / FNAME_TPL.format(ts=ts)
                center_roi = frame[y_c:y_c+h_c, x_c:x_c+w_c]
                ok = cv2.imwrite(str(out_path), center_roi)
                if ok:
                    saved += 1
                    print("[save#{}] {}".format(saved, out_path))
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
