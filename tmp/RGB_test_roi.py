# roi_view_realsense.py
# Jetson Nano + RealSense D435f 컬러 1920x1080@6fps 시각화
# 중앙 기준 직사각형 ROI (w,h)와 평행이동 (dx,dy) 조절

import argparse
import time
import sys
import cv2
import numpy as np

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def compute_roi_rect(w, h, roi_w, roi_h, dx, dy):
    cx = w // 2 + dx
    cy = h // 2 + dy
    x1 = clamp(cx - roi_w // 2, 0, w - 1)
    y1 = clamp(cy - roi_h // 2, 0, h - 1)
    # 화면 밖으로 나가지 않게 재조정
    x1 = clamp(x1, 0, w - roi_w)
    y1 = clamp(y1, 0, h - roi_h)
    x2 = x1 + roi_w
    y2 = y1 + roi_h
    return x1, y1, x2, y2

def draw_hud(img, roi_rect, roi_w, roi_h, dx, dy, fps, show_crop):
    x1, y1, x2, y2 = roi_rect
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
    # 중앙 십자선
    ch, cw = img.shape[:2]
    cv2.drawMarker(vis, (cw//2, ch//2), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)

    txt = f"ROI {roi_w}x{roi_h} | dx={dx}, dy={dy} | {cw}x{ch}@6fps | FPS~{fps:.1f} | crop_only={'ON' if show_crop else 'OFF'}"
    cv2.rectangle(vis, (8, 8), (8 + 9*len(txt), 36), (0, 0, 0), -1)
    cv2.putText(vis, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
    return vis

def run_realsense(args):
    import pyrealsense2 as rs  # 지역 import
    W, H, FPS = 1920, 1080, 6
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    prof = pipe.start(cfg)

    # 첫 프레임 동기화
    for _ in range(5):
        pipe.wait_for_frames()

    roi_w, roi_h = args.roi_w, args.roi_h
    dx, dy = args.dx, args.dy
    step = 10
    show_crop = False

    t_prev = time.time()
    fps_ema = 0.0

    try:
        while True:
            frames = pipe.wait_for_frames()
            c = frames.get_color_frame()
            if not c:
                continue
            frame = np.asanyarray(c.get_data())
            Hcur, Wcur = frame.shape[:2]

            # ROI 크기 화면에 맞춰 보정
            roi_w = clamp(roi_w, 10, Wcur)
            roi_h = clamp(roi_h, 10, Hcur)

            x1, y1, x2, y2 = compute_roi_rect(Wcur, Hcur, roi_w, roi_h, dx, dy)
            roi_rect = (x1, y1, x2, y2)

            t_now = time.time()
            inst = 1.0 / max(1e-6, (t_now - t_prev))
            t_prev = t_now
            fps_ema = 0.9*fps_ema + 0.1*inst if fps_ema > 0 else inst

            if show_crop:
                view = frame[y1:y2, x1:x2]
            else:
                view = draw_hud(frame, roi_rect, roi_w, roi_h, dx, dy, fps_ema, show_crop)

            cv2.imshow("D435f Color ROI", view)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):
                break
            elif key == ord('w'):
                dy -= step
            elif key == ord('s'):
                dy += step
            elif key == ord('a'):
                dx -= step
            elif key == ord('d'):
                dx += step
            elif key == ord('j'):
                roi_w -= step
            elif key == ord('l'):
                roi_w += step
            elif key == ord('i'):
                roi_h -= step
            elif key == ord('k'):
                roi_h += step
            elif key == ord('r'):
                dx, dy = 0, 0
                roi_w, roi_h = args.roi_w, args.roi_h
            elif key == ord('x'):
                show_crop = not show_crop

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

def run_v4l2(args):
    # /dev/video2 직접 접근 (MJPG 권장)
    W, H, FPS = 1920, 1080, 6
    dev = args.device
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"열 수 없음: {dev}", file=sys.stderr)
        return
    # 포맷/해상도/프레임
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # 설정 확인
    Wcur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hcur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPScur = cap.get(cv2.CAP_PROP_FPS)

    roi_w, roi_h = args.roi_w, args.roi_h
    dx, dy = args.dx, args.dy
    step = 10
    show_crop = False

    t_prev = time.time()
    fps_ema = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            Himg, Wimg = frame.shape[:2]

            roi_w = clamp(roi_w, 10, Wimg)
            roi_h = clamp(roi_h, 10, Himg)

            x1, y1, x2, y2 = compute_roi_rect(Wimg, Himg, roi_w, roi_h, dx, dy)
            roi_rect = (x1, y1, x2, y2)

            t_now = time.time()
            inst = 1.0 / max(1e-6, (t_now - t_prev))
            t_prev = t_now
            fps_ema = 0.9*fps_ema + 0.1*inst if fps_ema > 0 else inst

            if show_crop:
                view = frame[y1:y2, x1:x2]
            else:
                view = draw_hud(frame, roi_rect, roi_w, roi_h, dx, dy, fps_ema, show_crop)

            cv2.imshow("V4L2 Color ROI", view)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):
                break
            elif key == ord('w'):
                dy -= step
            elif key == ord('s'):
                dy += step
            elif key == ord('a'):
                dx -= step
            elif key == ord('d'):
                dx += step
            elif key == ord('j'):
                roi_w -= step
            elif key == ord('l'):
                roi_w += step
            elif key == ord('i'):
                roi_h -= step
            elif key == ord('k'):
                roi_h += step
            elif key == ord('r'):
                dx, dy = 0, 0
                roi_w, roi_h = args.roi_w, args.roi_h
            elif key == ord('x'):
                show_crop = not show_crop

    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    p = argparse.ArgumentParser(description="D435f 1920x1080@6fps ROI 시각화")
    p.add_argument("--backend", choices=["realsense", "v4l2"], default="realsense", help="realsense SDK 또는 v4l2(/dev/videoX)")
    p.add_argument("--device", default="/dev/video2", help="--backend v4l2 일 때 사용할 장치 경로")
    p.add_argument("--roi_w", type=int, default=400, help="중앙 기준 ROI 가로 픽셀")
    p.add_argument("--roi_h", type=int, default=300, help="중앙 기준 ROI 세로 픽셀")
    p.add_argument("--dx", type=int, default=0, help="중앙 기준 ROI x 평행이동(+우)")
    p.add_argument("--dy", type=int, default=0, help="중앙 기준 ROI y 평행이동(+하)")
    args = p.parse_args()

    if args.backend == "realsense":
        run_realsense(args)
    else:
        run_v4l2(args)

if __name__ == "__main__":
    main()
