#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, time
from pathlib import Path
import numpy as np, cv2, pyrealsense2 as rs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="imgfile/000004.jpg")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--jpegq", type=int, default=95)
    ap.add_argument("--ae_warmup_s", type=float, default=0.7)
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipe.start(cfg)

    # 워밍업
    t0 = time.time()
    while time.time() - t0 < args.ae_warmup_s:
        frames = pipe.wait_for_frames()

    # 1컷
    frames = pipe.wait_for_frames()
    color = frames.get_color_frame()
    if not color: pipe.stop(); raise RuntimeError("프레임 실패")
    img = np.asanyarray(color.get_data())
    pipe.stop()

    # ROI 선택 (드래그 → Enter/Space 확정, C 취소)
    r = cv2.selectROI("Select ROI (Enter to confirm)", img, False, False)
    cv2.destroyAllWindows()
    x,y,w,h = map(int, r)
    if w <= 0 or h <= 0:
        # ROI를 선택 안 하면 원본 저장
        crop = img
    else:
        crop = img[y:y+h, x:x+w]

    ok = cv2.imwrite(str(out), crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpegq)])
    if not ok: raise RuntimeError("저장 실패")
    print("[ok] saved →", out)

if __name__ == "__main__":
    main()
