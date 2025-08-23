# roi_depth_preproc_rs.py
# Jetson Nano + RealSense D435f
# Depth 1280x720 @ 6fps + RealSense 사전/후처리 필터 + 중앙 기준 ROI 편집기
# 변경점: HUD 텍스트 안전 여백 적용(위쪽 잘림 방지), 전체화면 토글/옵션 추가

import argparse, time, sys
import numpy as np
import cv2

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def compute_roi_rect(w, h, roi_w, roi_h, dx, dy):
    cx = w // 2 + dx
    cy = h // 2 + dy
    x1 = clamp(cx - roi_w // 2, 0, w - roi_w)
    y1 = clamp(cy - roi_h // 2, 0, h - roi_h)
    x2, y2 = x1 + roi_w, y1 + roi_h
    return x1, y1, x2, y2

def draw_hud(colorized, roi_rect, roi_w, roi_h, dx, dy,
             fps, depth_scale, z_stats, filt_state, crop_only):
    x1, y1, x2, y2 = roi_rect
    vis = colorized if crop_only else colorized.copy()
    if not crop_only:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2, cv2.LINE_AA)
        h, w = vis.shape[:2]
        cv2.drawMarker(vis, (w//2, h//2), (255,255,255),
                       markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)

    txt1 = f"ROI {roi_w}x{roi_h} | dx={dx}, dy={dy} | FPS~{fps:.1f} | crop_only={'ON' if crop_only else 'OFF'}"
    zmin, zmed, zmax, zn = z_stats
    if zn > 0:
        txt2 = f"Depth(m) ROI: min={zmin:.3f}, med={zmed:.3f}, max={zmax:.3f}, n={zn}"
    else:
        txt2 = "Depth(m) ROI: n=0 (no valid depth)"
    txt3 = ("Filt "
            f"decim={filt_state['decim']} "
            f"spat={'ON' if filt_state['spatial'] else 'OFF'} "
            f"temp={'ON' if filt_state['temporal'] else 'OFF'} "
            f"hole={filt_state['holes']} "
            f"disp={'ON' if filt_state['use_disp'] else 'OFF'} "
            f"th=[{filt_state['min_z']:.2f},{filt_state['max_z']:.2f}]m")

    # 안전 여백: 화면 위에서 글자 한 줄 만큼 내림
    y_base = 30  # 첫 줄 기준 y (이전 코드의 음수 y 방지)
    line_gap = 26
    pad = 6
    for i, txt in enumerate([txt1, txt2, txt3]):
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_text = y_base + i * line_gap
        # 배경 박스
        cv2.rectangle(vis,
                      (8, y_text - th - pad),
                      (8 + tw + pad*2, y_text + pad),
                      (0, 0, 0), -1)
        # 텍스트
        cv2.putText(vis, txt, (8 + pad, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
    return vis

def set_fullscreen(win_name, on=True):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        win_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL
    )

def main():
    p = argparse.ArgumentParser(description="D435f Depth ROI 편집기 (1280x720@6fps + Preproc)")
    p.add_argument("--roi_w", type=int, default=480, help="중앙 기준 ROI 가로(px)")
    p.add_argument("--roi_h", type=int, default=270, help="중앙 기준 ROI 세로(px)")
    p.add_argument("--dx", type=int, default=0, help="ROI x 평행이동(+우)")
    p.add_argument("--dy", type=int, default=0, help="ROI y 평행이동(+하)")
    p.add_argument("--crop_only", action="store_true", help="ROI만 표시")
    p.add_argument("--lock", action="store_true", help="ROI/필터 조작 잠금")
    p.add_argument("--fullscreen", action="store_true", help="시작부터 전체화면")

    # 필터 기본값
    p.add_argument("--decim", type=int, default=1, help="Decimation magnitude (1=off, 2~8)")
    p.add_argument("--spatial", action="store_true", help="Spatial filter on")
    p.add_argument("--spatial_mag", type=int, default=2, help="Spatial magnitude(1~5)")
    p.add_argument("--spatial_alpha", type=float, default=0.5, help="Spatial smooth_alpha(0~1)")
    p.add_argument("--spatial_delta", type=int, default=20, help="Spatial smooth_delta(1~50)")
    p.add_argument("--temporal", action="store_true", help="Temporal filter on")
    p.add_argument("--temp_alpha", type=float, default=0.4, help="Temporal smooth_alpha(0~1)")
    p.add_argument("--temp_delta", type=int, default=20, help="Temporal smooth_delta(1~100)")
    p.add_argument("--holes", type=int, default=1, help="Hole-filling mode (0~2)")
    p.add_argument("--use_disp", action="store_true", help="Disparity 변환 사용(Spatial/Temporal 앞/뒤)")
    p.add_argument("--min_z", type=float, default=0.15, help="Threshold min depth (m)")
    p.add_argument("--max_z", type=float, default=3.0, help="Threshold max depth (m)")
    args = p.parse_args()

    import pyrealsense2 as rs

    W, H, FPS = 1280, 720, 6  # D435 계열 depth 최대 해상도
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    profile = pipe.start(cfg)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    # 필터 구성
    decim = rs.decimation_filter()
    decim.set_option(rs.option.filter_magnitude, float(args.decim))

    thresh = rs.threshold_filter()
    thresh.set_option(rs.option.min_distance, float(args.min_z))
    thresh.set_option(rs.option.max_distance, float(args.max_z))

    spat = rs.spatial_filter()
    spat.set_option(rs.option.filter_magnitude, float(args.spatial_mag))
    spat.set_option(rs.option.filter_smooth_alpha, float(args.spatial_alpha))
    spat.set_option(rs.option.filter_smooth_delta, float(args.spatial_delta))
    spat.set_option(rs.option.holes_fill, 0)

    temp = rs.temporal_filter()
    temp.set_option(rs.option.filter_smooth_alpha, float(args.temp_alpha))
    temp.set_option(rs.option.filter_smooth_delta, float(args.temp_delta))

    hole = rs.hole_filling_filter(args.holes)
    d2d = rs.disparity_transform(True)
    disp2d = rs.disparity_transform(False)
    colorizer = rs.colorizer()  # 시각화용

    roi_w, roi_h = args.roi_w, args.roi_h
    dx, dy = args.dx, args.dy
    step = 10
    crop_only = args.crop_only

    t_prev = time.time()
    fps_ema = 0.0

    # 상태 플래그
    state = {
        "decim": int(args.decim),
        "spatial": bool(args.spatial),
        "temporal": bool(args.temporal),
        "holes": int(args.holes),
        "use_disp": bool(args.use_disp),
        "min_z": float(args.min_z),
        "max_z": float(args.max_z),
    }

    WIN = "D435f Depth ROI (1280x720@6fps)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        set_fullscreen(WIN, True)
    is_fullscreen = args.fullscreen

    try:
        # 워밍업
        for _ in range(5):
            pipe.wait_for_frames()

        while True:
            frames = pipe.wait_for_frames()
            d = frames.get_depth_frame()
            if not d:
                continue

            # 필터 파이프라인
            out = d
            if state["decim"] > 1:
                decim.set_option(rs.option.filter_magnitude, float(state["decim"]))
                out = decim.process(out)
            out = thresh.process(out)
            if state["use_disp"]:
                out = d2d.process(out)
            if state["spatial"]:
                out = spat.process(out)
            if state["temporal"]:
                out = temp.process(out)
            if state["use_disp"]:
                out = disp2d.process(out)
            if state["holes"] in (0,1,2):
                hole = rs.hole_filling_filter(state["holes"])
                out = hole.process(out)

            out = out.as_depth_frame()

            # 현재 해상도(Decimation 적용 시 변함)
            Wc, Hc = out.get_width(), out.get_height()

            # ROI 보정
            roi_w = clamp(roi_w, 10, Wc)
            roi_h = clamp(roi_h, 10, Hc)
            x1, y1, x2, y2 = compute_roi_rect(Wc, Hc, roi_w, roi_h, dx, dy)

            # 시각화용 컬러맵
            depth_vis = np.asanyarray(colorizer.colorize(out).get_data())

            # ROI 통계(유효 깊이만)
            depth_np = np.asanyarray(out.get_data())  # uint16
            roi_np = depth_np[y1:y2, x1:x2]
            valid = roi_np > 0
            if np.any(valid):
                z_vals_m = (roi_np[valid].astype(np.float32)) * depth_scale
                zmin = float(np.min(z_vals_m))
                zmed = float(np.median(z_vals_m))
                zmax = float(np.max(z_vals_m))
                zn = int(z_vals_m.size)
                z_stats = (zmin, zmed, zmax, zn)
            else:
                z_stats = (0.0, 0.0, 0.0, 0)

            # FPS
            t_now = time.time()
            inst = 1.0 / max(1e-6, (t_now - t_prev))
            t_prev = t_now
            fps_ema = 0.9*fps_ema + 0.1*inst if fps_ema > 0 else inst

            if crop_only:
                view = depth_vis[y1:y2, x1:x2]
            else:
                view = draw_hud(depth_vis, (x1,y1,x2,y2), roi_w, roi_h, dx, dy,
                                fps_ema, depth_scale, z_stats, state, crop_only)

            cv2.imshow(WIN, view)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):
                break

            # 전체화면 토글
            if key == ord('f'):
                is_fullscreen = not is_fullscreen
                set_fullscreen(WIN, is_fullscreen)
                continue

            if args.lock:
                continue  # ROI/필터 조작 잠금

            # ROI 조작
            if key == ord('w'): dy -= step
            elif key == ord('s'): dy += step
            elif key == ord('a'): dx -= step
            elif key == ord('d'): dx += step
            elif key == ord('j'): roi_w -= step
            elif key == ord('l'): roi_w += step
            elif key == ord('i'): roi_h -= step
            elif key == ord('k'): roi_h += step
            elif key == ord('x'): crop_only = not crop_only
            elif key == ord('r'):
                dx, dy = 0, 0
                roi_w, roi_h = args.roi_w, args.roi_h

            # 필터 토글/조절
            elif key == ord('g'):
                state["decim"] = 1 if state["decim"] > 1 else 2
            elif key == ord('s'):
                state["spatial"] = not state["spatial"]
            elif key == ord('t'):
                state["temporal"] = not state["temporal"]
            elif key == ord('h'):
                state["holes"] = (state["holes"] + 1) % 3
            elif key == ord('b'):
                state["use_disp"] = not state["use_disp"]

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
