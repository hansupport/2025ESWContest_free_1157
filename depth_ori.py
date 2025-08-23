# depth_ori.py
# D435f 컨베이어 L/W/H 측정 + 포장재 추천
# - 저FPS/젯슨 대응: poll_for_frames, non-blocking calibration, single-frame reuse
# - 얇은 물체 대응: ROI 스택 중간값 + 이중 임계 + 그래디언트/베이스라인 기반
# - HUD는 ASCII만 사용(한글 글리프 문제 방지)
# Keys: q quit | f fullscreen | +/- scale | [ ] ROI size | c/r start/stop CAL | t measure | a auto-trigger | x crop-only | m mask toggle
import pyrealsense2 as rs
import numpy as np
import cv2, time

rng = np.random.default_rng(42)

# ===== Stream setup =====
W, H, FPS = 1280, 720, 6

# ===== Display/UI =====
WIN_NAME = "Conveyor Depth Trigger (Low-FPS Safe)"
DISPLAY_SCALE = 1.20
FULLSCREEN = False
INFO_BAR_H = 72
CROP_ONLY = False
SHOW_MASK = True
SPINNER_CHARS = "|/-\\"

# ===== ROI: absolute at 1280x720 -> auto-scaled to current =====
ROI_REF_W, ROI_REF_H = 1280, 720
ROI_W_PX_REF = 230
ROI_H_PX_REF = 230
DX_PX_REF    = 5
DY_PX_REF    = -60
ROI_EDGE_EROSION = 2  # erode edges inside ROI to suppress boundary noise

# ===== Lightweight preprocessing (threshold only) =====
USE_THRESHOLD = True
MIN_Z, MAX_Z = 0.15, 3.00  # meters

# ===== Calibration (non-blocking) =====
CALI_MIN_STACK = 5
CALI_TARGET_STACK = 20
CALI_TIMEOUT_SEC = 45.0
PLANE_TAU = 0.006
PLANE_MIN_INL = 3000
BOTTOM_ROI_RATIO = 0.25
PLANE_SAMPLE_STRIDE = 4

# ===== Thin-object segmentation/measurement =====
MEAS_MIN_STACK = 2
MEAS_STACK_N   = 5
MEAS_TIMEOUT_SEC = 12.0
K_B = 3.0
K_G = 4.0
H_LO_FLOOR = 0.0005   # 0.5 mm
H_HI_FLOOR = 0.0015   # 1.5 mm
OPEN_K = (3,3); CLOSE_K = (5,5); CLOSE_ITER = 2
MIN_OBJ_PIX = 120

# ===== Packing config =====
ROLL_WIDTHS_MM = [300, 400, 500, 600]
EDGE_MARGIN_MM = 20
OVERLAP_MM     = 30
SAFETY_PAD_MM  = 5

# ===== Globals =====
X_MUL = Y_MUL = None
BASE_MU = BASE_SIG = None
BASE_ROI_RECT = None
PLANE_N = None; PLANE_P0 = None
HAVE_BASE = False
AUTO_TRIGGER = False

# Calibration state
CALI_ACTIVE = False
CALI_STACK_BUF = []
CALI_T0 = 0.0
CALI_LAST_MSG = ""
CALI_LAST_FN = -1

# Last frame cache
LAST_U16 = None
LAST_FN  = -1
LAST_TS_MS = 0.0

# ===== Utils =====
def clamp(v, lo, hi): return max(lo, min(hi, v))

def start_depth_pipeline(pipe, fps=FPS):
    cfg = rs.config()
    for w,h in [(W,H),(848,480),(640,480)]:
        try: cfg.disable_all_streams()
        except Exception: pass
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        try:
            return pipe.start(cfg)
        except Exception:
            continue
    cfg = rs.config(); cfg.enable_stream(rs.stream.depth)
    return pipe.start(cfg)

def precompute_intr_grid(intr):
    w, h = intr.width, intr.height
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_mul = (xs - intr.ppx) / intr.fx
    y_mul = (ys - intr.ppy) / intr.fy
    return x_mul, y_mul

def deproject_u16(depth_u16, depth_scale, x_mul, y_mul):
    z = depth_u16.astype(np.float32) * depth_scale
    x = x_mul * z; y = y_mul * z
    return np.dstack((x,y,z))

def center_roi_scaled(w, h):
    sx = w / float(ROI_REF_W); sy = h / float(ROI_REF_H)
    Sx = int(round(ROI_W_PX_REF * sx)); Sy = int(round(ROI_H_PX_REF * sy))
    S = int(max(8, min(min(w,h), min(Sx, Sy))))
    dx = int(round(DX_PX_REF * sx)); dy = int(round(DY_PX_REF * sy))
    cx = w//2 + dx; cy = h//2 + dy
    x0 = clamp(cx - S//2, 0, w - S); y0 = clamp(cy - S//2, 0, h - S)
    return x0, y0, x0+S, y0+S, S, dx, dy

def apply_colormap_u8(gray):
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray, cmap)

def orient_normal_to_camera(n):
    cam_dir = np.array([0., 0., -1.], dtype=np.float32)
    return n if np.dot(n, cam_dir) > 0 else -n

def fit_plane_ransac(P, iters=400, tau=PLANE_TAU, min_inliers=PLANE_MIN_INL):
    N = P.shape[0]
    if N < 3: return None
    best_m = None; best_n = None
    for _ in range(iters):
        A,B,C = P[rng.choice(N, 3, replace=False)]
        n = np.cross(B-A, C-A); nn = np.linalg.norm(n)
        if nn < 1e-8: continue
        n /= nn; d = -np.dot(n, A)
        dist = np.abs(P.dot(n) + d)
        m = dist < tau
        if best_m is None or m.sum() > best_m.sum():
            best_m = m; best_n = n
    if best_m is None or best_m.sum() < min_inliers: return None
    Pin = P[best_m]; c = Pin.mean(axis=0)
    _,_,Vt = np.linalg.svd(Pin - c, full_matrices=False)
    n_ref = Vt[-1]; n_ref /= (np.linalg.norm(n_ref)+1e-12)
    n_ref = orient_normal_to_camera(n_ref)
    return n_ref, c

def plane_axes_from_normal(n):
    t = np.array([1.,0.,0.]) if abs(n[0]) < 0.9 else np.array([0.,0.,1.])
    u = np.cross(n, t); u /= (np.linalg.norm(u)+1e-12)
    v = np.cross(n, u); v /= (np.linalg.norm(v)+1e-12)
    return u, v

def sdist_map(P3, n, p0):
    S = np.einsum('ijk,k->ij', P3 - p0, n).astype(np.float32)
    S[~np.isfinite(P3).all(axis=2)] = np.nan
    return S

def largest_external_component(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
    return filled, int(cv2.contourArea(c))

def hysteresis_region(Elo, seed_hi, max_iter=8):
    kernel = np.ones((3,3), np.uint8)
    cur = seed_hi.copy()
    for _ in range(max_iter):
        dil = cv2.dilate(cur, kernel, iterations=1)
        nxt = cv2.bitwise_and(dil, Elo)
        if np.array_equal(nxt, cur): break
        cur = nxt
    return cur

def recommend_pack(L_mm, W_mm, H_mm):
    need_w_A = L_mm + 2*H_mm + 2*EDGE_MARGIN_MM + SAFETY_PAD_MM
    cut_A    = 2*(W_mm + H_mm) + OVERLAP_MM + SAFETY_PAD_MM
    need_w_B = W_mm + 2*H_mm + 2*EDGE_MARGIN_MM + SAFETY_PAD_MM
    cut_B    = 2*(L_mm + H_mm) + OVERLAP_MM + SAFETY_PAD_MM
    def pick(need):
        cands = [rw for rw in ROLL_WIDTHS_MM if rw >= need]
        return (min(cands) if cands else None)
    opts = []
    ra = pick(need_w_A); rb = pick(need_w_B)
    if ra: opts.append(("A", ra, cut_A, need_w_A, cut_A + 0.2*(ra-need_w_A)))
    if rb: opts.append(("B", rb, cut_B, need_w_B, cut_B + 0.2*(rb-need_w_B)))
    if not opts: return None
    orient, roll, cutlen, needw, _ = min(opts, key=lambda x: x[-1])
    return dict(orientation=orient, roll_width_mm=int(round(roll)),
                required_width_mm=float(needw), cut_length_mm=float(cutlen))

def draw_progress_bar(img, x, y, w, h, ratio, fg=(0,220,255), bg=(45,45,45)):
    ratio = float(max(0.0, min(1.0, ratio)))
    cv2.rectangle(img, (x,y), (x+w, y+h), bg, -1)
    cv2.rectangle(img, (x,y), (x+int(w*ratio), y+h), fg, -1)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), 1)

# ===== Segmentation/measurement =====
def segment_and_measure(u16_med_roi, roi_rect, depth_scale):
    x0,y0,x1,y1,S = roi_rect
    global PLANE_N, PLANE_P0, BASE_MU, BASE_SIG, X_MUL, Y_MUL
    if PLANE_N is None or PLANE_P0 is None or BASE_MU is None or BASE_SIG is None:
        return None

    # recompute intr grid if needed (safety)
    if X_MUL.shape[0] < y1 or X_MUL.shape[1] < x1:
        return None

    P3 = deproject_u16(u16_med_roi, depth_scale,
                       X_MUL[y0:y1,x0:x1], Y_MUL[y0:y1,x0:x1])

    band_h0 = int(S*0.85)
    Sband = sdist_map(P3[band_h0:], PLANE_N, PLANE_P0)
    med_off = np.nanmedian(Sband[np.isfinite(Sband)])
    p0 = PLANE_P0 + (PLANE_N * float(med_off) if np.isfinite(med_off) else 0.0)

    s_map = sdist_map(P3, PLANE_N, p0)
    mu = BASE_MU; sg = BASE_SIG
    h_lo = np.maximum(H_LO_FLOOR, 3.0*sg)
    h_hi = np.maximum(H_HI_FLOOR, 5.0*sg)
    Eh_lo = (s_map > h_lo).astype(np.uint8)*255
    Eh_hi = (s_map > h_hi).astype(np.uint8)*255

    Z = u16_med_roi.astype(np.float32)*depth_scale
    Eb = (np.abs(Z - mu) > (K_B*sg)).astype(np.uint8)*255

    gx = cv2.Sobel(Z, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Z, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)
    tau_g = np.maximum(H_HI_FLOOR, K_G*sg)
    Eg = (grad > tau_g).astype(np.uint8)*255

    seed = cv2.bitwise_or(Eh_hi, cv2.bitwise_and(Eg, Eb))
    grown = hysteresis_region(Eh_lo, seed)

    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_K)
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_K)
    mask = cv2.morphologyEx(grown, cv2.MORPH_OPEN,  k3, iterations=1)
    mask = cv2.morphologyEx(mask,  cv2.MORPH_CLOSE, k5, iterations=CLOSE_ITER)

    if ROI_EDGE_EROSION>0:
        rim = np.ones((S,S), np.uint8)
        cv2.rectangle(rim, (0,0),(S-1,S-1), 0, thickness=ROI_EDGE_EROSION)
        mask = cv2.bitwise_and(mask, rim)

    comp = largest_external_component(mask)
    if comp is None or comp[1] < MIN_OBJ_PIX:
        return None
    comp_mask = comp[0]

    ys, xs = np.where(comp_mask>0)
    P_obj = P3[ys, xs, :]
    s_obj = s_map[ys, xs]
    Hm = float(np.nanpercentile(s_obj, 95))

    u, v = plane_axes_from_normal(PLANE_N)
    P_proj = P_obj - np.outer(s_obj, PLANE_N)
    U = P_proj.dot(u); V = P_proj.dot(v)
    UV = np.stack([U,V], axis=1).astype(np.float32)
    hull = cv2.convexHull(UV.reshape(-1,1,2))
    rect = cv2.minAreaRect(hull)
    (_, _), (wrect, hrect), _ = rect
    Lm = float(max(wrect, hrect)); Wm = float(min(wrect, hrect))

    Lmm = (Lm*1000.0) + SAFETY_PAD_MM
    Wmm = (Wm*1000.0) + SAFETY_PAD_MM
    Hmm = (Hm*1000.0) + SAFETY_PAD_MM
    Lmm, Wmm = (max(Lmm, Wmm), min(Lmm, Wmm))

    edge = cv2.Canny(comp_mask, 40, 120)
    return dict(edge=edge, Lmm=Lmm, Wmm=Wmm, Hmm=Hmm)

# ===== Main =====
def main():
    global X_MUL, Y_MUL, PLANE_N, PLANE_P0, BASE_MU, BASE_SIG, BASE_ROI_RECT, HAVE_BASE
    global DISPLAY_SCALE, FULLSCREEN, AUTO_TRIGGER, CROP_ONLY, SHOW_MASK
    global ROI_W_PX_REF, ROI_H_PX_REF
    global CALI_ACTIVE, CALI_STACK_BUF, CALI_T0, CALI_LAST_MSG, CALI_LAST_FN
    global LAST_U16, LAST_FN, LAST_TS_MS

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    pipe = rs.pipeline()
    profile = start_depth_pipeline(pipe, fps=FPS)
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
    depth_scale = float(depth_sensor.get_depth_scale())

    # Intrinsics cache
    vprof = pipe.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile()
    intr  = vprof.get_intrinsics()
    X_MUL, Y_MUL = precompute_intr_grid(intr)
    w, h = intr.width, intr.height

    # Threshold filter
    thr = rs.threshold_filter()
    thr.set_option(rs.option.min_distance, float(MIN_Z))
    thr.set_option(rs.option.max_distance, float(MAX_Z))

    poll_fail_sleep = 0.003
    spinner_idx = 0

    print("Keys: q quit | f full | +/- scale | [ ] ROI | c/r CAL start/stop | t measure | a auto | x crop | m mask")
    t_prev = time.time(); fps_ema = 0.0
    last_meas_frames = 0

    try:
        while True:
            # 1) Poll one frame per loop (non-blocking)
            frames = pipe.poll_for_frames()
            if frames:
                d = frames.get_depth_frame()
                if d:
                    if USE_THRESHOLD:
                        try: d = thr.process(d)
                        except Exception: pass
                    u16 = np.asanyarray(d.as_frame().get_data())
                    fn  = d.get_frame_number()
                    ts  = d.get_timestamp()
                    LAST_U16, LAST_FN, LAST_TS_MS = u16, fn, ts
                    if CALI_ACTIVE and fn != CALI_LAST_FN:
                        CALI_STACK_BUF.append(u16.copy())
                        CALI_LAST_FN = fn
            else:
                time.sleep(poll_fail_sleep)

            if LAST_U16 is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')): break
                continue

            u16 = LAST_U16
            w = u16.shape[1]; h = u16.shape[0]

            x0,y0,x1,y1,S,dx,dy = center_roi_scaled(w,h)
            roi_rect = (x0,y0,x1,y1,S)
            roi_u = u16[y0:y1, x0:x1]
            need_cal = (not HAVE_BASE) or (BASE_ROI_RECT is None) or (BASE_ROI_RECT[4] != S)

            vis_u8 = (np.clip(u16, 0, 4000)/4000*255).astype(np.uint8)
            vis_rgb = apply_colormap_u8(vis_u8)

            triggered = False
            if AUTO_TRIGGER and HAVE_BASE and (BASE_ROI_RECT[4]==S):
                Z = roi_u.astype(np.float32)*depth_scale
                Eb = (np.abs(Z - BASE_MU) > (K_B*BASE_SIG)).astype(np.uint8)
                triggered = Eb.mean() > 0.02

            # 2) Key handling
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('f'):
                FULLSCREEN = not FULLSCREEN
                cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if FULLSCREEN else cv2.WINDOW_NORMAL)
            elif key in (ord('+'), ord('=')): DISPLAY_SCALE = min(3.0, DISPLAY_SCALE + 0.1)
            elif key in (ord('-'), ord('_')): DISPLAY_SCALE = max(1.0, DISPLAY_SCALE - 0.1)
            elif key == ord('x'): CROP_ONLY = not CROP_ONLY
            elif key == ord('m'): SHOW_MASK = not SHOW_MASK
            elif key == ord('a'): AUTO_TRIGGER = not AUTO_TRIGGER
            elif key in (ord('c'), ord('r')):
                if not CALI_ACTIVE:
                    CALI_ACTIVE = True
                    CALI_STACK_BUF = []
                    CALI_T0 = time.time()
                    CALI_LAST_MSG = "CAL RUN"
                    CALI_LAST_FN = LAST_FN
                else:
                    CALI_ACTIVE = False
                    CALI_LAST_MSG = "CAL CANCELED"
            elif key == ord('['):
                ROI_W_PX_REF = max(8, ROI_W_PX_REF - 10); ROI_H_PX_REF = ROI_W_PX_REF
            elif key == ord(']'):
                ROI_W_PX_REF = min(ROI_REF_W, ROI_W_PX_REF + 10); ROI_H_PX_REF = ROI_W_PX_REF

            # 3) Calibration state update/finalize
            cali_txt = ""
            if CALI_ACTIVE:
                elapsed = time.time() - CALI_T0
                spinner_idx = (spinner_idx + 1) % len(SPINNER_CHARS)
                cali_txt = f"CAL RUN {SPINNER_CHARS[spinner_idx]}  {len(CALI_STACK_BUF)}/{CALI_TARGET_STACK}  {int(elapsed)}s/{int(CALI_TIMEOUT_SEC)}s"
                if (elapsed >= CALI_TIMEOUT_SEC) or (len(CALI_STACK_BUF) >= CALI_TARGET_STACK):
                    stack = CALI_STACK_BUF[:]
                    CALI_ACTIVE = False
                    if len(stack) < CALI_MIN_STACK:
                        HAVE_BASE = False
                        CALI_LAST_MSG = "CAL FAIL: few frames"
                    else:
                        med = np.median(np.stack(stack,0), axis=0).astype(np.uint16)
                        band_y0 = int(h*(1.0-BOTTOM_ROI_RATIO))
                        band_u16 = med[band_y0:h:PLANE_SAMPLE_STRIDE, ::PLANE_SAMPLE_STRIDE]
                        P3_band = deproject_u16(band_u16, depth_scale,
                                                X_MUL[band_y0:h:PLANE_SAMPLE_STRIDE, ::PLANE_SAMPLE_STRIDE],
                                                Y_MUL[band_y0:h:PLANE_SAMPLE_STRIDE, ::PLANE_SAMPLE_STRIDE])
                        pts = P3_band.reshape(-1,3)
                        valid = np.isfinite(pts).all(axis=1); pts = pts[valid]
                        if pts.shape[0] > 20000:
                            pts = pts[rng.choice(pts.shape[0], 20000, replace=False)]
                        res = fit_plane_ransac(pts, iters=400, tau=PLANE_TAU, min_inliers=PLANE_MIN_INL)
                        if not res:
                            HAVE_BASE = False
                            CALI_LAST_MSG = "CAL FAIL: plane"
                        else:
                            n, p0 = res
                            PLANE_N, PLANE_P0 = n, p0
                            x0b,y0b,x1b,y1b,Sb,_,_ = center_roi_scaled(w,h)
                            BASE_ROI_RECT = (x0b,y0b,x1b,y1b,Sb)
                            roi_stack = [s[y0b:y1b, x0b:x1b].astype(np.float32)*depth_scale for s in stack]
                            roi_arr = np.stack(roi_stack, 0)
                            mu = np.median(roi_arr, axis=0)
                            mad = np.median(np.abs(roi_arr - mu[None,...]), axis=0) + 1e-9
                            sig = 1.4826 * mad
                            BASE_MU = mu; BASE_SIG = sig
                            HAVE_BASE = True
                            CALI_LAST_MSG = f"CAL OK: {len(stack)} frames"
            elif CALI_LAST_MSG:
                cali_txt = CALI_LAST_MSG

            # 4) Measurement trigger
            meas_txt = ""; pack_txt = ""; warn_txt = ""
            do_measure = triggered or (key == ord('t'))
            if need_cal:
                warn_txt = "NEED CAL: press [c]"
            if do_measure and HAVE_BASE and not need_cal and (PLANE_N is not None):
                t_end = time.time() + MEAS_TIMEOUT_SEC
                stack = []
                start_fn = LAST_FN
                while (len(stack) < MEAS_STACK_N) and (time.time() < t_end):
                    if LAST_FN != start_fn and LAST_U16 is not None:
                        stack.append(LAST_U16[y0:y1, x0:x1].copy())
                        start_fn = LAST_FN
                    cv2.waitKey(1)
                last_meas_frames = len(stack)
                if last_meas_frames >= MEAS_MIN_STACK:
                    u16_med_roi = np.median(np.stack(stack,0), axis=0).astype(np.uint16)
                    res = segment_and_measure(u16_med_roi, roi_rect, depth_scale)
                    if res:
                        Lmm, Wmm, Hmm = res["Lmm"], res["Wmm"], res["Hmm"]
                        meas_txt = f"L={Lmm:.1f}mm  W={Wmm:.1f}mm  H={Hmm:.1f}mm"
                        rec = recommend_pack(Lmm, Wmm, Hmm)
                        if rec:
                            orient = "L-width" if rec["orientation"]=="A" else "W-width"
                            pack_txt = f"ROLL {rec['roll_width_mm']}mm ({orient}) / CUT {rec['cut_length_mm']:.0f}mm"
                        if SHOW_MASK:
                            sub = vis_rgb[y0:y1, x0:x1]
                            sub[res["edge"]>0] = (0,255,255)
                    else:
                        meas_txt = "no object / weak signal"
                else:
                    meas_txt = "measure stack too small"

            # 5) Draw
            cv2.rectangle(vis_rgb, (x0,y0),(x1,y1), (0,200,255), 2)
            wv,hv = vis_rgb.shape[1], vis_rgb.shape[0]
            cv2.drawMarker(vis_rgb, (wv//2,hv//2), (255,255,255), cv2.MARKER_CROSS, 18, 1)

            info = np.zeros((INFO_BAR_H, vis_rgb.shape[1], 3), dtype=np.uint8)
            def put(img, text, x, y, color=(255,255,255)):
                cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)

            put(info, "c/r:cal  t:measure  a:auto  f:full  +/-:scale  [ ]:ROI  x:crop  m:mask  q:quit", 10, 24)
            put(info, f"ROI {S}x{S}px  dx={dx} dy={dy}  auto={'ON' if AUTO_TRIGGER else 'OFF'}", 10, 48, (200,255,200))

            right_x = vis_rgb.shape[1] - 380
            if cali_txt:
                put(info, cali_txt, right_x, 24, (0,200,255))
                if CALI_ACTIVE:
                    ratio = min(1.0, len(CALI_STACK_BUF)/float(CALI_TARGET_STACK))
                    draw_progress_bar(info, right_x, 28, 220, 10, ratio)
            if last_meas_frames>0:
                put(info, f"meas_stack={last_meas_frames}", right_x, 48, (200,200,255))
            if meas_txt: put(info, meas_txt, 10, 68)
            if pack_txt: put(info, pack_txt, 10 + vis_rgb.shape[1]//2, 68, (0,255,0))
            if warn_txt: put(info, warn_txt, 10, 24, (0,200,255))

            view = vis_rgb[y0:y1, x0:x1] if CROP_ONLY else vis_rgb
            vis = np.vstack([view, info])
            if DISPLAY_SCALE != 1.0:
                vis = cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST)

            t_now = time.time(); inst = 1.0/max(1e-6, t_now - t_prev); t_prev = t_now
            fps_ema = 0.9*fps_ema + 0.1*inst if fps_ema>0 else inst
            cv2.rectangle(vis, (8,8), (230,30), (0,0,0), -1)
            cv2.putText(vis, f"loop~{fps_ema:.1f} fps", (14,24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)

            cv2.imshow(WIN_NAME, vis)

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
