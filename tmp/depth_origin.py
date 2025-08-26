# conveyor_measure_pack.py
# D435f 깊이 기반 컨베이어 물체 치수 측정 + 포장재 추천 (정지형 테스트 + 손잡이 보강)
# - Height: 바닥 기준 +s, p95 퍼센타일 (max 금지)
# - L/W: 로버스트 PCA + 퍼센타일 폭(Q98-Q02) + RGB 에지 보강
# - 북쪽 방향 누락 대응: IR Emitter ON/OFF 2샷 s_map nanmedian 융합
# - 디버그: 'i'로 IR 좌/우 뷰 오버레이, 'e'로 2샷 융합 ON/OFF 토글
# 조작: c/r=평면 캘리브 | f=전체화면 | i=IR뷰 | e=2샷융합 | +/-=배율 | [ ]=ROI 크기 | q=종료

import pyrealsense2 as rs
import numpy as np
import cv2, time, os

# ===== Pillow 한글 텍스트 =====
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
FONT_CACHE = {}
def _get_font(size=20):
    if size in FONT_CACHE:
        return FONT_CACHE[size]
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size); FONT_CACHE[size] = f; return f
            except Exception: continue
    FONT_CACHE[size] = ImageFont.load_default(); return FONT_CACHE[size]

def draw_text(img_bgr, text, x, y, size=20, color=(255,255,255), stroke=2, stroke_color=(0,0,0)):
    if PIL_OK:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(img_rgb); draw = ImageDraw.Draw(pil_im)
        font = _get_font(size); pos = (x, y - int(size*0.75))
        rgb = (int(color[2]), int(color[1]), int(color[0])); srgb = (int(stroke_color[2]), int(stroke_color[1]), int(stroke_color[0]))
        try: draw.text(pos, text, font=font, fill=rgb, stroke_width=stroke, stroke_fill=srgb)
        except Exception:
            ascii_text = text.encode('utf-8','ignore').decode('ascii','ignore')
            draw.text(pos, ascii_text, font=font, fill=rgb, stroke_width=stroke, stroke_fill=srgb)
        return cv2.cvtColor(np.asarray(pil_im), cv2.COLOR_RGB2BGR)
    else:
        ascii_text = text.encode('utf-8','ignore').decode('ascii','ignore')
        cv2.putText(img_bgr, ascii_text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, size/32.0), color, 1, cv2.LINE_AA); return img_bgr

rng = np.random.default_rng(42)

# ===== 해상도/프레임 =====
W, H, FPS = 1280, 720, 6
WIN_NAME = "Conveyor Measure + Packing"
DISPLAY_SCALE = 1.5
FULLSCREEN = False
INFO_BAR_H = 64

# ===== ROI =====
ROI_RATIO = 0.60
ROI_MIN, ROI_MAX = 0.30, 0.95
ROI_STEP = 0.05
ROI_W_PX = 230
ROI_H_PX = 230
DX_PX, DY_PX = 20, -60

# ===== 포장재 =====
ROLL_WIDTHS_MM = [300, 400, 500, 600]
EDGE_MARGIN_MM, OVERLAP_MM, SAFETY_PAD_MM = 20, 30, 5

# ===== 파라미터 =====
DECIM = 1
PLANE_TAU = 0.004
H_MIN_BASE = 0.003
H_MAX = 0.40
MIN_OBJ_PIX = 40
BOTTOM_ROI_RATIO = 0.20
HOLE_FILL = False
CORE_MARGIN_PX = 1
P_LO, P_HI = 2.0, 98.0

# ===== 보강 옵션 =====
FUSE_TWO_SHOTS = True    # 기본: 2샷 융합 켜기
FUSE_MODE = "nanmedian"  # "nanmedian" | "nanmax" | "p80"
SHOW_IR_DEBUG = False    # 'i'로 토글

def apply_colormap_u8(gray):
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray, cmap)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def compute_roi_indices(w, h):
    global ROI_W_PX, ROI_H_PX, DX_PX, DY_PX, ROI_RATIO
    if ROI_W_PX > 0 and ROI_H_PX > 0:
        rw = clamp(ROI_W_PX, 8, min(w, h)); rh = clamp(ROI_H_PX, 8, min(w, h)); S = int(min(rw, rh))
    else:
        S = int(min(w, h) * clamp(ROI_RATIO, ROI_MIN, ROI_MAX))
    cx = w // 2 + DX_PX; cy = h // 2 + DY_PX
    x0 = clamp(cx - S // 2, 0, w - S); y0 = clamp(cy - S // 2, 0, h - S)
    return x0, x0 + S, y0, y0 + S, S

def orient_normal_to_camera(n):
    cam_dir = np.array([0., 0., -1.], dtype=np.float32)
    return n if np.dot(n, cam_dir) > 0 else -n

def fit_plane_ransac(P, iters=300, tau=PLANE_TAU, min_inliers=1200):
    N = P.shape[0]
    if N < 3: return None
    best_mask=None; best_n=None; best_p0=None
    for _ in range(iters):
        ids = rng.choice(N, size=3, replace=False); A,B,C = P[ids]
        n = np.cross(B-A, C-A); nn = np.linalg.norm(n)
        if nn < 1e-8: continue
        n = n/nn; d = -np.dot(n, A); dist = np.abs(P.dot(n) + d); mask = dist < tau
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask = mask; best_n = n; best_p0 = A
    if best_mask is None or best_mask.sum() < min_inliers: return None
    Pin = P[best_mask]; c = Pin.mean(axis=0)
    U,S,Vt = np.linalg.svd(Pin - c, full_matrices=False)
    n_ref = Vt[-1]; n_ref /= (np.linalg.norm(n_ref)+1e-12)
    n_ref = orient_normal_to_camera(n_ref)
    return n_ref, c

def plane_axes_from_normal(n):
    t = np.array([1.,0.,0.]) if abs(n[0]) < 0.9 else np.array([0.,0.,1.])
    u = np.cross(n, t); u /= (np.linalg.norm(u)+1e-12)
    v = np.cross(n, u); v /= (np.linalg.norm(v)+1e-12)
    return u, v

def signed_distance_map(P3, plane_n, plane_p0):
    S = np.einsum('ijk,k->ij', P3 - plane_p0, plane_n).astype(np.float32)
    invalid = ~np.isfinite(P3).all(axis=2); S[invalid] = np.nan
    return S

def dynamic_h_min(s_map, base=H_MIN_BASE):
    s_valid = s_map[np.isfinite(s_map)]
    if s_valid.size < 500: return base
    near = s_valid[np.abs(s_valid) < 0.05]; 
    if near.size < 500: near = s_valid
    med = np.median(near); mad = np.median(np.abs(near - med)) + 1e-9
    sigma = 1.4826 * mad; thr = max(base, med + max(0.002, 1.3*sigma))
    return float(thr)

def object_mask_from_height(s_map, h_min, h_max):
    mask = (s_map > h_min) & (s_map < h_max) & np.isfinite(s_map)
    mask = mask.astype(np.uint8)*255
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)
    return mask

def largest_external_component(mask):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea); filled = np.zeros_like(mask)
    cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
    return filled, int(cv2.contourArea(c))

def erode_core(mask, rpx=CORE_MARGIN_PX):
    if mask is None: return None
    dt = cv2.distanceTransform((mask>0).astype(np.uint8), cv2.DIST_L2, 3)
    core = (dt > float(rpx)).astype(np.uint8)*255
    if core.sum() < mask.sum()*0.2:
        core = (dt > max(1, rpx-1)).astype(np.uint8)*255
    return core

def robust_pca_lengths(UV):
    mu = UV.mean(axis=0, keepdims=True); X = UV - mu
    if X.shape[0] < 3:
        p1, p2 = X[:,0], X[:,1]
        L = float(np.nanpercentile(p1, P_HI) - np.nanpercentile(p1, P_LO))
        W = float(np.nanpercentile(p2, P_HI) - np.nanpercentile(p2, P_LO))
        evecs = np.eye(2, dtype=np.float32); return L, W, mu[0], evecs
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    evecs = Vt.T.astype(np.float32); P = X.dot(evecs)
    p1, p2 = P[:,0], P[:,1]
    L = float(np.nanpercentile(p1, P_HI) - np.nanpercentile(p1, P_LO))
    W = float(np.nanpercentile(p2, P_HI) - np.nanpercentile(p2, P_LO))
    if W > L: L, W = W, L; evecs = evecs[:, ::-1]
    return L, W, mu[0], evecs

def fuse_smaps(S_list, h_min=H_MIN_BASE, h_max=0.20, mode=FUSE_MODE):
    if not S_list: return None
    S_stack = np.stack(S_list, axis=0).astype(np.float32)
    S_stack[~np.isfinite(S_stack)] = np.nan
    S_stack[(S_stack < h_min) | (S_stack > h_max)] = np.nan
    if mode == "nanmedian": return np.nanmedian(S_stack, axis=0)
    if mode == "nanmax":    return np.nanmax(S_stack, axis=0)
    return np.nanpercentile(S_stack, 80, axis=0)

def rgb_edge_mask(rgb_roi):
    gray = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 30, 5)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    return edges

def measure_lwh(points3d, plane_n, plane_p0, s_external=None, rgb_roi=None):
    # s_map
    s = s_external if s_external is not None else signed_distance_map(points3d, plane_n, plane_p0)
    h_min_dyn = dynamic_h_min(s, base=H_MIN_BASE)

    # 물체 마스크(+s)
    raw_mask = object_mask_from_height(s, h_min_dyn, H_MAX)
    if raw_mask is None: return None, None, None, None, None, None, h_min_dyn
    lg = largest_external_component(raw_mask)
    if lg is None or lg[1] < MIN_OBJ_PIX: return None, None, None, None, None, None, h_min_dyn
    comp = lg[0]

    # 높이: p95
    ysH, xsH = np.where(comp > 0); s_obj_full = s[ysH, xsH]
    s_vals = s_obj_full[np.isfinite(s_obj_full)]
    if s_vals.size < 5: return None, None, None, None, None, None, h_min_dyn
    H_obj = float(np.nanpercentile(s_vals, 95))

    # L/W: 깊이 코어 + (옵션) RGB 에지 보강
    comp_core = erode_core(comp, rpx=CORE_MARGIN_PX)
    if comp_core is None or comp_core.sum() < MIN_OBJ_PIX: comp_core = comp

    # 깊이 코어 포인트
    ys, xs = np.where(comp_core > 0)
    P_obj = points3d[ys, xs, :]; s_obj = s[ys, xs]
    valid_mask = ~np.isnan(P_obj).any(axis=1) & np.isfinite(s_obj)
    P_obj = P_obj[valid_mask]; s_obj = s_obj[valid_mask]

    # 평면 투영
    u, v = plane_axes_from_normal(plane_n)
    P_proj = P_obj - np.outer(s_obj, plane_n)
    U = P_proj.dot(u); V = P_proj.dot(v)
    UV = np.stack([U, V], axis=1).astype(np.float32)

    # RGB 에지로 보강 (유효 깊이가 있는 에지 픽셀만 추가)
    if rgb_roi is not None:
        e = rgb_edge_mask(rgb_roi)
        ysE, xsE = np.where(e > 0)
        if ysE.size > 0:
            P_edge = points3d[ysE, xsE, :]
            valid_e = np.isfinite(P_edge).all(axis=1)
            if np.count_nonzero(valid_e) > 10:
                P_edge = P_edge[valid_e]
                # 평면 투영
                s_edge = np.einsum('ij,j->i', P_edge - plane_p0, plane_n)
                P_edge_proj = P_edge - np.outer(s_edge, plane_n)
                Ue = P_edge_proj.dot(u); Ve = P_edge_proj.dot(v)
                UV_edge = np.stack([Ue, Ve], axis=1).astype(np.float32)
                UV = np.vstack([UV, UV_edge])

    if UV.shape[0] < 3: return None, None, H_obj, None, None, comp, h_min_dyn

    # 로버스트 PCA + 퍼센타일 폭
    L, W_, mu, E = robust_pca_lengths(UV)

    # 박스
    halfL, halfW = L/2.0, W_/2.0
    corners_local = np.array([[+halfL, +halfW],[+halfL, -halfW],[-halfL, -halfW],[-halfL, +halfW]], dtype=np.float32)
    boxUV = corners_local.dot(E.T) + mu

    return L, W_, H_obj, UV, boxUV.astype(np.float32), comp, h_min_dyn

def recommend_pack(L_mm, W_mm, H_mm, roll_list_mm, edge_margin=20, overlap=30, pad=5):
    need_w_A = L_mm + 2*H_mm + 2*edge_margin + pad
    cut_A    = 2*(W_mm + H_mm) + overlap + pad
    need_w_B = W_mm + 2*H_mm + 2*edge_margin + pad
    cut_B    = 2*(L_mm + H_mm) + overlap + pad
    def pick_roll(need):
        cands = [rw for rw in roll_list_mm if rw >= need]
        return (min(cands) if cands else None)
    roll_A = pick_roll(need_w_A); roll_B = pick_roll(need_w_B)
    options = []
    if roll_A is not None:
        waste_A = roll_A - need_w_A; score_A = cut_A + waste_A*0.2
        options.append(("A", roll_A, cut_A, need_w_A, score_A))
    if roll_B is not None:
        waste_B = roll_B - need_w_B; score_B = cut_B + waste_B*0.2
        options.append(("B", roll_B, cut_B, need_w_B, score_B))
    if not options: return None
    orient, roll, cutlen, needw, _ = min(options, key=lambda x: x[-1])
    return dict(orientation=orient, roll_width_mm=int(round(roll)),
                required_width_mm=float(needw), cut_length_mm=float(cutlen))

def grab_smap(pipe, depth_sensor, pc, align, plane_n, plane_p0, roi_xyxy, emitter_on):
    """ Emitter 상태를 지정해 1프레임 캡처하고 ROI s_map, 3D, RGB ROI, IR(옵션) 반환 """
    # 설정
    try:
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if emitter_on else 0.0)
    except Exception: pass

    # 안정화 프레임 1장 버림
    _ = pipe.wait_for_frames()
    frames = pipe.wait_for_frames()
    frames = align.process(frames)

    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    ir1 = frames.get_infrared_frame(1) if frames.get_infrared_frame(1) else None
    ir2 = frames.get_infrared_frame(2) if frames.get_infrared_frame(2) else None
    if not depth: return None, None, None, None, None

    # 포인트클라우드 (depth 기준)
    points = pc.calculate(depth)
    verts = np.asanyarray(points.get_vertices()).view(np.float32)
    vprof = depth.get_profile().as_video_stream_profile()
    intr  = vprof.get_intrinsics()
    w, h  = intr.width, intr.height
    P3_full = verts.reshape(h, w, 3)
    P3_full[P3_full[:,:,2] == 0] = np.nan

    x0,x1,y0,y1 = roi_xyxy
    P3_roi = P3_full[y0:y1, x0:x1, :]

    # s_map + 하단 밴드 median=0 보정
    S = signed_distance_map(P3_roi, plane_n, plane_p0)
    band_h0 = int(P3_roi.shape[0]*(1.0 - BOTTOM_ROI_RATIO))
    Sband = S[band_h0:, :]
    med = np.nanmedian(Sband[np.isfinite(Sband)])
    if np.isfinite(med): S = S - float(med)

    # RGB ROI (color는 align(depth)되어 있음)
    rgb_roi = None
    if color:
        cimg = np.asanyarray(color.get_data())
        rgb_roi = cimg[y0:y1, x0:x1].copy()

    # IR (디버그용)
    ir1_img = np.asanyarray(ir1.get_data()) if ir1 else None
    ir2_img = np.asanyarray(ir2.get_data()) if ir2 else None

    return S, P3_roi, rgb_roi, ir1_img, ir2_img

def main():
    global DISPLAY_SCALE, FULLSCREEN, ROI_RATIO, ROI_W_PX, ROI_H_PX, DX_PX, DY_PX
    global FUSE_TWO_SHOTS, SHOW_IR_DEBUG

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)          # RGB
    cfg.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)       # IR Left
    cfg.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)       # IR Right
    profile = pipe.start(cfg)

    # align color/IR to depth (depth 좌표계 기준)
    align = rs.align(rs.stream.depth)

    # 센서 옵션
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    try: depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
    except Exception: pass
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
    try:
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
        if depth_sensor.supports(rs.option.exposure):
            depth_sensor.set_option(rs.option.exposure, 2000.0)
        if depth_sensor.supports(rs.option.gain):
            depth_sensor.set_option(rs.option.gain, 18.0)
    except Exception: pass

    try:
        depth_scale = depth_sensor.get_depth_scale()
    except Exception:
        depth_scale = 0.001  # m

    # 필터 (간결하게: 포인트클라우드는 내부적으로 disparity 변환 없이도 잘 동작)
    dec  = rs.decimation_filter(DECIM)
    spat = rs.spatial_filter(); temp = rs.temporal_filter()
    try:
        spat.set_option(rs.option.filter_magnitude, 2)
        spat.set_option(rs.option.filter_smooth_alpha, 0.45)
        spat.set_option(rs.option.filter_smooth_delta, 18)
        temp.set_option(rs.option.filter_smooth_alpha, 0.35)
        temp.set_option(rs.option.filter_smooth_delta, 18)
    except Exception: pass
    hole = rs.hole_filling_filter(1) if HOLE_FILL else None
    pc = rs.pointcloud()

    have_plane = False
    plane_n = None; plane_p0 = None
    band_med_mm = None

    print("c: 평면 캘리브(빈 벨트) | r: 재캘리브 | f: 전체 | i: IR뷰 | e: 2샷융합 | +/-: 배율 | [ ]: ROI | q: 종료")

    try:
        while True:
            t0 = time.time()

            # 원 프레임 얻어 시각화용 depth/color 생성 (빠른 미리보기)
            frames = pipe.wait_for_frames()
            frames = align.process(frames)
            depth  = frames.get_depth_frame()
            color  = frames.get_color_frame()
            if not depth: continue

            # 간단 시각화 depth
            depth_np_full = np.asanyarray(depth.get_data())
            depth_vis_full = (np.clip(depth_np_full, 0, 4000)/4000*255).astype(np.uint8)
            depth_vis_full = apply_colormap_u8(depth_vis_full)

            # ROI
            vprof = depth.get_profile().as_video_stream_profile()
            intr  = vprof.get_intrinsics()
            w, h  = intr.width, intr.height
            x0, x1, y0, y1, S = compute_roi_indices(w, h)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('f'):
                FULLSCREEN = not FULLSCREEN
                cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if FULLSCREEN else cv2.WINDOW_NORMAL)
            elif key in (ord('+'), ord('=')): DISPLAY_SCALE = min(3.0, DISPLAY_SCALE + 0.1)
            elif key in (ord('-'), ord('_')): DISPLAY_SCALE = max(1.0, DISPLAY_SCALE - 0.1)
            elif key == ord('['):
                if ROI_W_PX > 0 and ROI_H_PX > 0:
                    ROI_W_PX = ROI_H_PX = max(8, min(min(w,h), S - 10))
                else: ROI_RATIO = max(ROI_MIN, ROI_RATIO - ROI_STEP)
            elif key == ord(']'):
                if ROI_W_PX > 0 and ROI_H_PX > 0:
                    ROI_W_PX = ROI_H_PX = max(8, min(min(w,h), S + 10))
                else: ROI_RATIO = min(ROI_MAX, ROI_RATIO + ROI_STEP)
            elif key in (ord('c'), ord('r')):
                # 빈 벨트에서 ROI 하단으로 평면 추정
                # 한 번 더 최신 프레임 가져와서 dec/spat/temp 적용
                d2 = dec.process(depth); d2 = spat.process(d2); d2 = temp.process(d2)
                if hole is not None: d2 = hole.process(d2)
                points = pc.calculate(d2)
                verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(h, w, 3)
                verts[verts[:,:,2] == 0] = np.nan
                P3 = verts[y0:y1, x0:x1, :]
                ch, cw = P3.shape[:2]
                roi_h0 = int(ch*(1.0 - BOTTOM_ROI_RATIO))
                roi = P3[roi_h0:ch, :, :].reshape(-1,3)
                valid = np.isfinite(roi).all(axis=1); roi = roi[valid]
                if roi.shape[0] > 8000: roi = roi[rng.choice(roi.shape[0], 8000, replace=False)]
                res = fit_plane_ransac(roi, iters=300, tau=PLANE_TAU, min_inliers=1200)
                if res is not None:
                    plane_n, plane_p0 = res; have_plane = True
                    print(f"평면 캘리브 완료. n={plane_n}, p0={plane_p0}")
                else:
                    print("평면 추정 실패. 바닥만 보이게 하고 다시 c를 눌러주세요.")
            elif key == ord('i'):
                SHOW_IR_DEBUG = not SHOW_IR_DEBUG
            elif key == ord('e'):
                FUSE_TWO_SHOTS = not FUSE_TWO_SHOTS

            txt_top = "c:캘리브 r:재캘리브 f:전체 i:IR e:2샷융합 +/-:[ ]:ROI q:종료"
            meas_txt = ""; pack_txt = ""; thr_txt  = ""; band_txt = ""; dbg_txt = ""

            if have_plane:
                # --- 두 샷 융합 또는 단일 샷 ---
                if FUSE_TWO_SHOTS:
                    # ON/OFF 두 장
                    S_on,  P3_on,  rgb_on,  ir1_on, ir2_on   = grab_smap(pipe, depth_sensor, pc, align, plane_n, plane_p0, (x0,x1,y0,y1), True)
                    S_off, P3_off, rgb_off, ir1_off, ir2_off = grab_smap(pipe, depth_sensor, pc, align, plane_n, plane_p0, (x0,x1,y0,y1), False)
                    # ROI 3D/RGB는 ON 기준으로 취함(동일 정지 상태 가정)
                    if S_on is not None and S_off is not None:
                        S = fuse_smaps([S_on, S_off], h_min=H_MIN_BASE, h_max=0.20, mode=FUSE_MODE)
                        P3_roi = P3_on if P3_on is not None else P3_off
                        rgb_roi = rgb_on if rgb_on is not None else rgb_off
                        dbg_txt = f"FUSED 2x ({FUSE_MODE})"
                    else:
                        # 폴백: 현재 depth 한 장에서 계산
                        d2 = dec.process(depth); d2 = spat.process(d2); d2 = temp.process(d2)
                        if hole is not None: d2 = hole.process(d2)
                        points = pc.calculate(d2)
                        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(h, w, 3)
                        verts[verts[:,:,2] == 0] = np.nan
                        P3_roi = verts[y0:y1, x0:x1, :]
                        S = signed_distance_map(P3_roi, plane_n, plane_p0)
                        rgb_roi = np.asanyarray(color.get_data())[y0:y1, x0:x1] if color else None
                        dbg_txt = "FUSE FAIL → single"
                else:
                    d2 = dec.process(depth); d2 = spat.process(d2); d2 = temp.process(d2)
                    if hole is not None: d2 = hole.process(d2)
                    points = pc.calculate(d2)
                    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(h, w, 3)
                    verts[verts[:,:,2] == 0] = np.nan
                    P3_roi = verts[y0:y1, x0:x1, :]
                    S = signed_distance_map(P3_roi, plane_n, plane_p0)
                    rgb_roi = np.asanyarray(color.get_data())[y0:y1, x0:y1] if color else None
                    dbg_txt = "Single shot"

                # 하단 밴드 median=0 보정(한 번 더)
                if S is not None:
                    band_h0 = int(S.shape[0] * (1.0 - BOTTOM_ROI_RATIO))
                    med = np.nanmedian(S[band_h0:][np.isfinite(S[band_h0:])])
                    if np.isfinite(med): S = S - float(med); band_med_mm = med*1000.0
                    else: band_med_mm = None

                # 최종 측정 (s_external + rgb 윤곽 보강)
                if S is not None and P3_roi is not None:
                    L, W_, H_obj, UV, boxUV, objmask, h_min_dyn = measure_lwh(P3_roi, plane_n, plane_p0, s_external=S, rgb_roi=rgb_roi)
                else:
                    L=W_=H_obj=UV=boxUV=objmask=h_min_dyn=None

                thr_txt = f"h_min≈{(h_min_dyn or 0)*1000:.0f} mm | ROI={S.shape[1]}x{S.shape[0]} | {dbg_txt}"
                if band_med_mm is not None: band_txt = f"바닥중앙값≈{band_med_mm:+.1f} mm"

                if L is not None:
                    Lmm = (L*1000.0) + SAFETY_PAD_MM; Wmm = (W_*1000.0) + SAFETY_PAD_MM; Hmm = (H_obj*1000.0) + SAFETY_PAD_MM
                    Lmm, Wmm = (max(Lmm, Wmm), min(Lmm, Wmm))
                    meas_txt = f"L={Lmm:.1f} mm, W={Wmm:.1f} mm, H={Hmm:.1f} mm"
                    rec = recommend_pack(Lmm, Wmm, Hmm, ROLL_WIDTHS_MM, edge_margin=EDGE_MARGIN_MM, overlap=OVERLAP_MM, pad=SAFETY_PAD_MM)
                    if rec:
                        orient = "L-폭기준" if rec["orientation"]=="A" else "W-폭기준"
                        pack_txt = f"롤폭 {rec['roll_width_mm']} mm ({orient}) / 절단 {rec['cut_length_mm']:.0f} mm"
                    else: pack_txt = "사용 가능한 롤 폭 없음(폭 확대 필요)"

                    # ROI 시각화
                    roi_vis = depth_vis_full[y0:y1, x0:x1].copy()
                    if objmask is not None:
                        edge = cv2.Canny(objmask, 40, 120); roi_vis[edge>0] = (0,255,255)
                    if boxUV is not None and boxUV.shape == (4,2):
                        nx = (boxUV[:,0] - boxUV[:,0].min()) / (boxUV[:,0].ptp()+1e-6)
                        ny = (boxUV[:,1] - boxUV[:,1].min()) / (boxUV[:,1].ptp()+1e-6)
                        cxs = np.clip(nx * (roi_vis.shape[1]-1), 0, roi_vis.shape[1]-1)
                        cys = np.clip(ny * (roi_vis.shape[0]-1), 0, roi_vis.shape[0]-1)
                        poly = np.stack([cxs, cys], axis=1).astype(np.int32)
                        cv2.polylines(roi_vis, [poly], isClosed=True, color=(0,255,0), thickness=2)
                    depth_vis_full[y0:y1, x0:x1] = roi_vis
                else:
                    meas_txt = "물체 미검출 또는 깊이 부족(ROI/각도/재질 확인)"

            # ROI 박스
            cv2.rectangle(depth_vis_full, (x0, y0), (x1, y1), (0, 200, 255), 2)

            # IR 디버그 오버레이
            if SHOW_IR_DEBUG:
                # 최신 정렬 프레임에서 IR 가져오기 (align 후에도 IR는 그대로 Y8)
                ir1 = frames.get_infrared_frame(1); ir2 = frames.get_infrared_frame(2)
                if ir1:
                    ir1_img = np.asanyarray(ir1.get_data()); ir1_v = cv2.cvtColor(ir1_img, cv2.COLOR_GRAY2BGR)
                    ir1_v = cv2.resize(ir1_v, (W//5, H//5)); depth_vis_full[10:10+ir1_v.shape[0], 10:10+ir1_v.shape[1]] = ir1_v
                    depth_vis_full = draw_text(depth_vis_full, "IR-Left", 16, 10+ir1_v.shape[0]-4, 18, (255,255,255), 2)
                if ir2:
                    ir2_img = np.asanyarray(ir2.get_data()); ir2_v = cv2.cvtColor(ir2_img, cv2.COLOR_GRAY2BGR)
                    ir2_v = cv2.resize(ir2_v, (W//5, H//5)); xoff = 20 + (W//5)
                    depth_vis_full[10:10+ir2_v.shape[0], xoff:xoff+ir2_v.shape[1]] = ir2_v
                    depth_vis_full = draw_text(depth_vis_full, "IR-Right", xoff+6, 10+ir2_v.shape[0]-4, 18, (255,255,255), 2)

            # 하단 정보바
            info = np.zeros((INFO_BAR_H, depth_vis_full.shape[1], 3), dtype=np.uint8)
            info = draw_text(info, txt_top, 10, 22, size=20, color=(255,255,255), stroke=2)
            if meas_txt: info = draw_text(info, meas_txt, 10, 46, size=20, color=(255,255,255), stroke=2)
            if pack_txt: info = draw_text(info, pack_txt, 10 + depth_vis_full.shape[1]//2, 46, size=20, color=(0,255,0), stroke=2)
            if thr_txt:  info = draw_text(info, thr_txt, depth_vis_full.shape[1]-580, 22, size=18, color=(200,200,255), stroke=2)
            if band_txt: info = draw_text(info, band_txt, depth_vis_full.shape[1]-580, 46, size=18, color=(255,220,180), stroke=2)

            vis = np.vstack([depth_vis_full, info])
            if DISPLAY_SCALE != 1.0:
                vis = cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST)

            fps = 1.0/(time.time()-t0+1e-6)
            vis = draw_text(vis, f"{fps:.1f} FPS  |  Emitter2x={'ON' if FUSE_TWO_SHOTS else 'OFF'}  |  IRdbg={'ON' if SHOW_IR_DEBUG else 'OFF'}",
                            10, vis.shape[0]-10, size=18, color=(255,255,255), stroke=2)
            cv2.imshow(WIN_NAME, vis)

    finally:
        pipe.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
