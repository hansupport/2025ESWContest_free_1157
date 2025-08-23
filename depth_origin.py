# conveyor_measure_pack.py
# D435f 하드웨어 깊이 기반 컨베이어 물체 치수 측정 + 포장재 추천 (중앙 ROI 검출판, 개선)
# - 전체 화면 표시, 중앙 정사각 ROI에서만 검출/치수 측정
# - 평면 법선 카메라쪽 정렬, s/−s 부호 자동 선택
# - 동적 임계치 완화(추가 편차 최소 2mm, 계수 1.3), 작은 물체 보존, 바닥 오프셋 미세 보정
# - 링/구멍 물체 대응: 외곽 컨투어만 채워서 측정, convex hull 기반 직사각 추정
# 조작: c/r=평면 캘리브 | f=전체화면 | +/-=배율 | [ ]=ROI 크기 | q=종료

import pyrealsense2 as rs
import numpy as np
import cv2, time

rng = np.random.default_rng(42)

# 해상도/프레임
W, H, FPS = 1280, 720, 6
WIN_NAME = "Conveyor Measure + Packing"
DISPLAY_SCALE = 1.5
FULLSCREEN = False
INFO_BAR_H = 56

# 중앙 검출 ROI 비율(정사각), '[' ']'로 조절
ROI_RATIO = 0.60
ROI_MIN, ROI_MAX = 0.30, 0.95
ROI_STEP = 0.05

# 절대 ROI(픽셀) + 오프셋(픽셀) 전역 플래그
ROI_W_PX = 230
ROI_H_PX = 230
DX_PX    = 5
DY_PX    = -60

# 포장재 설정
ROLL_WIDTHS_MM = [300, 400, 500, 600]
EDGE_MARGIN_MM = 20
OVERLAP_MM     = 30
SAFETY_PAD_MM  = 5

# 측정 파라미터
DECIM = 1
PLANE_TAU = 0.010
H_MIN_BASE = 0.005      # 3 mm
H_MAX = 1.000
MIN_OBJ_PIX = 60        # 작은 물체도 통과
BOTTOM_ROI_RATIO = 0.25

def apply_colormap_u8(gray):
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray, cmap)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def compute_roi_indices(w, h):
    global ROI_W_PX, ROI_H_PX, DX_PX, DY_PX, ROI_RATIO
    if ROI_W_PX > 0 and ROI_H_PX > 0:
        rw = clamp(ROI_W_PX, 8, min(w, h))
        rh = clamp(ROI_H_PX, 8, min(w, h))
        S = int(min(rw, rh))  # 정사각 유지
    else:
        S = int(min(w, h) * clamp(ROI_RATIO, ROI_MIN, ROI_MAX))
    cx = w // 2 + DX_PX
    cy = h // 2 + DY_PX
    x0 = clamp(cx - S // 2, 0, w - S)
    y0 = clamp(cy - S // 2, 0, h - S)
    return x0, x0 + S, y0, y0 + S, S

def orient_normal_to_camera(n):
    cam_dir = np.array([0., 0., -1.], dtype=np.float32)
    return n if np.dot(n, cam_dir) > 0 else -n

def fit_plane_ransac(P, iters=300, tau=PLANE_TAU, min_inliers=1200):
    N = P.shape[0]
    if N < 3: return None
    best_mask = None; best_n = None; best_p0 = None
    for _ in range(iters):
        ids = rng.choice(N, size=3, replace=False)
        A,B,C = P[ids]
        n = np.cross(B-A, C-A)
        nn = np.linalg.norm(n)
        if nn < 1e-8: continue
        n = n/nn
        d = -np.dot(n, A)
        dist = np.abs(P.dot(n) + d)
        mask = dist < tau
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask = mask; best_n = n; best_p0 = A
    if best_mask is None or best_mask.sum() < min_inliers:
        return None
    Pin = P[best_mask]
    c = Pin.mean(axis=0)
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
    invalid = ~np.isfinite(P3).all(axis=2)
    S[invalid] = np.nan
    return S

# 동적 임계 완화: 추가 편차 하한 2mm, 계수 1.3
def dynamic_h_min(s_map, base=H_MIN_BASE):
    s_valid = s_map[np.isfinite(s_map)]
    if s_valid.size < 500:
        return base
    near = s_valid[np.abs(s_valid) < 0.05]
    if near.size < 500: near = s_valid
    med = np.median(near)
    mad = np.median(np.abs(near - med)) + 1e-9
    sigma = 1.4826 * mad
    thr = max(base, med + max(0.002, 1.3*sigma))  # 2 mm, 1.3×
    return float(thr)

# 마스크 후처리 강화: open(3) + close(5, iter=2)
def object_mask_from_height(s_map, h_min, h_max):
    mask = (s_map > h_min) & (s_map < h_max) & np.isfinite(s_map)
    mask = mask.astype(np.uint8)*255
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    return mask

# 외곽(contour)만 채워서 반환(내부 구멍 무시)
def largest_external_component(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
    return filled, int(cv2.contourArea(c))

def pick_mask_by_sign(s, h_min):
    mpos = object_mask_from_height(s,  h_min, H_MAX)
    mneg = object_mask_from_height(-s, h_min, H_MAX)
    ap = largest_external_component(mpos); an = largest_external_component(mneg)
    area_p = ap[1] if ap else 0
    area_n = an[1] if an else 0
    if area_n > area_p:
        return mneg, -1
    else:
        return mpos, +1

def measure_lwh(points3d, plane_n, plane_p0):
    s = signed_distance_map(points3d, plane_n, plane_p0)
    h_min_dyn = dynamic_h_min(s, base=H_MIN_BASE)

    # s와 -s 중 더 좋은 쪽 선택(법선 부호 안전장치)
    raw_mask, sign = pick_mask_by_sign(s, h_min_dyn)
    lg = largest_external_component(raw_mask)  # 외곽만 채워서 사용
    if lg is None or lg[1] < MIN_OBJ_PIX:
        return None, None, None, None, None, None, h_min_dyn, sign
    comp = lg[0]

    ys, xs = np.where(comp > 0)
    P_obj = points3d[ys, xs, :]
    s_obj = s[ys, xs] * sign
    H_obj = float(np.nanmax(s_obj))  # 높이[m]

    # 평면 투영 → 2D 치수
    P_proj = P_obj - np.outer(s_obj, plane_n * sign)
    u, v = plane_axes_from_normal(plane_n)
    U = P_proj.dot(u); V = P_proj.dot(v)
    UV = np.stack([U, V], axis=1).astype(np.float32)

    # convex hull로 윤곽 안정화 후 직사각 추정
    pts = UV.reshape(-1,1,2)
    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts)
        rect = cv2.minAreaRect(hull)
    else:
        rect = cv2.minAreaRect(pts)

    (cx, cy), (w, h), theta = rect
    L = float(max(w, h)); W_ = float(min(w, h))
    box = cv2.boxPoints(rect).astype(np.float32)
    return L, W_, H_obj, UV, box, comp, h_min_dyn, sign

def recommend_pack(L_mm, W_mm, H_mm, roll_list_mm, edge_margin=20, overlap=30, pad=5):
    need_w_A = L_mm + 2*H_mm + 2*edge_margin + pad
    cut_A    = 2*(W_mm + H_mm) + overlap + pad
    need_w_B = W_mm + 2*H_mm + 2*edge_margin + pad
    cut_B    = 2*(L_mm + H_mm) + overlap + pad
    def pick_roll(need):
        cands = [rw for rw in roll_list_mm if rw >= need]
        return (min(cands) if cands else None)
    roll_A = pick_roll(need_w_A)
    roll_B = pick_roll(need_w_B)
    options = []
    if roll_A is not None:
        waste_A = roll_A - need_w_A
        score_A = cut_A + waste_A*0.2
        options.append(("A", roll_A, cut_A, need_w_A, score_A))
    if roll_B is not None:
        waste_B = roll_B - need_w_B
        score_B = cut_B + waste_B*0.2
        options.append(("B", roll_B, cut_B, need_w_B, score_B))
    if not options:
        return None
    orient, roll, cutlen, needw, _ = min(options, key=lambda x: x[-1])
    return dict(orientation=orient, roll_width_mm=int(round(roll)),
                required_width_mm=float(needw),
                cut_length_mm=float(cutlen))

def main():
    global DISPLAY_SCALE, FULLSCREEN, ROI_RATIO, ROI_W_PX, ROI_H_PX, DX_PX, DY_PX

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    profile = pipe.start(cfg)

    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    try:
        depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_density)
    except Exception:
        pass
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
    depth_scale = depth_sensor.get_depth_scale()

    # 필터 체인
    dec  = rs.decimation_filter(DECIM)
    to_d = rs.disparity_transform(True)
    spat = rs.spatial_filter()
    temp = rs.temporal_filter()
    to_z = rs.disparity_transform(False)
    try:
        hole = rs.hole_filling_filter(0)
        spat.set_option(rs.option.filter_magnitude, 2)
        spat.set_option(rs.option.filter_smooth_alpha, 0.45)
        spat.set_option(rs.option.filter_smooth_delta, 18)
        temp.set_option(rs.option.filter_smooth_alpha, 0.35)
        temp.set_option(rs.option.filter_smooth_delta, 18)
    except Exception:
        hole = rs.hole_filling_filter(0)
        pass
    pc   = rs.pointcloud()

    have_plane = False
    plane_n = None; plane_p0 = None

    print("c: 캘리브레이션(벨트만 보이게), r: 재캘리브레이션, f: 전체화면, +/-: 배율, [ ]: ROI 크기, q: 종료")

    try:
        while True:
            t0 = time.time()
            frames = pipe.wait_for_frames()
            depth  = frames.get_depth_frame()
            if not depth: continue

            # 필터
            depth = dec.process(depth)
            depth = to_d.process(depth); depth = spat.process(depth); depth = temp.process(depth); depth = to_z.process(depth)
            depth = hole.process(depth)

            # 포인트클라우드
            points = pc.calculate(depth)
            verts = np.asanyarray(points.get_vertices()).view(np.float32)
            vprof = depth.get_profile().as_video_stream_profile()
            intr  = vprof.get_intrinsics()
            w, h  = intr.width, intr.height
            P3_full = verts.reshape(h, w, 3)

            # 시각화
            depth_np_full = np.asanyarray(depth.as_frame().get_data())
            depth_vis_full = (np.clip(depth_np_full, 0, 4000)/4000*255).astype(np.uint8)
            depth_vis_full = apply_colormap_u8(depth_vis_full)

            # 중앙 ROI(절대/비율)
            x0, x1, y0, y1, S = compute_roi_indices(w, h)
            P3 = P3_full[y0:y1, x0:x1, :]
            ch, cw = P3.shape[:2]

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                FULLSCREEN = not FULLSCREEN
                cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if FULLSCREEN else cv2.WINDOW_NORMAL)
            elif key in (ord('+'), ord('=')):
                DISPLAY_SCALE = min(3.0, DISPLAY_SCALE + 0.1)
            elif key in (ord('-'), ord('_')):
                DISPLAY_SCALE = max(1.0, DISPLAY_SCALE - 0.1)
            elif key == ord('['):
                if ROI_W_PX > 0 and ROI_H_PX > 0:
                    Snew = max(8, min(min(w,h), S - 10))
                    ROI_W_PX = ROI_H_PX = Snew
                else:
                    ROI_RATIO = max(ROI_MIN, ROI_RATIO - ROI_STEP)
            elif key == ord(']'):
                if ROI_W_PX > 0 and ROI_H_PX > 0:
                    Snew = max(8, min(min(w,h), S + 10))
                    ROI_W_PX = ROI_H_PX = Snew
                else:
                    ROI_RATIO = min(ROI_MAX, ROI_RATIO + ROI_STEP)
            elif key == ord('c') or key == ord('r'):
                # ROI 하단에서 평면 추정
                roi_h0 = int(ch*(1.0 - BOTTOM_ROI_RATIO))
                roi = P3[roi_h0:ch, :, :].reshape(-1,3)
                valid = np.isfinite(roi).all(axis=1)
                roi = roi[valid]
                if roi.shape[0] > 8000:
                    roi = roi[rng.choice(roi.shape[0], 8000, replace=False)]
                res = fit_plane_ransac(roi, iters=300, tau=PLANE_TAU, min_inliers=1200)
                if res is not None:
                    plane_n, plane_p0 = res
                    have_plane = True
                    print(f"평면 캘리브레이션 완료. n={plane_n}, p0={plane_p0}")
                else:
                    print("평면 추정 실패. 바닥만 보이게 하고 다시 c를 눌러주세요.")

            txt_top = "c: calibrate | r: recal | f: fullscreen | +/-: scale | [ ]: ROI size | q: quit"
            meas_txt = ""; pack_txt = ""; thr_txt  = ""; sign_txt = ""
            roi_desc = (f"ROI={S}x{S}px, dx={DX_PX}, dy={DY_PX}"
                        if (ROI_W_PX>0 and ROI_H_PX>0)
                        else f"ROI={int(ROI_RATIO*100)}%")
            if have_plane:
                # 프레임별 바닥 오프셋 미세 보정(ROI 하단 밴드 중앙값을 0으로)
                band_h0 = int(P3.shape[0]*0.85)
                Sband = signed_distance_map(P3[band_h0:], plane_n, plane_p0)
                med = np.nanmedian(Sband[np.isfinite(Sband)])
                if np.isfinite(med): plane_p0 = plane_p0 + plane_n * float(med)

                L, W_, H_obj, UV, boxUV, objmask, h_min_dyn, sign = measure_lwh(P3, plane_n, plane_p0)
                thr_txt  = f"h_min≈{h_min_dyn*1000:.0f} mm | {roi_desc}"
                sign_txt = "sign=+s" if sign>0 else "sign=-s"
                if L is not None:
                    Lmm = (L*1000.0) + SAFETY_PAD_MM
                    Wmm = (W_*1000.0) + SAFETY_PAD_MM
                    Hmm = (H_obj*1000.0) + SAFETY_PAD_MM
                    Lmm, Wmm = (max(Lmm, Wmm), min(Lmm, Wmm))
                    meas_txt = f"L={Lmm:.1f} mm, W={Wmm:.1f} mm, H={Hmm:.1f} mm"
                    rec = recommend_pack(Lmm, Wmm, Hmm, ROLL_WIDTHS_MM,
                                         edge_margin=EDGE_MARGIN_MM,
                                         overlap=OVERLAP_MM,
                                         pad=SAFETY_PAD_MM)
                    if rec:
                        orient = "L-폭기준" if rec["orientation"]=="A" else "W-폭기준"
                        pack_txt = f"롤폭 {rec['roll_width_mm']} mm ({orient}) / 절단 {rec['cut_length_mm']:.0f} mm"
                    else:
                        pack_txt = "사용 가능한 롤 폭 없음(폭 확대 필요)"
                    if objmask is not None:
                        edge = cv2.Canny(objmask, 40, 120)
                        roi_vis = depth_vis_full[y0:y1, x0:x1]
                        roi_vis[edge>0] = (0,255,255)
                else:
                    meas_txt = "물체 미검출 또는 너무 작음"

            # ROI 박스
            cv2.rectangle(depth_vis_full, (x0, y0), (x1, y1), (0, 200, 255), 2)

            # 하단 정보바
            info = np.zeros((INFO_BAR_H, depth_vis_full.shape[1], 3), dtype=np.uint8)
            def put_line(img, text, x, y, color=(255,255,255)):
                cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            put_line(info, txt_top, 10, 20)
            if meas_txt: put_line(info, meas_txt, 10, 40)
            if pack_txt: put_line(info, pack_txt, 10 + depth_vis_full.shape[1]//2, 40, (0,255,0))
            if thr_txt:  put_line(info, thr_txt, depth_vis_full.shape[1]-360, 20, (200,200,255))
            if sign_txt: put_line(info, sign_txt, depth_vis_full.shape[1]-120, 40, (200,255,200))

            vis = np.vstack([depth_vis_full, info])
            if DISPLAY_SCALE != 1.0:
                vis = cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST)

            fps = 1.0/(time.time()-t0+1e-6)
            cv2.putText(vis, f"{fps:.1f} FPS", (10, vis.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow(WIN_NAME, vis)

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
