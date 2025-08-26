# depth.py
# D435f 깊이 기반 컨베이어 물체 치수 측정 + 통계 피처 산출(10프레임)
# - Height: 바닥 평면 기준(+s), 물체 마스크 전체에서 s의 최대값
# - L/W: 평면 투영 후 PCA 축에서 퍼센타일 폭(Q99-Q01)
# - 여러 프레임 요약: d1,d2,d3(중앙값), MAD1..3, r̄1..3, sr1..3, logV̄, logsV, q
# - 외부에서 DepthEstimator 클래스를 통해 이용 (ROI는 roi_px + roi_offset 사용)

import numpy as np
import cv2, time
import pyrealsense2 as rs

rng = np.random.default_rng(42)

# 기본 파라미터
W, H, FPS = 1280, 720, 6
DECIM = 1
PLANE_TAU = 0.004       # ~4mm
H_MIN_BASE = 0.003      # ~3mm
H_MAX = 0.40
MIN_OBJ_PIX = 40
BOTTOM_ROI_RATIO = 0.20
HOLE_FILL = False
CORE_MARGIN_PX = 1
P_LO, P_HI = 1.0, 99.0  # 퍼센타일 폭

def clamp(v, lo, hi): return max(lo, min(hi, v))

def orient_normal_to_camera(n):
    cam_dir = np.array([0., 0., -1.], dtype=np.float32)
    return n if float(np.dot(n, cam_dir)) > 0 else -n

def plane_axes_from_normal(n):
    t = np.array([1.,0.,0.]) if abs(n[0]) < 0.9 else np.array([0.,0.,1.])
    u = np.cross(n, t); u /= (np.linalg.norm(u)+1e-12)
    v = np.cross(n, u); v /= (np.linalg.norm(v)+1e-12)
    return u, v

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
    return n_ref.astype(np.float32), c.astype(np.float32)

def signed_distance_map(P3, plane_n, plane_p0):
    S = np.einsum('ijk,k->ij', P3 - plane_p0, plane_n).astype(np.float32)
    invalid = ~np.isfinite(P3).all(axis=2)
    S[invalid] = np.nan
    return S

def dynamic_h_min(s_map, base=H_MIN_BASE):
    s_valid = s_map[np.isfinite(s_map)]
    if s_valid.size < 500:
        return base
    near = s_valid[np.abs(s_valid) < 0.05]
    if near.size < 500: near = s_valid
    med = np.median(near)
    mad = np.median(np.abs(near - med)) + 1e-9
    sigma = 1.4826 * mad
    thr = max(base, float(med + max(0.002, 1.3*sigma)))
    return thr

def object_mask_from_height(s_map, h_min, h_max):
    mask = (s_map > h_min) & (s_map < h_max) & np.isfinite(s_map)
    mask = mask.astype(np.uint8)*255
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)  # ← 3.6 호환(월러스 제거)
    return mask

def largest_external_component(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(mask)
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
    mu = UV.mean(axis=0, keepdims=True)
    X = UV - mu
    if X.shape[0] < 3:
        p1 = X[:,0]; p2 = X[:,1]
        L = float(np.nanpercentile(p1, P_HI) - np.nanpercentile(p1, P_LO))
        W = float(np.nanpercentile(p2, P_HI) - np.nanpercentile(p2, P_LO))
        evecs = np.eye(2, dtype=np.float32)
        if W > L:
            L, W = W, L
            evecs = evecs[:, ::-1]
        return L, W, mu[0], evecs
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    evecs = Vt.T.astype(np.float32)
    P = X.dot(evecs)
    p1, p2 = P[:,0], P[:,1]
    L = float(np.nanpercentile(p1, P_HI) - np.nanpercentile(p1, P_LO))
    W = float(np.nanpercentile(p2, P_HI) - np.nanpercentile(p2, P_LO))
    if W > L:
        L, W = W, L
        evecs = evecs[:, ::-1]
    return L, W, mu[0], evecs

class DepthEstimator:
    def __init__(self,
                 width=W, height=H, fps=FPS,
                 roi_ratio=0.60, roi_px=(230,230), roi_offset=(20,-210)):
        self.width = width; self.height = height; self.fps = fps
        self.roi_ratio = roi_ratio
        self.roi_w_px, self.roi_h_px = roi_px
        self.dx_px, self.dy_px = roi_offset
        self.pipe = None
        self.profile = None
        self.pc = None
        self.have_plane = False
        self.plane_n = None
        self.plane_p0 = None

        # filters
        self.dec  = rs.decimation_filter(DECIM)
        self.to_d = rs.disparity_transform(True)
        self.spat = rs.spatial_filter()
        self.temp = rs.temporal_filter()
        self.to_z = rs.disparity_transform(False)
        self.hole = rs.hole_filling_filter(1) if HOLE_FILL else None
        try:
            self.spat.set_option(rs.option.filter_magnitude, 1)
            self.spat.set_option(rs.option.filter_smooth_alpha, 0.2)
            self.spat.set_option(rs.option.filter_smooth_delta, 10)
            self.temp.set_option(rs.option.filter_smooth_alpha, 0.25)
            self.temp.set_option(rs.option.filter_smooth_delta, 10)
        except Exception:
            pass

    # ---------- pipeline ----------
    def start(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.profile = self.pipe.start(cfg)
        dev = self.profile.get_device()
        depth_sensor = dev.first_depth_sensor()
        try:
            depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
        except Exception:
            pass
        try:
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1)
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if depth_sensor.supports(rs.option.exposure):
                depth_sensor.set_option(rs.option.exposure, 2000.0)
            if depth_sensor.supports(rs.option.gain):
                depth_sensor.set_option(rs.option.gain, 18.0)
        except Exception:
            pass
        self.pc = rs.pointcloud()

    def stop(self):
        try:
            if self.pipe: self.pipe.stop()
        except Exception:
            pass
        self.pipe = None
        self.profile = None
        self.pc = None
        self.have_plane = False
        self.plane_n = None; self.plane_p0 = None

    def warmup(self, seconds=1.0):
        t0 = time.time(); cnt = 0
        while time.time() - t0 < seconds:
            frames = self.pipe.wait_for_frames()
            depth = frames.get_depth_frame()
            if not depth: continue
            cnt += 1
        return cnt

    # ---------- utils ----------
    def _grab_depth_points(self):
        frames = self.pipe.wait_for_frames()
        depth  = frames.get_depth_frame()
        if not depth: return None, None, None
        d = self.dec.process(depth)
        d = self.to_d.process(d); d = self.spat.process(d); d = self.temp.process(d); d = self.to_z.process(d)
        if self.hole is not None: d = self.hole.process(d)
        points = self.pc.calculate(d)
        verts = np.asanyarray(points.get_vertices()).view(np.float32)
        vprof = d.get_profile().as_video_stream_profile()
        intr  = vprof.get_intrinsics()
        w, h  = intr.width, intr.height
        P3_full = verts.reshape(h, w, 3)
        z = P3_full[:,:,2]
        P3_full[z == 0] = np.nan
        return d, P3_full, (w,h)

    def _compute_roi(self, w, h):
        if self.roi_w_px > 0 and self.roi_h_px > 0:
            S = int(min(clamp(self.roi_w_px,8,min(w,h)),
                        clamp(self.roi_h_px,8,min(w,h))))
        else:
            S = int(min(w, h) * clamp(self.roi_ratio, 0.30, 0.95))
        cx = w//2 + self.dx_px
        cy = h//2 + self.dy_px
        x0 = clamp(cx - S//2, 0, w - S)
        y0 = clamp(cy - S//2, 0, h - S)
        return x0, x0+S, y0, y0+S, S

    # ---------- calibration ----------
    def calibrate(self, max_seconds=3.0):
        t0 = time.time()
        ok = False
        while time.time() - t0 < max_seconds:
            d, P3_full, wh = self._grab_depth_points()
            if P3_full is None: continue
            w,h = wh
            x0,x1,y0,y1,S = self._compute_roi(w,h)
            P3 = P3_full[y0:y1, x0:x1, :]
            ch, cw = P3.shape[:2]
            roi_h0 = int(ch*(1.0 - BOTTOM_ROI_RATIO))
            roi = P3[roi_h0:ch, :, :].reshape(-1,3)
            valid = np.isfinite(roi).all(axis=1)
            roi = roi[valid]
            if roi.shape[0] > 8000:
                roi = roi[rng.choice(roi.shape[0], 8000, replace=False)]
            res = fit_plane_ransac(roi, iters=300, tau=PLANE_TAU, min_inliers=1200)
            if res is not None:
                self.plane_n, self.plane_p0 = res
                ok = True
                break
        self.have_plane = ok
        return ok

    # ---------- single measurement ----------
    def measure_once(self):
        if not self.have_plane:
            return None, None, None
        d, P3_full, wh = self._grab_depth_points()
        if P3_full is None: return None, None, None
        w,h = wh
        x0,x1,y0,y1,S = self._compute_roi(w,h)
        P3 = P3_full[y0:y1, x0:x1, :]
        ch, cw = P3.shape[:2]

        # 프레임별 바닥 오프셋 보정(ROI 하단 중앙값=0)
        band_h0 = int(P3.shape[0]*(1.0 - BOTTOM_ROI_RATIO))
        Sband = signed_distance_map(P3[band_h0:], self.plane_n, self.plane_p0)
        med = np.nanmedian(Sband[np.isfinite(Sband)])
        if np.isfinite(med):
            self.plane_p0 = self.plane_p0 + self.plane_n * float(med)

        s = signed_distance_map(P3, self.plane_n, self.plane_p0)
        h_min_dyn = dynamic_h_min(s, base=H_MIN_BASE)

        raw_mask = object_mask_from_height(s, h_min_dyn, H_MAX)
        if raw_mask is None:
            return None, None, None
        lg = largest_external_component(raw_mask)
        if lg is None or lg[1] < MIN_OBJ_PIX:
            return None, None, None
        comp = lg[0]

        ysH, xsH = np.where(comp > 0)
        s_obj_full = s[ysH, xsH]
        if not np.any(np.isfinite(s_obj_full)):
            return None, None, None
        H_obj = float(np.nanmax(s_obj_full))

        comp_core = erode_core(comp, rpx=CORE_MARGIN_PX)
        if comp_core is None or comp_core.sum() < MIN_OBJ_PIX:
            comp_core = comp
        ys, xs = np.where(comp_core > 0)
        s_obj = s[ys, xs]
        P_obj = P3[ys, xs, :]

        valid_mask = ~np.isnan(P_obj).any(axis=1)
        P_obj = P_obj[valid_mask]
        s_obj = s_obj[valid_mask]
        if P_obj.shape[0] < 3:
            return None, None, H_obj

        P_proj = P_obj - np.outer(s_obj, self.plane_n)
        u, v = plane_axes_from_normal(self.plane_n)
        U = P_proj.dot(u); V = P_proj.dot(v)
        UV = np.stack([U, V], axis=1).astype(np.float32)

        L, W_, mu, E = robust_pca_lengths(UV)
        Lmm = L*1000.0
        Wmm = W_*1000.0
        Hmm = H_obj*1000.0

        d_sorted = np.sort([Lmm, Wmm, Hmm]).astype(np.float32)
        return float(d_sorted[0]), float(d_sorted[1]), float(d_sorted[2])

    # ---------- multi-frame summary ----------
    def measure_dimensions(self, duration_s=0.7, n_frames=10):
        t0 = time.time()
        vals = []
        while len(vals) < n_frames and (time.time()-t0) < duration_s:
            d1,d2,d3 = self.measure_once()
            if d1 is None:
                continue
            vals.append([d1,d2,d3])
        if len(vals) == 0:
            return None

        vals = np.asarray(vals, dtype=np.float32)  # (n,3)
        d_med = np.median(vals, axis=0)
        mad = np.median(np.abs(vals - d_med), axis=0)

        r1 = vals[:,1]/(vals[:,0]+1e-6)
        r2 = vals[:,2]/(vals[:,1]+1e-6)
        r3 = vals[:,2]/(vals[:,0]+1e-6)
        r_mean = np.array([r1.mean(), r2.mean(), r3.mean()], dtype=np.float32)
        r_std  = np.array([r1.std(),  r2.std(),  r3.std()],  dtype=np.float32)

        vols = (vals[:,0]*vals[:,1]*vals[:,2])
        logVmean = float(np.log(np.mean(vols)+1e-6))
        logsV    = float(np.log(np.std(vols)+1e-6))

        CVs = np.std(vals,axis=0)/(np.mean(vals,axis=0)+1e-6)
        q = float(np.exp(-0.5*float(np.sum(CVs**2))))

        return dict(
            d1=float(d_med[0]), d2=float(d_med[1]), d3=float(d_med[2]),
            mad1=float(mad[0]), mad2=float(mad[1]), mad3=float(mad[2]),
            r1=float(r_mean[0]), r2=float(r_mean[1]), r3=float(r_mean[2]),
            sr1=float(r_std[0]), sr2=float(r_std[1]), sr3=float(r_std[2]),
            logV=float(logVmean), logsV=float(logsV), q=q
        )
