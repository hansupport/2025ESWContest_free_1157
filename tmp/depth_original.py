# depth.py
# D435f 기반 컨베이어 물체 치수(가로 L, 세로 W, 높이 H) 측정 모듈
# - 시각화/키조작 없음
# - 외부 호출용 API:
#     DepthEstimator.start()
#     DepthEstimator.warmup(seconds=1.0)
#     DepthEstimator.calibrate(max_seconds=2.0)
#     DepthEstimator.measure_dimensions(duration_s=0.7) -> (L_mm|None, W_mm|None, H_mm|None)
#     DepthEstimator.measure_depth(duration_s=0.7) -> float|None   # H만 반환(호환용)
#     DepthEstimator.stop()

import time
import numpy as np
import pyrealsense2 as rs
import cv2

rng = np.random.default_rng(42)

# 기본 설정
W, H, FPS = 848, 480, 30
ROI_RATIO_DEFAULT = 0.60
BOTTOM_ROI_RATIO_DEFAULT = 0.25
PLANE_TAU = 0.008
DECIM = 1
H_MIN_BASE = 0.003     # 3 mm
H_MAX = 1.000
MIN_OBJ_PIX = 30

def center_square_indices(w, h, ratio):
    S = int(min(w, h) * ratio)
    x0 = w//2 - S//2; y0 = h//2 - S//2
    return x0, x0+S, y0, y0+S

def orient_normal_to_camera(n):
    cam_dir = np.array([0., 0., -1.], dtype=np.float32)
    return n if np.dot(n, cam_dir) > 0 else -n

def fit_plane_ransac(P, iters=300, tau=PLANE_TAU, min_inliers=1200):
    N = P.shape[0]
    if N < 3:
        return None
    best_mask = None; best_n = None; best_p0 = None
    for _ in range(iters):
        ids = rng.choice(N, size=3, replace=False)
        A, B, C = P[ids]
        n = np.cross(B-A, C-A)
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            continue
        n = n / nn
        d = -np.dot(n, A)
        dist = np.abs(P.dot(n) + d)
        mask = dist < tau
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask = mask; best_n = n; best_p0 = A
    if best_mask is None or best_mask.sum() < min_inliers:
        return None
    Pin = P[best_mask]
    c = Pin.mean(axis=0)
    U, S, Vt = np.linalg.svd(Pin - c, full_matrices=False)
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

def dynamic_h_min(s_map, base=H_MIN_BASE):
    s_valid = s_map[np.isfinite(s_map)]
    if s_valid.size < 500:
        return base
    near = s_valid[np.abs(s_valid) < 0.05]  # ±5 cm
    if near.size < 500:
        near = s_valid
    med = np.median(near)
    mad = np.median(np.abs(near - med)) + 1e-9
    sigma = 1.4826 * mad
    thr = max(base, med + max(0.004, 1.5*sigma))  # 최소 4 mm
    return float(thr)

def object_mask_from_height(s_map, h_min, h_max):
    mask = (s_map > h_min) & (s_map < h_max) & np.isfinite(s_map)
    mask = mask.astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def largest_component(mask):
    num, lab = cv2.connectedComponents(mask)
    if num <= 1:
        return None
    areas = [(lab==i).sum() for i in range(1, num)]
    idx = int(np.argmax(areas)) + 1
    comp = (lab==idx).astype(np.uint8)*255
    return comp, areas[idx-1]

def pick_mask_by_sign(s, h_min):
    mpos = object_mask_from_height(s,  h_min, H_MAX)
    mneg = object_mask_from_height(-s, h_min, H_MAX)
    ap = largest_component(mpos); an = largest_component(mneg)
    area_p = ap[1] if ap else 0
    area_n = an[1] if an else 0
    if area_n > area_p:
        return mneg, -1
    else:
        return mpos, +1

def measure_lwh(points3d, plane_n, plane_p0):
    s = signed_distance_map(points3d, plane_n, plane_p0)
    h_min_dyn = dynamic_h_min(s, base=H_MIN_BASE)

    mask, sign = pick_mask_by_sign(s, h_min_dyn)
    lg = largest_component(mask)
    if lg is None or lg[1] < MIN_OBJ_PIX:
        return None, None, None, None, None, None, h_min_dyn, sign
    comp = lg[0]

    ys, xs = np.where(comp > 0)
    P_obj = points3d[ys, xs, :]
    s_obj = s[ys, xs] * sign
    H_obj = float(np.nanmax(s_obj))  # m

    P_proj = P_obj - np.outer(s_obj, plane_n * sign)
    u, v = plane_axes_from_normal(plane_n)
    U = P_proj.dot(u); V = P_proj.dot(v)
    UV = np.stack([U, V], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(UV.reshape(-1,1,2))
    (cx, cy), (w, h), theta = rect
    L = float(max(w, h)); W_ = float(min(w, h))
    box = cv2.boxPoints(rect).astype(np.float32)
    return L, W_, H_obj, UV, box, comp, h_min_dyn, sign

class DepthEstimator:
    def __init__(self,
                 w=W, h=H, fps=FPS,
                 roi_ratio=ROI_RATIO_DEFAULT,
                 bottom_roi_ratio=BOTTOM_ROI_RATIO_DEFAULT):
        self.w = w; self.h = h; self.fps = fps
        self.roi_ratio = roi_ratio
        self.bottom_roi_ratio = bottom_roi_ratio

        self.pipe = None
        self.profile = None
        self.depth_sensor = None
        self.depth_scale = None

        self.dec = None; self.to_d = None; self.spat = None
        self.temp = None; self.to_z = None; self.hole = None
        self.pc = None

        self.have_plane = False
        self.plane_n = None
        self.plane_p0 = None

        self._intr = None

    def start(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, self.fps)
        self.profile = self.pipe.start(cfg)

        dev = self.profile.get_device()
        self.depth_sensor = dev.first_depth_sensor()
        try:
            self.depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_density)
        except Exception:
            pass
        if self.depth_sensor.supports(rs.option.emitter_enabled):
            self.depth_sensor.set_option(rs.option.emitter_enabled, 1)
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # 필터 체인
        self.dec  = rs.decimation_filter(DECIM)
        self.to_d = rs.disparity_transform(True)
        self.spat = rs.spatial_filter()
        self.temp = rs.temporal_filter()
        self.to_z = rs.disparity_transform(False)
        try:
            self.hole = rs.hole_filling_filter(0)
            self.spat.set_option(rs.option.filter_magnitude, 2)
            self.spat.set_option(rs.option.filter_smooth_alpha, 0.45)
            self.spat.set_option(rs.option.filter_smooth_delta, 18)
            self.temp.set_option(rs.option.filter_smooth_alpha, 0.35)
            self.temp.set_option(rs.option.filter_smooth_delta, 18)
        except Exception:
            self.hole = rs.hole_filling_filter(0)
            pass
        self.pc = rs.pointcloud()

        # intrinsics 캐시
        frames = self.pipe.wait_for_frames()
        depth  = frames.get_depth_frame()
        vprof = depth.get_profile().as_video_stream_profile()
        intr  = vprof.get_intrinsics()
        self._intr = intr

    def stop(self):
        if self.pipe is not None:
            try:
                self.pipe.stop()
            except Exception:
                pass
        self.pipe = None

    def _get_P3_roi(self):
        frames = self.pipe.wait_for_frames()
        depth  = frames.get_depth_frame()
        if not depth:
            return None, None, None, None

        # 필터
        depth = self.dec.process(depth)
        depth = self.to_d.process(depth)
        depth = self.spat.process(depth)
        depth = self.temp.process(depth)
        depth = self.to_z.process(depth)
        depth = self.hole.process(depth)

        points = self.pc.calculate(depth)
        verts = np.asanyarray(points.get_vertices()).view(np.float32)
        w, h = self._intr.width, self._intr.height
        P3_full = verts.reshape(h, w, 3)

        x0, x1, y0, y1 = center_square_indices(w, h, self.roi_ratio)
        P3 = P3_full[y0:y1, x0:x1, :]
        return P3, (x0, x1, y0, y1), depth, P3_full

    def warmup(self, seconds=1.0):
        t_end = time.time() + float(seconds)
        cnt = 0
        while time.time() < t_end:
            _ = self._get_P3_roi()
            cnt += 1
        return cnt

    def calibrate(self, max_seconds=2.0, sample_cap=8000):
        """
        ROI 하단 밴드에서 바닥 평면 추정.
        성공 시 True, 실패 시 False 반환.
        """
        t_end = time.time() + float(max_seconds)
        ok = False
        while time.time() < t_end:
            P3, _, _, _ = self._get_P3_roi()
            if P3 is None:
                continue
            ch, cw = P3.shape[:2]
            roi_h0 = int(ch * (1.0 - self.bottom_roi_ratio))
            roi = P3[roi_h0:ch, :, :].reshape(-1, 3)
            valid = np.isfinite(roi).all(axis=1)
            roi = roi[valid]
            if roi.shape[0] > sample_cap:
                roi = roi[rng.choice(roi.shape[0], sample_cap, replace=False)]
            res = fit_plane_ransac(roi, iters=300, tau=PLANE_TAU, min_inliers=1200)
            if res is not None:
                self.plane_n, self.plane_p0 = res
                self.have_plane = True
                ok = True
                break
        return ok

    def _measure_dims_once_mm(self):
        """
        한 프레임에서 L, W, H(mm)를 추정. 실패 시 None.
        L은 바닥 평면상 긴 변, W는 짧은 변.
        """
        if not self.have_plane:
            return None

        P3, _, _, _ = self._get_P3_roi()
        if P3 is None:
            return None

        # 프레임별 바닥 오프셋 미세 보정(ROI 하단 밴드 중앙값을 0으로)
        band_h0 = int(P3.shape[0] * 0.85)
        Sband = signed_distance_map(P3[band_h0:], self.plane_n, self.plane_p0)
        med = np.nanmedian(Sband[np.isfinite(Sband)])
        if np.isfinite(med):
            self.plane_p0 = self.plane_p0 + self.plane_n * float(med)

        L, W_, H_obj, UV, boxUV, objmask, h_min_dyn, sign = measure_lwh(P3, self.plane_n, self.plane_p0)
        if H_obj is None or L is None or W_ is None:
            return None
        return float(L*1000.0), float(W_*1000.0), float(H_obj*1000.0)  # mm

    @staticmethod
    def _trimmed_mean(arr):
        if len(arr) == 0:
            return None
        if len(arr) >= 3:
            a = np.sort(np.asarray(arr, dtype=np.float64))
            inner = a[1:-1]
            return float(inner.mean()) if inner.size > 0 else float(a.mean())
        else:
            return float(np.mean(arr))

    def measure_dimensions(self, duration_s=0.7):
        """
        duration_s 동안 여러 프레임에서 L/W/H(mm)를 수집.
        각 차원별로 최소/최대 한 개씩 제거한 뒤 평균을 반환.
        유효 샘플이 없으면 (None, None, None).
        """
        t_end = time.time() + float(duration_s)
        Ls, Ws, Hs = [], [], []
        while time.time() < t_end:
            res = self._measure_dims_once_mm()
            if res is None:
                continue
            Lmm, Wmm, Hmm = res
            if np.isfinite(Lmm) and np.isfinite(Wmm) and np.isfinite(Hmm):
                Ls.append(Lmm); Ws.append(Wmm); Hs.append(Hmm)

        if len(Ls) == 0:
            return None, None, None

        Lm = self._trimmed_mean(Ls)
        Wm = self._trimmed_mean(Ws)
        Hm = self._trimmed_mean(Hs)
        return Lm, Wm, Hm

    # 호환용: 높이만 반환
    def measure_depth(self, duration_s=0.7):
        """
        duration_s 동안 높이(mm)만 평균(트림드) 반환. 유효 샘플 없으면 None.
        """
        _, _, Hm = self.measure_dimensions(duration_s=duration_s)
        return Hm


if __name__ == "__main__":
    # 간단 자가 테스트: 시작 → 웜업 → 캘리브 → 0.7초 측정 → 출력
    d = DepthEstimator()
    try:
        d.start()
        d.warmup(seconds=1.0)
        ok = d.calibrate(max_seconds=2.0)
        if not ok:
            print("평면 캘리브 실패")
        else:
            Lm, Wm, Hm = d.measure_dimensions(duration_s=0.7)
            print(f"측정 평균(극단값 2개 제외) L={Lm} mm, W={Wm} mm, H={Hm} mm")
    finally:
        d.stop()
