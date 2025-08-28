# datamatrix.py
# 다중 ROI + 좌우반전 + 전처리·멀티트라이 강화(시간예산) + 간단 디워프 + 1초 하트비트 고정
# ROI/해상도 불변

import time
import sys
import threading
import queue
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
import cv2 as cv

TICK = 0.20
LOG_EVERY_DECODE = 1
MAX_BACKOFF = 1.0
# AUTO 미사용. 기본 카메라를 2번으로 고정.
DEFAULT_CAMERA: Union[int,str] = 2

# ===== RGB 전용 기본 ROI (요청값 반영) =====
# - 메인: 530x530, dx=+100, dy=-180
# - 좌 미러: 400x550, dx=-470, dy=-200, 좌우반전
# - 우 미러: 390x550, dx=+680, dy=-190, 좌우반전
DEFAULT_ROIS: List[Dict[str, Any]] = [
    {"name": "RGB_MAIN",   "size": [530, 530], "offset": [ +90, -130], "hflip": False},
    {"name": "RGB_L_MIRR", "size": [390, 500], "offset": [ -450, -140], "hflip": True },
    {"name": "RGB_R_MIRR", "size": [390, 490], "offset": [ +660, -140], "hflip": True },
]

def log(msg: str):
    ts = time.strftime("%H:%M:%S", time.localtime())
    sys.stderr.write(f"[{ts}] {msg}\n")
    sys.stderr.flush()

try:
    cv.setUseOptimized(True); cv.setNumThreads(1)
    log("OpenCV optimized=True, threads=1")
except Exception:
    pass

def load_dmtx_decode():
    t0 = time.perf_counter()
    try:
        from pylibdmtx.pylibdmtx import decode
        log(f"pylibdmtx import(ok) in {(time.perf_counter()-t0)*1000:.2f} ms")
        return decode
    except Exception:
        pass
    try:
        t1 = time.perf_counter()
        import pylibdmtx.pylibdmtx as dmtx
        log(f"pylibdmtx fallback import(ok) in {(time.perf_counter()-t1)*1000:.2f} ms")
        return dmtx.decode
    except Exception as e:
        log(f"pylibdmtx import FAIL in {(time.perf_counter()-t0)*1000:.2f} ms")
        raise ImportError(
            "pylibdmtx decode 로드 실패: " + str(e) +
            "\n(참고: sudo apt install libdmtx0a libdmtx-dev 후 "
            "pip install --no-binary :all: pylibdmtx==0.1.10)"
        )

_dm_decode = load_dmtx_decode()

def open_camera(camera=DEFAULT_CAMERA, prefer_res=(1920,1080), prefer_fps=6):
    t_all0 = time.perf_counter()

    # AUTO 경로를 사용하지 않으므로 바로 V4L2로 진입
    cam_id = 0 if camera in (None, "auto") else camera
    cap = cv.VideoCapture(cam_id, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  int(prefer_res[0]))
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(prefer_res[1]))
    cap.set(cv.CAP_PROP_FPS,          int(prefer_fps))
    # 내부 버퍼를 얕게 유지(지연 최소화)
    try:
        cap.set(cv.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass
    ok, _ = cap.read()
    if ok:
        log(f"Webcam open {prefer_res[0]}x{prefer_res[1]}@{prefer_fps} in {(time.perf_counter()-t_all0)*1000:.2f} ms")
        return ("webcam", cap)

    # 혹시 실패 시 마지막 폴백으로 RealSense 시도(명시적으로 쓰진 않지만 살려둠)
    try:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        tried = []
        for w, h, fps in [(int(prefer_res[0]), int(prefer_res[1]), int(prefer_fps)), (1920,1080,6), (1280,720,6)]:
            t0 = time.perf_counter()
            try:
                config.disable_all_streams()
                config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
                profile = pipeline.start(config)
                dev = profile.get_device()
                color_sensor = dev.first_color_sensor()
                try: color_sensor.set_option(rs.option.frames_queue_size, 2)
                except Exception: pass
                log(f"RealSense start {w}x{h}@{fps} in {(time.perf_counter()-t0)*1000:.2f} ms")
                log(f"Camera total open time {(time.perf_counter()-t_all0)*1000:.2f} ms")
                return ("realsense", (pipeline, rs))
            except Exception as e:
                tried.append(f"{w}x{h}@{fps} 실패: {e}")
                try: pipeline.stop()
                except Exception: pass
                pipeline = rs.pipeline(); config = rs.config()
        log("[WARN] RealSense 설정 실패: " + " | ".join(tried))
    except Exception as e:
        log(f"[WARN] RealSense 실패: {e}")

    log("[ERR] 카메라 열기 실패"); sys.exit(1)

def read_frame_nonblocking(cam):
    if cam[0] == "realsense":
        pipeline, rs = cam[1]
        frames = pipeline.poll_for_frames()
        if not frames:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=50)
            except Exception:
                return None
        c = frames.get_color_frame()
        if not c: return None
        return np.asanyarray(c.get_data())
    else:
        cap = cam[1]
        ret, frame = cap.read()
        if not ret: return None
        return frame

def likely_has_code(gray):
    lap_var = cv.Laplacian(gray, cv.CV_64F).var()
    if lap_var < 15.0:
        return False
    edges = cv.Canny(gray, 50, 150)
    if (edges > 0).mean() < 0.01:
        return False
    return True

def crop_roi_center(bgr, roi_w, roi_h, dx, dy):
    h, w = bgr.shape[:2]
    x0 = (w - roi_w) // 2 + dx
    y0 = (h - roi_h) // 2 + dy
    x0 = max(0, min(w - roi_w, x0))
    y0 = max(0, min(h - roi_h, y0))
    return bgr[y0:y0+roi_h, x0:x0+roi_w]

# ── 보조 전처리 ──
def _clahe(gray):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def _sharpen(gray):
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
    return cv.filter2D(gray, -1, k)

def _binarize(gray):
    return cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, 21, 5)

def _rot90s(img):
    return [
        img,
        cv.rotate(img, cv.ROTATE_180),
        cv.rotate(img, cv.ROTATE_90_CLOCKWISE),
        cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE),
    ]

# ── 간단 디워프(사변형 → 정사각) ──
def _try_dewarp(gray):
    g = cv.GaussianBlur(gray, (3,3), 0)
    edges = cv.Canny(g, 60, 180)
    cnts, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    best = None; best_area = 0
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4 and cv.isContourConvex(approx):
            area = cv.contourArea(approx)
            if area > best_area:
                best = approx; best_area = area
    if best is None or best_area < 600:
        return None
    pts = best.reshape(4,2).astype(np.float32)
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1).ravel()
    rect = np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)
    w = int(max(np.linalg.norm(rect[1]-rect[0]), np.linalg.norm(rect[2]-rect[3])))
    h = int(max(np.linalg.norm(rect[3]-rect[0]), np.linalg.norm(rect[2]-rect[1])))
    side = max(64, min(1024, max(w, h)))
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype=np.float32)
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(gray, M, (side, side))
    return warped

# ── 제한시간(프레임/ROI 별) 내에서 빠른→정교 순으로 시도 ──
def decode_payloads_robust(gray, max_count=4, time_budget_ms=120):
    t_start = time.perf_counter()

    def left_ms():
        return max(0.0, time_budget_ms - (time.perf_counter()-t_start)*1000.0)

    # Fast path: 원본/180°, 짧은 timeout
    for img in _rot90s(gray)[:2]:  # [0°, 180°]
        try:
            res = _dm_decode(img, max_count=max_count, timeout=int(min(60, left_ms())))
        except TypeError:
            res = _dm_decode(img, max_count=max_count)
        if res:
            out = []
            for r in res:
                try: out.append(r.data.decode("utf-8", errors="replace").strip())
                except Exception: out.append(str(r.data))
            if out: return out
        if left_ms() <= 0: return []

    # 업샘플+샤픈 후 0/90/180/270°
    if left_ms() <= 0: return []
    up = cv.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv.INTER_CUBIC)
    sharp = _sharpen(up)
    for img in _rot90s(sharp):  # 4방향
        if left_ms() <= 0: return []
        try:
            res = _dm_decode(img, max_count=max_count, timeout=int(min(90, left_ms())))
        except TypeError:
            res = _dm_decode(img, max_count=max_count)
        if res:
            out = []
            for r in res:
                try: out.append(r.data.decode("utf-8", errors="replace").strip())
                except Exception: out.append(str(r.data))
            if out: return out

    # CLAHE / 이진화 (0/90°만)
    if left_ms() <= 0: return []
    cla = _clahe(up)
    for img in [cla, cv.rotate(cla, cv.ROTATE_90_CLOCKWISE)]:
        if left_ms() <= 0: return []
        try:
            res = _dm_decode(img, max_count=max_count, timeout=int(min(80, left_ms())))
        except TypeError:
            res = _dm_decode(img, max_count=max_count)
        if res:
            out = []
            for r in res:
                try: out.append(r.data.decode("utf-8", errors="replace").strip())
                except Exception: out.append(str(r.data))
            if out: return out

    if left_ms() <= 0: return []
    binimg = _binarize(up)
    for img in [binimg, cv.rotate(binimg, cv.ROTATE_90_CLOCKWISE)]:
        if left_ms() <= 0: return []
        try:
            res = _dm_decode(img, max_count=max_count, timeout=int(min(80, left_ms())))
        except TypeError:
            res = _dm_decode(img, max_count=max_count)
        if res:
            out = []
            for r in res:
                try: out.append(r.data.decode("utf-8", errors="replace").strip())
                except Exception: out.append(str(r.data))
            if out: return out

    # 디워프 1회 (기울어진 30~50° 대응)
    if left_ms() <= 0: return []
    warped = _try_dewarp(gray)
    if warped is not None and left_ms() > 0:
        try:
            res = _dm_decode(warped, max_count=max_count, timeout=int(min(100, left_ms())))
        except TypeError:
            res = _dm_decode(warped, max_count=max_count)
        if res:
            out = []
            for r in res:
                try: out.append(r.data.decode("utf-8", errors="replace").strip())
                except Exception: out.append(str(r.data))
            if out: return out

    return []

def decode_payloads_fast4(gray, max_count=4, time_budget_ms=80):
    """
    가장 빠른 경로만 수행:
      - 전처리/업샘플/CLAHE/이진화/디워프 전부 없음
      - 0°, 90°, 180°, 270° 네 방향만 시도
      - 남은 예산 내에서 pylibdmtx timeout 전달(가능한 버전일 때)
    """
    t0 = time.perf_counter()
    rotations = [
        gray,
        cv.rotate(gray, cv.ROTATE_90_CLOCKWISE),
        cv.rotate(gray, cv.ROTATE_180),
        cv.rotate(gray, cv.ROTATE_90_COUNTERCLOCKWISE),
    ]
    for img in rotations:
        left = time_budget_ms - (time.perf_counter() - t0) * 1000.0
        if left <= 5:
            break
        try:
            res = _dm_decode(img, max_count=max_count, timeout=int(min(80, max(5, left))))
        except TypeError:
            # 구버전 pylibdmtx는 timeout 인자를 지원 안 할 수 있음
            res = _dm_decode(img, max_count=max_count)
        if res:
            out = []
            for r in res:
                try:
                    out.append(r.data.decode("utf-8", errors="replace").strip())
                except Exception:
                    out.append(str(r.data))
            if out:
                return out
    return []

class DMatrixWatcher:
    def __init__(self,
                 camera=DEFAULT_CAMERA,
                 prefer_res=(1920,1080),
                 prefer_fps=6,
                 roi_px: Optional[Tuple[int,int]] = None,
                 roi_offset: Tuple[int,int] = (0, 0),
                 rois: Optional[List[Dict[str, Any]]] = None,
                 decode_interval: float = TICK,
                 log_every_decode: int = LOG_EVERY_DECODE,
                 max_backoff: float = MAX_BACKOFF):
        self.camera = camera
        self.prefer_res = prefer_res
        self.prefer_fps = prefer_fps
        self.rois = self._build_rois(rois, roi_px, roi_offset)
        self.decode_interval_base = float(max(0.02, decode_interval))
        self.log_every_decode = max(1, int(log_every_decode))
        self.max_backoff = float(max_backoff)

        self._cap = None
        self._run = False
        self._paused = False
        self._lock = threading.Lock()
        self._q: "queue.Queue[Tuple[float, List[str]]]" = queue.Queue()
        self._thr: Optional[threading.Thread] = None
        self._last_decode_ts = 0.0

    @staticmethod
    def _build_rois(rois, roi_px, roi_offset):
        if rois and isinstance(rois, list):
            parsed = []
            for i, r in enumerate(rois):
                name = r.get("name", f"ROI{i+1}")
                w, h = r.get("size", [0, 0])
                dx, dy = r.get("offset", [0, 0])
                hflip = bool(r.get("hflip", False))
                parsed.append(dict(name=name, size=[int(w), int(h)], offset=[int(dx), int(dy)], hflip=hflip))
            return parsed
        if roi_px is not None:
            w, h = int(roi_px[0]), int(roi_px[1])
            dx, dy = int(roi_offset[0]), int(roi_offset[1])
            return [dict(name="ROI", size=[w, h], offset=[dx, dy], hflip=False)]
        return DEFAULT_ROIS

    def start(self):
        with self._lock:
            if self._run:
                return
            self._run = True
            self._paused = False
            self._thr = threading.Thread(target=self._loop, daemon=True)
            self._thr.start()

    def stop(self):
        with self._lock:
            self._run = False
        if self._thr:
            self._thr.join(timeout=2.0)
        self._release_camera()

    def pause(self):
        with self._lock:
            self._paused = True

    def resume(self):
        with self._lock:
            self._paused = False

    def get_detection(self, timeout: Optional[float] = None):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _open_camera(self):
        self._cap = open_camera(self.camera, self.prefer_res, self.prefer_fps)

    def _release_camera(self):
        if self._cap is None:
            return
        try:
            if self._cap[0] == "realsense":
                pipeline, rs = self._cap[1]; pipeline.stop()
            else:
                self._cap[1].release()
        except Exception:
            pass
        self._cap = None

    def _read(self):
        return read_frame_nonblocking(self._cap) if self._cap is not None else None

    def _try_decode_multi(self, bgr) -> List[str]:
        for r in self.rois:
            name = r["name"]; (rw, rh) = r["size"]; (dx, dy) = r["offset"]; hflip = r["hflip"]
            roi = crop_roi_center(bgr, int(rw), int(rh), int(dx), int(dy))
            if roi.size == 0:
                continue
            if hflip: roi = cv.flip(roi, 1)
            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

            if likely_has_code(gray):
                payloads = decode_payloads_robust(gray, max_count=4, time_budget_ms=120)
                if payloads:
                    log(f"{name} HIT")
                    return payloads
        return []

    def _loop(self):
        self._open_camera()
        self._last_decode_ts = 0.0
        decode_interval = self.decode_interval_base * 0.9
        miss_streak = 0
        iter_idx = 0
        log("Watcher loop start")
        while True:
            with self._lock:
                run = self._run; paused = self._paused
            if not run:
                break
            if paused:
                if self._cap is not None:
                    log("pause → release camera")
                    self._release_camera()
                time.sleep(0.05); continue
            if self._cap is None:
                self._open_camera()
                if self._cap is None:
                    time.sleep(0.2); continue

            now = time.monotonic()
            if now - self._last_decode_ts < decode_interval:
                time.sleep(0.001); continue
            self._last_decode_ts = now
            t0 = time.perf_counter()
            frame = self._read()
            t1 = time.perf_counter()
            payloads = self._try_decode_multi(frame) if frame is not None else []
            t2 = time.perf_counter()

            iter_idx += 1
            if iter_idx % self.log_every_decode == 0:
                log(f"decode#{iter_idx} read={(t1-t0)*1000:.2f} ms | roi+gray+decode={(t2-t1)*1000:.2f} ms | total={(t2-t0)*1000:.2f} ms | result={'O' if payloads else 'X'} | interval={decode_interval:.3f}s")

            if payloads:
                miss_streak = 0
                decode_interval = self.decode_interval_base * 0.9
                ts = time.time()
                self._q.put((ts, payloads))
                with self._lock:
                    self._paused = True
            else:
                miss_streak += 1
                decode_interval = min(self.max_backoff, decode_interval * 1.5) if miss_streak >= 3 else self.decode_interval_base * 0.9
        log("Watcher loop end")

# ===== 1초 고정 하트비트 Standalone =====
def _standalone():
    rois = _load_rois_from_config() or DEFAULT_ROIS
    cam = open_camera(DEFAULT_CAMERA, (1920,1080), 6)

    state = {"detected": False, "last_payloads": [], "lock": threading.Lock(), "running": True}

    def worker_decode():
        decode_iter = 0
        log("=== Worker start (decoding loop) ===")
        try:
            while state["running"]:
                t0 = time.perf_counter()
                frame = read_frame_nonblocking(cam)
                t1 = time.perf_counter()
                payloads = []
                if frame is not None:
                    for r in rois:
                        (rw, rh) = r.get("size",[0,0])
                        (dx, dy) = r.get("offset",[0,0])
                        hflip = bool(r.get("hflip", False))
                        roi = crop_roi_center(frame, int(rw), int(rh), int(dx), int(dy))
                        if roi.size == 0: continue
                        if hflip: roi = cv.flip(roi, 1)
                        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                        if likely_has_code(gray):
                            payloads = decode_payloads_robust(gray, max_count=4, time_budget_ms=120)
                            if payloads:
                                log("HIT")
                                break
                t2 = time.perf_counter()
                decode_iter += 1
                if LOG_EVERY_DECODE and decode_iter % LOG_EVERY_DECODE == 0:
                    log(f"decode#{decode_iter} read={(t1-t0)*1000:.2f} ms | roi+decode={(t2-t1)*1000:.2f} ms | result={'O' if payloads else 'X'}")
                with state["lock"]:
                    state["detected"] = bool(payloads)
                    state["last_payloads"] = payloads
        except Exception as e:
            log(f"[ERR] worker exception: {e}")

    def heartbeat_print():
        log("=== Heartbeat start (1s) ===")
        print("[INFO] 1초마다 O/X 출력. Ctrl+C 종료", flush=True)
        try:
            while state["running"]:
                with state["lock"]:
                    detected = state["detected"]
                print("O" if detected else "X", flush=True)
                time.sleep(1.0)
        except Exception as e:
            log(f"[ERR] heartbeat exception: {e}")

    thr_worker = threading.Thread(target=worker_decode, daemon=True)
    thr_beat = threading.Thread(target=heartbeat_print, daemon=True)
    thr_worker.start()
    thr_beat.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] 종료"); log("KeyboardInterrupt. Exiting...")
    finally:
        state["running"] = False
        thr_worker.join(timeout=2.0)
        thr_beat.join(timeout=2.0)
        if cam[0] == "realsense":
            pipeline, rs = cam[1]; pipeline.stop()
        else:
            cam[1].release()
        log("Camera released.")

# ---------------- Config Loader ----------------
def _load_rois_from_config() -> Optional[List[Dict[str, Any]]]:
    import json
    from pathlib import Path
    cand = [Path("datamatrix.yaml"), Path("datamatrix.yml"), Path("config.yaml"), Path("config.json")]
    for p in cand:
        if not p.exists(): continue
        try:
            if p.suffix.lower() in (".yaml", ".yml"):
                import yaml
                with p.open("r", encoding="utf-8") as f: cfg = yaml.safe_load(f) or {}
            else:
                with p.open("r", encoding="utf-8") as f: cfg = json.load(f)
            dm = cfg.get("datamatrix", {})
            rois = dm.get("rois")
            if rois: return rois
        except Exception:
            pass
    return None

if __name__ == "__main__":
    _standalone()
