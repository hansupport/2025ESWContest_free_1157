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
DEFAULT_CAMERA: Union[int,str] = 2

# RGB 전용 기본 ROI
DEFAULT_ROIS: List[Dict[str, Any]] = [
    {"name": "RGB_MAIN",   "size": [530, 530], "offset": [ +90, -130], "hflip": False},
    {"name": "RGB_L_MIRR", "size": [390, 500], "offset": [ -450, -140], "hflip": True },
    {"name": "RGB_R_MIRR", "size": [390, 490], "offset": [ +660, -140], "hflip": True },
]

# 타임스탬프 프리픽스 추가 후 즉시 flush 처리
def log(msg: str):
    ts = time.strftime("%H:%M:%S", time.localtime())
    sys.stderr.write(f"[{ts}] {msg}\n")
    sys.stderr.flush()

try:
    cv.setUseOptimized(True); cv.setNumThreads(1)
    log("OpenCV optimized=True, threads=1")
except Exception:
    pass

# pylibdmtx decode 심볼 동적 로딩. import → 폴백, 실패 시 설치 가이드 포함 예외
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

# GStreamer appsink→ 실패 시 V4L2 → 최종 RealSense 컬러 스트림. (type, handle) 튜플 반환.
def open_camera(camera=DEFAULT_CAMERA, prefer_res=(1920,1080), prefer_fps=6):
    t_all0 = time.perf_counter()
    cam_id = 0 if camera in (None, "auto") else camera

    # 1) GStreamer appsink
    try:
        if "GStreamer: YES" in cv.getBuildInformation():
            device = cam_id if (isinstance(cam_id, str) and cam_id.startswith("/dev/video")) else f"/dev/video{int(cam_id)}"
            w, h = int(prefer_res[0]), int(prefer_res[1])
            fps = int(prefer_fps)

            gst_candidates = [
                # MJPG 경로
                (
                    f"v4l2src device={device} io-mode=2 ! "
                    f"image/jpeg,framerate={fps}/1 ! jpegdec ! "
                    f"videoscale ! video/x-raw,width={w},height={h} ! "
                    f"appsink drop=true max-buffers=1 sync=false"
                ),
                # YUY2 경로
                (
                    f"v4l2src device={device} io-mode=2 ! "
                    f"video/x-raw,format=YUY2,width={w},height={h},framerate={fps}/1 ! "
                    f"videoconvert ! "
                    f"appsink drop=true max-buffers=1 sync=false"
                ),
            ]
            for gst in gst_candidates:
                cap = cv.VideoCapture(gst, cv.CAP_GSTREAMER)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        log(f"GStreamer appsink open {w}x{h}@{fps} in {(time.perf_counter()-t_all0)*1000:.2f} ms")
                        return ("webcam", cap)
            log("[WARN] GStreamer appsink 실패 → V4L2 폴백")
    except Exception as e:
        log(f"[WARN] GStreamer 경로 예외: {e}")

    # 2) V4L2 폴백 열기, 성공 시 ("webcam", cap) 반환.
    cap = cv.VideoCapture(cam_id, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  int(prefer_res[0]))
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(prefer_res[1]))
    cap.set(cv.CAP_PROP_FPS,          int(prefer_fps))
    try:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    try:
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    ok, _ = cap.read()
    if ok:
        log(f"V4L2 open {prefer_res[0]}x{prefer_res[1]}@{prefer_fps} in {(time.perf_counter()-t_all0)*1000:.2f} ms")
        return ("webcam", cap)

    # 3) RealSense 컬러 파이프라인 폴백, 성공 시 ("realsense", (pipeline, rs)) 반환.
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
                try: color_sensor.set_option(rs.option.frames_queue_size, 1)
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

# 최신 프레임 우선 읽기
def read_frame_nonblocking(cam):
    if cam[0] == "realsense":
        pipeline, rs = cam[1]
        for _ in range(3):
            try:
                _ = pipeline.poll_for_frames()
            except Exception:
                break
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
        try:
            for _ in range(6):
                cap.grab()
        except Exception:
            pass
        ret, frame = cap.read()
        if not ret: return None
        return frame

# 코드 존재 가능성 판단
def likely_has_code(gray):
    lap_var = cv.Laplacian(gray, cv.CV_64F).var()
    if lap_var < 15.0:
        return False
    edges = cv.Canny(gray, 50, 150)
    if (edges > 0).mean() < 0.01:
        return False
    return True

# 중앙 기준 ROI 절대 좌표 크롭, BGR 서브이미지 반환
def crop_roi_center(bgr, roi_w, roi_h, dx, dy):
    h, w = bgr.shape[:2]
    x0 = (w - roi_w) // 2 + dx
    y0 = (h - roi_h) // 2 + dy
    x0 = max(0, min(w - roi_w, x0))
    y0 = max(0, min(h - roi_h, y0))
    return bgr[y0:y0+roi_h, x0:x0+roi_w]

# CLAHE 대비 보정
def _clahe(gray):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

# 샤프닝 커널 적용
def _sharpen(gray):
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
    return cv.filter2D(gray, -1, k)

# 적응형 이진화
def _binarize(gray):
    return cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, 21, 5)

# 0/90/180/270 회전 리스트 생성, 회전된 이미지 리스트 반환
def _rot90s(img):
    return [
        img,
        cv.rotate(img, cv.ROTATE_180),
        cv.rotate(img, cv.ROTATE_90_CLOCKWISE),
        cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE),
    ]

# 사변형 검출해 정사각 투시 보정. (성공: 정사각 gray 반환. 실패: None 반환)
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

# robust 디코드
def decode_payloads_robust(gray, max_count=4, time_budget_ms=120):
    t_start = time.perf_counter()
    def left_ms():
        return max(0.0, time_budget_ms - (time.perf_counter()-t_start)*1000.0)

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
    for img in _rot90s(sharp):
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

# 초고속 4방향 시도. 전처리/업샘플/CLAHE/이진화/디워프 없음. 문자열 리스트 반환.
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

# 카메라/ROI/주기/백오프 관리 + 백그라운드 디코드 큐 제공
class DMatrixWatcher:
    # 초기화
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

    # ROI 파라미터 파싱 (직접 전달 rois 우선-> roi_px/offset 단일 ROI/ 없으면 기본값)
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

    # 백그라운드 워커 시작
    def start(self):
        with self._lock:
            if self._run:
                return
            self._run = True
            self._paused = False
            self._thr = threading.Thread(target=self._loop, daemon=True)
            self._thr.start()

    # 워커 정지
    def stop(self):
        with self._lock:
            self._run = False
        if self._thr:
            self._thr.join(timeout=2.0)
        self._release_camera()

    # 일시정지 플래그 on
    def pause(self):
        with self._lock:
            self._paused = True

    # 일시정지 해제
    def resume(self):
        with self._lock:
            self._paused = False

    # 디코드 결과 큐 pop
    def get_detection(self, timeout: Optional[float] = None):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    # 내부 카메라 오픈
    def _open_camera(self):
        self._cap = open_camera(self.camera, self.prefer_res, self.prefer_fps)

    # 내부 카메라 해제
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

    # 프레임 읽기 헬퍼
    def _read(self):
        return read_frame_nonblocking(self._cap) if self._cap is not None else None

    # 다중 ROI 순회 디코드
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

    # 일시정지면 카메라 해제 후 sleep
    # 주기적 디코드: 프레임 읽기 → 다중 ROI 디코드 → 로그
    # 성공 시 큐에 put하고 일시정지, 실패 누적 시 백오프 확대
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

if __name__ == "__main__":
    _standalone()
