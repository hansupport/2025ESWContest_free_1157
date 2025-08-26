# datamatrix.py
import time
import threading
import queue
from typing import Optional, Tuple, List
import numpy as np
import cv2 as cv

# ===== SETTINGS =====
ROI_RATIO = 0.60         # 중앙 정사각형 ROI (min(w,h) * 비율)
DECODE_INTERVAL = 0.20   # 디코딩 간격(초)
DEFAULT_CAMERA = "/dev/video2"  # 기본 카메라: 인덱스 2
# ====================

# OpenCV 최적화
try:
    cv.setUseOptimized(True)
    cv.setNumThreads(1)
except Exception:
    pass

# pylibdmtx 안전 임포트
def load_dmtx_decode():
    try:
        from pylibdmtx.pylibdmtx import decode
        return decode
    except Exception:
        pass
    try:
        import pylibdmtx.pylibdmtx as dmtx
        return dmtx.decode
    except Exception as e:
        raise ImportError(
            "pylibdmtx decode 로드 실패: " + str(e) +
            "\n(참고: sudo apt install libdmtx0a libdmtx-dev 후 "
            "pip install --no-binary :all: pylibdmtx==0.1.10)"
        )

_dm_decode = load_dmtx_decode()

class DMatrixWatcher:
    """
    - 별도 스레드에서 지속 프레임 캡처 + DataMatrix 디코딩
    - 디코딩 성공 시 (timestamp, payloads:list[str])를 큐로 전달하고 '일시정지'로 전환
    - 일시정지 동안 카메라 스트림 완전 stop → img2emb가 같은 카메라(2번)를 사용 가능
    - main의 resume() 호출 시 스트림 재시작
    """
    def __init__(self,
                 camera=DEFAULT_CAMERA,
                 prefer_res=(1920,1080),
                 prefer_fps=6,
                 roi_ratio: float = ROI_RATIO,
                 decode_interval: float = DECODE_INTERVAL):
        self.camera = camera
        self.prefer_res = prefer_res
        self.prefer_fps = prefer_fps
        self.roi_ratio = float(roi_ratio)
        self.decode_interval = float(decode_interval)

        self._cap: Optional[cv.VideoCapture] = None
        self._run = False
        self._paused = False
        self._lock = threading.Lock()
        self._q: "queue.Queue[Tuple[float, List[str]]]" = queue.Queue()
        self._thr: Optional[threading.Thread] = None
        self._last_decode_ts = 0.0

    # ---------- Camera ----------
    def _open_camera(self):
        cam = self.camera
        # 문자열 숫자면 정수로 변환
        if isinstance(cam, str) and cam.isdigit():
            cam = int(cam)
        # OpenCV V4L2로 오픈
        cap = cv.VideoCapture(cam, cv.CAP_V4L2)
        if not cap.isOpened():
            print(f"[Watcher] 카메라 열기 실패: {self.camera}")
            self._cap = None
            return
        # 해상도/FPS 설정
        w, h = self.prefer_res
        cap.set(cv.CAP_PROP_FRAME_WIDTH,  int(w))
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(h))
        cap.set(cv.CAP_PROP_FPS,          int(self.prefer_fps))
        # 첫 프레임 확인
        ok, _ = cap.read()
        if not ok:
            print("[Watcher] 첫 프레임 실패, 카메라 해제")
            cap.release()
            self._cap = None
            return
        print(f"[Watcher] Camera 시작: {self.camera} @ {w}x{h}@{self.prefer_fps}fps")
        self._cap = cap

    def _release_camera(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    # ---------- Public API ----------
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
        # 카메라는 루프에서 자동 재오픈

    def get_detection(self, timeout: Optional[float] = None) -> Optional[Tuple[float, List[str]]]:
        try:
            ts, payloads = self._q.get(timeout=timeout)
            return ts, payloads
        except queue.Empty:
            return None

    # ---------- Main loop ----------
    def _loop(self):
        self._open_camera()
        self._last_decode_ts = 0.0
        print("[Watcher] 시작. 디코딩 간격:", self.decode_interval, "초")

        while True:
            with self._lock:
                run = self._run
                paused = self._paused
            if not run:
                break

            # 일시정지: 카메라 완전 해제
            if paused:
                if self._cap is not None:
                    print("[Watcher] 일시정지: 카메라 해제")
                    self._release_camera()
                time.sleep(0.05)
                continue

            # 실행 상태: 카메라 없으면 재오픈
            if self._cap is None:
                self._open_camera()
                if self._cap is None:
                    time.sleep(0.2)
                    continue

            frame = self._read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            if now - self._last_decode_ts < self.decode_interval:
                continue
            self._last_decode_ts = now

            # ROI 추출
            h, w = frame.shape[:2]
            side = int(min(w, h) * self.roi_ratio)
            x0 = (w - side) // 2
            y0 = (h - side) // 2
            roi = frame[y0:y0+side, x0:x0+side]

            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

            # DataMatrix 디코딩
            try:
                res = _dm_decode(gray, max_count=4)
            except TypeError:
                res = _dm_decode(gray)

            if res:
                payloads = []
                for r in res:
                    try:
                        payloads.append(r.data.decode("utf-8", errors="replace").strip())
                    except Exception:
                        payloads.append(str(r.data))

                print("[Watcher] DETECTED:", payloads)
                self._q.put((now, payloads))
                with self._lock:
                    self._paused = True   # 자동 일시정지(→ 카메라 해제)
                # 다음 루프에서 카메라 해제됨

        print("[Watcher] 종료")
