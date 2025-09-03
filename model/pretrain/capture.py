# 3 ROI(lm/rm/c) 시각화(최소 포함 박스) + 각 ROI 라벨/박스
# 'd'+Enter → c/lm/rm 각각 저장(3장). 저장은 비동기 워커가 처리 → 촬영 속도 유지
# 저장 완료 시점에: [batch] saved 3 files (total=..., sets=...) 출력
# - 캡처 버퍼 1장으로 지연 누적 방지, MJPG 우선
import sys, time, select, queue, threading
from pathlib import Path
import cv2

WINDOW_TITLE = "D435f 3-ROI View"
ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT / "pretrain" / "img_data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "/dev/video2"
WIDTH, HEIGHT, FPS = 1920, 1080, 6

ROI_LM_WH  = (320, 470); ROI_LM_OFF = (-420, -160)
ROI_RM_WH  = (330, 480); ROI_RM_OFF = (630,  -180)
ROI_C_WH   = (520, 550); ROI_C_OFF  = (90,   -160)

JPEG_QUALITY = 95

# 색상(BGR)
COL_BOX_LM = (255, 128, 0)
COL_BOX_RM = (0, 128, 255)
COL_BOX_C  = (0, 255, 0)
COL_BBOX   = (255, 255, 255)

# OpenCV 스레드 최적화
try:
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
except Exception:
    pass


# GStreamer 파이프라인 문자열 생성, MJPG 우선. appsink drop/max-buffers/sync로 지연 방지
def build_gst_pipeline(dev: str, w: int, h: int, fps: int, mjpg: bool) -> str:
    dev = str(dev)
    if mjpg:
        return (
            f"v4l2src device={dev} io-mode=2 ! "
            f"image/jpeg, width={w}, height={h}, framerate={fps}/1 ! "
            f"jpegdec ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
        )
    else:
        return (
            f"v4l2src device={dev} io-mode=2 ! "
            f"video/x-raw, format=YUY2, width={w}, height={h}, framerate={fps}/1 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )


# 카메라 열기 (V4L2 → GStreamer(MJPG) → GStreamer(YUY2))
def open_camera(device: str, w: int, h: int, fps: int):
    cam_id = device
    if isinstance(device, str) and device.startswith("/dev/video"):
        try: cam_id = int(device.replace("/dev/video", ""))
        except Exception: pass

    # V4L2 시도
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        ok, _ = cap.read()
        if ok: return cap
        cap.release()

    gst = build_gst_pipeline(device, w, h, fps, mjpg=True)
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ok, _ = cap.read()
        if ok: return cap
        cap.release()

    gst = build_gst_pipeline(device, w, h, fps, mjpg=False)
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ok, _ = cap.read()
        if ok: return cap
        cap.release()

    return None


# 중심 기반 ROI를 절대 좌표(x,y,w,h)로 변환, 이미지 경계로 클램핑
def compute_center_roi_xy(img_shape, roi_wh, roi_off):
    H, W = img_shape[:2]
    rw, rh = map(int, roi_wh)
    dx, dy = map(int, roi_off)
    rw = max(1, min(rw, W)); rh = max(1, min(rh, H))
    cx, cy = W // 2 + dx, H // 2 + dy
    x = max(0, min(W - rw, cx - rw // 2))
    y = max(0, min(H - rh, cy - rh // 2))
    return x, y, rw, rh


# 여러 사각형의 최소 포함 박스 반환, 결과 클램핑핑
def union_rects(rects, img_shape):
    H, W = img_shape[:2]
    if not rects: return (0, 0, W, H)
    x1 = min([r[0] for r in rects]); y1 = min([r[1] for r in rects])
    x2 = max([r[0] + r[2] for r in rects]); y2 = max([r[1] + r[3] for r in rects])
    x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
    x2 = max(1, min(W, x2));     y2 = max(1, min(H, y2))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


# ROI 박스와 라벨 오버레이, 라벨 배경 상자 포함
def draw_roi_rect(dst_img, rect, color, label=None, thickness=2, font_scale=0.6):
    x, y, w, h = rect
    cv2.rectangle(dst_img, (x, y), (x + w, y + h), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness))
        tx, ty = x + 5, max(th + 6, y - 6)
        cv2.rectangle(dst_img, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), color, -1)
        cv2.putText(dst_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, thickness), cv2.LINE_AA)


# stdin 비차단 라인 읽기, select로 타임아웃 처리
def stdin_readline_nonblock(timeout_sec=0.001):
    r, _, _ = select.select([sys.stdin], [], [], timeout_sec)
    if r: return sys.stdin.readline().strip()
    return None


# 비동기 JPEG 저장 워커 클래스. 큐 소비하여 저장
class AsyncSaver(object):
    # 초기화
    def __init__(self, jpeg_quality=80, max_queue=16):
        self.q = queue.Queue(max_queue)
        self.quality = int(jpeg_quality)
        self.total_saved = 0
        self._lock = threading.Lock()
        self._done = {}
        self._running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    # 종료
    def stop(self):
        self._running = False
        try: self.q.put_nowait(None)
        except Exception: pass
        try: self.th.join(timeout=2.0)
        except Exception: pass

    # 태그 등록 후 각 이미지를 copy하여 큐에 push함
    def enqueue_batch(self, tag, items):
        self._done[tag] = 0
        for (fn, im) in items:
            self.q.put((tag, fn, im.copy()))

    # 내부 루프. 큐 소비
    def _loop(self):
        while self._running:
            try:
                task = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if task is None:
                continue
            tag, fn, im = task
            ok = cv2.imwrite(fn, im, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            with self._lock:
                if ok: self.total_saved += 1
                print("[save] {}".format(Path(fn).name))
                self._done[tag] = self._done.get(tag, 0) + (1 if ok else 0)
                if self._done[tag] >= 3:
                    sets = self.total_saved // 3
                    print("[batch] saved 3 files (total={}, sets={})".format(self.total_saved, sets))
                    del self._done[tag]
            self.q.task_done()

# 프레임 읽기 → 3 ROI 계산/시각화 → 화면 표시 → stdin 명령 처리
#  finally에서 자원 정리(cap/saver/window)
def main():
    print("[open] device=", DEVICE, "res=", (WIDTH, HEIGHT), "fps=", FPS)
    print("[hint] 'd'+Enter=3뷰 저장 / 'f'+Enter=전체화면 / 'q'+Enter=종료")
    cap = open_camera(DEVICE, WIDTH, HEIGHT, FPS)
    if not cap or not cap.isOpened():
        print("[fatal] failed to open camera", DEVICE)
        return

    saver = AsyncSaver(jpeg_quality=JPEG_QUALITY, max_queue=32)

    # 윈도우
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    is_fullscreen = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.003)
                continue

            # ROI 절대좌표
            x_lm, y_lm, w_lm, h_lm = compute_center_roi_xy(frame.shape, ROI_LM_WH, ROI_LM_OFF)
            x_rm, y_rm, w_rm, h_rm = compute_center_roi_xy(frame.shape, ROI_RM_WH, ROI_RM_OFF)
            x_c,  y_c,  w_c,  h_c  = compute_center_roi_xy(frame.shape, ROI_C_WH,  ROI_C_OFF)

            # 3 ROI 포함 최소 바운딩 박스 시각화
            bbx, bby, bbw, bbh = union_rects(
                [(x_lm, y_lm, w_lm, h_lm), (x_rm, y_rm, w_rm, h_rm), (x_c, y_c, w_c, h_c)],
                frame.shape
            )
            crop = frame[bby:bby+bbh, bbx:bbx+bbw].copy()

            lm_local = (x_lm - bbx, y_lm - bby, w_lm, h_lm)
            rm_local = (x_rm - bbx, y_rm - bby, w_rm, h_rm)
            c_local  = (x_c  - bbx, y_c  - bby, w_c,  h_c)

            draw_roi_rect(crop, c_local,  COL_BOX_C,  "CENTER (c)", thickness=2)
            draw_roi_rect(crop, lm_local, COL_BOX_LM, "LEFT MIRROR (lm)", thickness=2)
            draw_roi_rect(crop, rm_local, COL_BOX_RM, "RIGHT MIRROR (rm)", thickness=2)
            cv2.rectangle(crop, (1, 1), (crop.shape[1]-2, crop.shape[0]-2), COL_BBOX, 1)

            cv2.imshow(WINDOW_TITLE, crop)
            cv2.waitKey(1)

            # 입력
            cmd = stdin_readline_nonblock(0.001)
            if not cmd:
                continue
            key = cmd.strip().lower()
            if key == 'q':
                break
            elif key == 'f':
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(
                    WINDOW_TITLE,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                )
                print("[ui] fullscreen={}".format('ON' if is_fullscreen else 'OFF'))
            elif key == 'd':
                ts = time.strftime("%Y%m%d_%H%M%S")
                c_img  = frame[y_c:y_c+h_c,   x_c:x_c+w_c]
                lm_img = frame[y_lm:y_lm+h_lm, x_lm:x_lm+w_lm]
                rm_img = frame[y_rm:y_rm+h_rm, x_rm:x_rm+w_rm]
                if lm_img.size > 0: lm_img = cv2.flip(lm_img, 1)
                if rm_img.size > 0: rm_img = cv2.flip(rm_img, 1)

                items = []
                items.append((str(SAVE_DIR / ("capture_{}_c.jpg".format(ts))),  c_img))
                items.append((str(SAVE_DIR / ("capture_{}_lm.jpg".format(ts))), lm_img))
                items.append((str(SAVE_DIR / ("capture_{}_rm.jpg".format(ts))), rm_img))

                saver.enqueue_batch(ts, items)

    finally:
        try: cap.release()
        except Exception: pass
        try:
            saver.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[exit] bye")


if __name__ == "__main__":
    main()
