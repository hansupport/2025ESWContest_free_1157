# img2emb.py (ONNXRuntime CPU-only)
# Jetson Nano: OpenCV CPU + ONNXRuntime
# 최적화 포인트:
#  - ONNXRuntime CPUExecutionProvider
#  - ROI: 절대좌표(ROI=(x,y,w,h))와 중심기준(ROI_PX=(w,h), ROI_OFF=(dx,dy)) 모두 지원
#  - 내부 구간 프로파일(--profile)
#  - 엔드투엔드 워밍업(--e2e_warmup), 사전 grab(--pregrab)
#  - OpenCV 스레드 초기화 비용 축소(cv2.setNumThreads(1))
#
# NOTE:
#  - PyTorch 관련 옵션(--no_cudnn, --no_fp16, --channels_last, --pinned 등)은
#    하위호환을 위해 파싱만 하고 무시됩니다(경고 출력).

import os, sys, time, glob, csv, argparse, select
from pathlib import Path
import numpy as np
import cv2

# ===== ONNXRuntime =====
try:
    import onnxruntime as ort
except Exception as e:
    print("[ERR] onnxruntime 를 불러올 수 없습니다. `pip install onnxruntime` 후 다시 실행하세요.", file=sys.stderr)
    raise

# OpenCV 스레드 풀 초기화 고정
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# ========= 전역 기본값 =========
PRETRAINED = True
PRETRAINED_MODE = "onnx"               # 호환용 필드(미사용). onnx 고정.
PRETRAINED_PATH = "mobilenetv3_small_emb.onnx"

EMBED_DIM = 256
INPUT_SIZE = 224
WIDTH_SCALE = 1.0                      # 호환용(ONNX에서 미사용)
USE_FP16 = False                       # 호환용(ONNX에서 미사용)
CHANNELS_LAST = False                  # 호환용(ONNX에서 미사용)
FRAME_SKIP_N = 1
CUDNN_BENCHMARK = False                # 호환용(ONNX에서 미사용)
WARMUP_STEPS = 0                       # (batched images path에서만 의미)
NO_DEPTHWISE = False                   # 호환용(ONNX에서 미사용)
NO_BN = False                          # 호환용(ONNX에서 미사용)
USE_PINNED = False                     # 호환용(ONNX에서 미사용)

# ROI 기본값
# 1) 절대좌표 방식: (x,y,w,h)
ROI = (280, 96, 288, 288)
# 2) 중심기준 방식: (w,h), (dx,dy) — 두 값이 모두 설정되면 이 방식을 우선 사용
ROI_PX = None
ROI_OFF = None

# ========= 시간/로그 =========
T0 = time.perf_counter()
VERBOSE = False
TIME_LOG = True
PROFILE = False
def vlog(msg: str):
    if VERBOSE:
        if TIME_LOG: print(f"[{time.perf_counter()-T0:7.3f}s] {msg}", flush=True)
        else:        print(msg, flush=True)

# ========= 전처리 =========
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    return x / (n + eps)

def _safe_crop_xywh(img, x, y, w, h):
    H, W = img.shape[:2]
    x2, y2 = x + w, y + h
    x  = max(0, x);  y  = max(0, y)
    x2 = min(W, x2); y2 = min(H, y2)
    if x >= x2 or y >= y2:
        raise ValueError(f"ROI out of bounds: img=({W}x{H}), roi={(x,y,w,h)}")
    return img[y:y2, x:x2]

def _compute_center_roi_xy(img, px_wh, off_xy):
    H, W = img.shape[:2]
    rw, rh = map(int, px_wh)
    dx, dy = map(int, off_xy)
    rw = max(1, min(rw, W))
    rh = max(1, min(rh, H))
    cx = W // 2 + dx
    cy = H // 2 + dy
    x = max(0, min(W - rw, cx - rw // 2))
    y = max(0, min(H - rh, cy - rh // 2))
    return x, y, rw, rh

def safe_crop(img, roi):
    if roi is None:
        return img
    x, y, w, h = map(int, roi)
    return _safe_crop_xywh(img, x, y, w, h)

def preprocess_bgr(img_bgr, size=INPUT_SIZE):
    # ROI 우선순위: 중심기준(ROI_PX/ROI_OFF) → 절대좌표(ROI) → 전체
    if ROI_PX is not None and ROI_OFF is not None:
        x, y, w, h = _compute_center_roi_xy(img_bgr, ROI_PX, ROI_OFF)
        img = _safe_crop_xywh(img_bgr, x, y, w, h)
    elif ROI is not None:
        img = safe_crop(img_bgr, ROI)
    else:
        img = img_bgr

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (img.shape[1], img.shape[0]) != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0).astype(np.float32)  # [1,3,H,W]

def imread_bgr(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None: img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img

def list_images(patterns):
    paths = []
    for pat in patterns: paths.extend(glob.glob(pat))
    return sorted({p for p in paths if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))})

def save_matrix_and_index(pairs, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    embs = [e for _, e in pairs if e is not None]
    names = [p for p, e in pairs if e is not None]
    if not embs:
        print("저장할 임베딩이 없습니다.", file=sys.stderr); return
    np.save(str(out / "embeddings.npy"), np.stack(embs, 0).astype(np.float32))
    with open(str(out / "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["row","path"])
        for i, n in enumerate(names): w.writerow([i, n])
    print("저장 완료:", out / "embeddings.npy", ",", out / "index.csv")

# ========= ONNX Embedder =========
class OnnxEmbedder:
    def __init__(self, onnx_path, size=INPUT_SIZE, out_dim=EMBED_DIM):
        if not onnx_path or not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 파일이 없습니다: {onnx_path}")
        self.size = int(size)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1  # Jetson Nano: 과도한 스레드 방지
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )
        self.inp = self.session.get_inputs()[0].name
        self.out = self.session.get_outputs()[0].name
        self.out_dim = int(out_dim)

    def _run(self, X: np.ndarray) -> np.ndarray:
        # X: [N,3,H,W], float32
        y = self.session.run([self.out], {self.inp: X})[0]  # [N,D]
        return y

    def embed_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        X = preprocess_bgr(img_bgr, size=self.size)                # [1,3,H,W]
        t1 = time.perf_counter()
        f = self._run(X)                                           # [1,D]
        t2 = time.perf_counter()
        v = l2_normalize(f[0])                                     # [D]
        t3 = time.perf_counter()

        if PROFILE:
            print("[profile]",
                  f"preproc_ms={(t1-t0)*1000:.1f}",
                  f"forward_ms={(t2-t1)*1000:.1f}",
                  f"norm_ms={(t3-t2)*1000:.1f}",
                  f"total_ms={(t3-t0)*1000:.1f}",
                  flush=True)
        return v.astype(np.float32)

    def embed_batch_np(self, batch_np) -> np.ndarray:
        # batch_np: list of [1,3,H,W] np.float32
        X = np.concatenate(batch_np, 0)            # [N,3,H,W]
        f = self._run(X)                            # [N,D]
        return l2_normalize(f).astype(np.float32)   # [N,D]

# ========= ROI 유틸: 중심기준 세터 =========
def set_center_roi(px_wh, offset_xy):
    global ROI_PX, ROI_OFF, ROI
    ROI_PX = tuple(map(int, px_wh)) if px_wh is not None else None
    ROI_OFF = tuple(map(int, offset_xy)) if offset_xy is not None else None
    # 중심기준을 쓰면 ROI 절대좌표는 비활성화
    if ROI_PX is not None and ROI_OFF is not None:
        ROI = None
    vlog(f"set_center_roi px={ROI_PX} off={ROI_OFF}")

# ========= Embedder factory =========
def build_embedder_from_flags():
    if not PRETRAINED:
        raise RuntimeError("--pretrained=1 과 --pretrained_path=<onnx> 를 지정하세요.")
    return OnnxEmbedder(PRETRAINED_PATH, size=INPUT_SIZE, out_dim=EMBED_DIM)

# ========= 카메라 =========
def build_gst_pipeline(dev, w=None, h=None, fps=None, pixfmt="YUYV"):
    parts=[f"v4l2src device={dev} io-mode=2"]
    if pixfmt and pixfmt.upper()=="MJPG":
        caps="image/jpeg"
        if any([w,h,fps]):
            wh=[]
            if w: wh.append(f"width={int(w)}")
            if h: wh.append(f"height={int(h)}")
            if fps: wh.append(f"framerate={int(fps)}/1")
            caps += ", " + ", ".join(wh)
        parts+=[f"! {caps}", "! jpegdec"]
    else:
        wh=["format=YUY2"]
        if w: wh.append(f"width={int(w)}")
        if h: wh.append(f"height={int(h)}")
        if fps: wh.append(f"framerate={int(fps)}/1")
        parts += [f"! video/x-raw, {', '.join(wh)}"]
    parts += ["! videoconvert", "! appsink"]
    return " ".join(parts)

def open_camera(cam, backend="auto", w=None, h=None, fps=None, pixfmt="YUYV"):
    if isinstance(cam, str) and not cam.isdigit() and ("v4l2src" in cam or "nvarguscamerasrc" in cam):
        vlog("[cam] using provided GStreamer pipeline")
        return cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
    cam_id = int(cam) if isinstance(cam, str) and cam.isdigit() else cam
    try_flag = cv2.CAP_V4L2 if backend!="gstreamer" else cv2.CAP_GSTREAMER
    vlog(f"[cam] try V4L2 backend={try_flag}")
    cap = cv2.VideoCapture(cam_id, try_flag)
    if cap.isOpened():
        if w: cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        if h: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
        if fps: cap.set(cv2.CAP_PROP_FPS, int(fps))
        if pixfmt:
            fourcc = cv2.VideoWriter_fourcc(*pixfmt.upper())
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        ret, _ = cap.read()
        if ret:
            vlog("[cam] V4L2 open OK"); return cap
        cap.release(); vlog("[cam] V4L2 read failed, fallback to GStreamer")
    dev = f"/dev/video{cam_id}" if isinstance(cam_id,int) else cam_id
    gst = build_gst_pipeline(dev, w, h, fps, pixfmt=pixfmt or "YUYV")
    vlog(f"[cam] try GStreamer: {gst}")
    return cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

# ========= 유틸 =========
def stdin_readline_nonblock(timeout_sec=0.1):
    r,_,_ = select.select([sys.stdin], [], [], timeout_sec)
    if r: return sys.stdin.readline().strip()
    return None

def print_embed(v, ms=None, print_k=8):
    l2 = float(np.linalg.norm(v))
    head = np.round(v[:max(0,print_k)], 4) if print_k>0 else np.array([])
    if ms is None:
        print(f"shape={v.shape}, L2={l2:.6f}, first{print_k}={head}", flush=True)
    else:
        print(f"shape={v.shape}, L2={l2:.6f}, time_ms={ms:.1f}, first{print_k}={head}", flush=True)

def roi_snapshot_from_cap(cap, out_path="ROI_snapshot.jpg", annotate=True):
    if not cap or not cap.isOpened():
        print("[roi_snap] 카메라 열기 실패", file=sys.stderr); return False
    ok, img = cap.read()
    if not ok:
        print("[roi_snap] 프레임 획득 실패", file=sys.stderr); return False
    draw = img.copy()
    # 중심기준이 설정되어 있으면 그것 기준으로 박스 그려줌
    if ROI_PX is not None and ROI_OFF is not None:
        x, y, w, h = _compute_center_roi_xy(img, ROI_PX, ROI_OFF)
        cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        info = f"{img.shape[1]}x{img.shape[0]} ROI_CENTER(px={ROI_PX}, off={ROI_OFF}) -> xywh=({x},{y},{w},{h})"
    elif ROI is not None:
        x, y, w, h = map(int, ROI)
        cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        info = f"{img.shape[1]}x{img.shape[0]} ROI_XYWH={ROI}"
    else:
        info = f"{img.shape[1]}x{img.shape[0]} ROI=None"
    if annotate:
        cv2.putText(draw, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite(out_path, draw)
    print(f"[roi_snap] 저장 완료: {out_path}")
    return True

def diag_env():
    print("[env]", "onnxruntime:", ort.__version__)
    try:
        print("[env]", "providers:", ort.get_available_providers())
    except Exception:
        pass
    print("[env]", "cv2:", cv2.__version__)
    print("[env]", "ROI_XYWH:", ROI, "ROI_CENTER(px/off):", ROI_PX, ROI_OFF)

def diag_fps_headless(emb, cap, frames=60):
    if not cap or not cap.isOpened():
        print("[fps] 카메라 열기 실패", file=sys.stderr); return
    t0 = time.perf_counter(); n = 0
    while n < frames:
        ok, bgr = cap.read()
        if not ok: break
        if FRAME_SKIP_N <= 1 or (n % FRAME_SKIP_N == 0):
            _ = emb.embed_bgr(bgr)
        n += 1
    fps = n / (time.perf_counter() - t0 + 1e-6)
    print(f"[fps] frames={n}, avg_fps={fps:.2f} (FRAME_SKIP_N={FRAME_SKIP_N})", flush=True)

# ========= E2E 워밍업 =========
def e2e_warmup(emb, cap, n=30, pregrab=5):
    if not cap or not cap.isOpened() or n <= 0: return
    for _ in range(max(0, pregrab)):
        cap.grab()
    t0 = time.perf_counter(); cnt = 0
    while cnt < n:
        ok, bgr = cap.read()
        if not ok: break
        _ = emb.embed_bgr(bgr)
        cnt += 1
    dt = (time.perf_counter()-t0)/max(1,cnt)
    print(f"[warmup] e2e frames={cnt}, avg_ms={dt*1000:.1f}", flush=True)

# ========= 데모 =========
def manual_demo_headless(emb, cap, print_k=8, save_npy=None, save_img=None):
    if not cap or not cap.isOpened():
        print("카메라/비디오 열기 실패", file=sys.stderr); return
    print("headless manual: 's'+Enter=infer, 'q'+Enter=quit", flush=True)
    if save_npy: Path(save_npy).mkdir(parents=True, exist_ok=True)
    if save_img: Path(save_img).mkdir(parents=True, exist_ok=True)
    while True:
        cap.grab()
        cmd = stdin_readline_nonblock(0.1)
        if cmd is None: continue
        if cmd.lower() == 'q': break
        if cmd.lower() == 's':
            ok, bgr = cap.read()
            if not ok:
                print("프레임 획득 실패", file=sys.stderr); continue
            t0 = time.perf_counter()
            v = emb.embed_bgr(bgr)
            ms = (time.perf_counter()-t0)*1000.0
            print_embed(v, ms=ms, print_k=print_k)
            if save_npy:
                ts = time.strftime("%Y%m%d_%H%M%S")
                np.save(str(Path(save_npy) / f"emb_{ts}.npy"), v.astype(np.float32))
            if save_img:
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(Path(save_img) / f"img_{ts}.jpg"), bgr)

def oneshot(emb, cap, print_k=8, save_npy=None, save_img=None):
    if not cap or not cap.isOpened():
        print("카메라/비디오 열기 실패", file=sys.stderr); return
    ok, bgr = cap.read()
    if not ok:
        print("프레임 획득 실패", file=sys.stderr); return
    t0 = time.perf_counter()
    v = emb.embed_bgr(bgr)
    ms = (time.perf_counter()-t0)*1000.0
    print_embed(v, ms=ms, print_k=print_k)
    if save_npy:
        Path(save_npy).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        np.save(str(Path(save_npy) / f"emb_{ts}.npy"), v.astype(np.float32))
    if save_img:
        Path(save_img).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(Path(save_img) / f"img_{ts}.jpg"), bgr)

def realtime_demo_headless(emb, cap, print_every=30):
    if not cap or not cap.isOpened():
        print("카메라/비디오 열기 실패", file=sys.stderr); return
    t0 = time.perf_counter(); n=0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if FRAME_SKIP_N<=1 or (n%FRAME_SKIP_N==0):
            _ = emb.embed_bgr(bgr)
        n+=1
        if n % print_every == 0:
            fps = n / (time.perf_counter()-t0+1e-6)
            print(f"fps≈{fps:.2f}", flush=True)

# ========= CLI =========
def main():
    global PRETRAINED, PRETRAINED_MODE, PRETRAINED_PATH
    global INPUT_SIZE, EMBED_DIM, WIDTH_SCALE, USE_FP16, CHANNELS_LAST, FRAME_SKIP_N
    global VERBOSE, CUDNN_BENCHMARK, WARMUP_STEPS, TIME_LOG, PROFILE
    global NO_DEPTHWISE, NO_BN, USE_PINNED
    global ROI, ROI_PX, ROI_OFF

    ap = argparse.ArgumentParser()
    # 로깅/성능
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--time_log", action="store_true")
    ap.add_argument("--no_time_log", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--cudnn_benchmark", action="store_true")  # ignored
    ap.add_argument("--no_cudnn", action="store_true", help="(무시) PyTorch 전용")
    ap.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    # 모델/전역
    ap.add_argument("--pretrained", type=int, default=int(PRETRAINED))
    ap.add_argument("--pretrained_mode", type=str, default=PRETRAINED_MODE)  # ignored
    ap.add_argument("--pretrained_path", type=str, default=PRETRAINED_PATH)
    ap.add_argument("--size", type=int, default=INPUT_SIZE)
    ap.add_argument("--out_dim", type=int, default=EMBED_DIM)
    ap.add_argument("--width", type=float, default=WIDTH_SCALE)  # ignored
    ap.add_argument("--no_fp16", action="store_true")            # ignored
    ap.add_argument("--channels_last", action="store_true")      # ignored
    ap.add_argument("--frame_skip", type=int, default=FRAME_SKIP_N)
    ap.add_argument("--no_depthwise", action="store_true")       # ignored
    ap.add_argument("--no_bn", action="store_true")              # ignored
    ap.add_argument("--pinned", action="store_true")             # ignored
    # 입출력 동작
    ap.add_argument("--images", nargs="+")
    ap.add_argument("--outdir", type=str, default="emb_out")
    ap.add_argument("--camera", type=str, default=None)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--manual", action="store_true")
    ap.add_argument("--oneshot", action="store_true")
    ap.add_argument("--no_window", action="store_true")
    # 카메라
    ap.add_argument("--cam_backend", type=str, default="auto", help="auto|v4l2|gstreamer")
    ap.add_argument("--cam_w", type=int, default=None)
    ap.add_argument("--cam_h", type=int, default=None)
    ap.add_argument("--cam_fps", type=int, default=None)
    ap.add_argument("--cam_pixfmt", type=str, default="YUYV", help="YUYV|MJPG|AUTO")
    # ROI 옵션
    ap.add_argument("--roi", type=int, nargs=4, default=None, help="절대좌표 ROI: x y w h")
    ap.add_argument("--roi_px", type=int, nargs=2, default=None, help="중심기준 ROI 폭높이: w h")
    ap.add_argument("--roi_off", type=int, nargs=2, default=None, help="중심기준 ROI 오프셋: dx dy")
    # 출력 제어
    ap.add_argument("--print_k", type=int, default=8)
    ap.add_argument("--save_npy", type=str, default=None)
    ap.add_argument("--save_img", type=str, default=None)
    # 진단
    ap.add_argument("--env_check", action="store_true")
    ap.add_argument("--fps_test", type=int, default=0)
    # ROI 스냅샷
    ap.add_argument("--roi_snap", type=str, default=None)
    # E2E 워밍업
    ap.add_argument("--e2e_warmup", type=int, default=0, help="카메라 프레임으로 엔드투엔드 워밍업 반복 횟수")
    ap.add_argument("--pregrab", type=int, default=5, help="워밍업 전에 grab만 수행할 프레임 수")

    args = ap.parse_args()

    VERBOSE = args.verbose
    TIME_LOG = True
    if args.no_time_log: TIME_LOG = False
    elif args.time_log:  TIME_LOG = True
    PROFILE = bool(args.profile)
    CUDNN_BENCHMARK = bool(args.cudnn_benchmark)
    WARMUP_STEPS = max(0, int(args.warmup))
    PRETRAINED = bool(int(args.pretrained))
    PRETRAINED_MODE = args.pretrained_mode
    PRETRAINED_PATH = args.pretrained_path
    INPUT_SIZE = int(args.size); EMBED_DIM = int(args.out_dim)
    WIDTH_SCALE = float(args.width)
    USE_FP16 = not args.no_fp16
    CHANNELS_LAST = bool(args.channels_last); FRAME_SKIP_N = max(1, int(args.frame_skip))
    NO_DEPTHWISE = bool(args.no_depthwise)
    NO_BN = bool(args.no_bn)
    USE_PINNED = bool(args.pinned)

    # 무시 옵션 경고
    if args.no_cudnn or args.channels_last or args.no_fp16 or args.no_depthwise or args.no_bn or args.pinned:
        vlog("[warn] 일부 PyTorch 전용 옵션은 ONNXRuntime 경로에서 무시됩니다.")

    # ROI 파싱
    if args.roi_px is not None and args.roi_off is not None:
        set_center_roi(args.roi_px, args.roi_off)
    elif args.roi is not None:
        ROI = tuple(map(int, args.roi))
        ROI_PX = None; ROI_OFF = None
        vlog(f"set ROI_XYWH={ROI}")

    if args.env_check: diag_env()

    cap = None
    need_cam = (args.camera is not None) and (args.realtime or args.manual or args.oneshot or args.roi_snap or args.fps_test>0 or args.e2e_warmup>0)
    if need_cam:
        vlog("open camera first...")
        pixfmt = None if args.cam_pixfmt=="AUTO" else args.cam_pixfmt
        cap = open_camera(args.camera, backend=args.cam_backend, w=args.cam_w, h=args.cam_h,
                          fps=args.cam_fps, pixfmt=pixfmt or "YUYV")
        if not cap or not cap.isOpened():
            print(f"카메라 열기 실패: {args.camera}", file=sys.stderr); return
        ok, _ = cap.read()
        if not ok:
            print("카메라 첫 프레임 획득 실패", file=sys.stderr); cap.release(); return
        vlog("camera ready")
        if args.roi_snap:
            roi_snapshot_from_cap(cap, out_path=args.roi_snap)
            if not (args.oneshot or args.manual or args.realtime or args.images or args.fps_test>0 or args.e2e_warmup>0):
                cap.release(); return

    vlog("build embedder...")
    emb = build_embedder_from_flags()
    vlog("embedder ready")

    # 카메라 프레임으로 파이프라인 전체 워밍업
    if need_cam and args.e2e_warmup>0:
        e2e_warmup(emb, cap, n=args.e2e_warmup, pregrab=args.pregrab)

    if args.images:
        paths = list_images(args.images)
        t0 = time.perf_counter(); pairs=[]; batch=[]; keep=[]
        for p in paths:
            img = imread_bgr(p)
            if img is None: print(f"로드 실패: {p}", file=sys.stderr); pairs.append((p,None)); continue
            try:
                X = preprocess_bgr(img, size=INPUT_SIZE)
                batch.append(X); keep.append(p)
                if len(batch)>=16:
                    feats = emb.embed_batch_np(batch); pairs += list(zip(keep, feats)); batch=[]; keep=[]
            except Exception as e:
                print(f"전처리 실패 - {p} | {e}", file=sys.stderr); pairs.append((p,None))
        if batch:
            feats = emb.embed_batch_np(batch); pairs += list(zip(keep, feats))
        print(f"임베딩 완료: {sum(1 for _,e in pairs if e is not None)} / {len(pairs)} | {(time.perf_counter()-t0):.3f}s")
        save_matrix_and_index(pairs, args.outdir)

    if need_cam:
        if args.fps_test > 0:
            diag_fps_headless(emb, cap, frames=args.fps_test)
        if args.oneshot:
            oneshot(emb, cap, print_k=args.print_k, save_npy=args.save_npy, save_img=args.save_img)
        elif args.manual:
            manual_demo_headless(emb, cap, print_k=args.print_k, save_npy=args.save_npy, save_img=args.save_img)
        elif args.realtime:
            realtime_demo_headless(emb, cap)
        cap.release()

    if not (args.images or need_cam or args.env_check):
        print("동작이 지정되지 않았습니다. --camera 와 --manual/--oneshot/--realtime/--roi_snap 또는 --images 등을 지정하세요.")

if __name__ == "__main__":
    main()
