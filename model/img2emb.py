# -3뷰 센터와 좌우 거울 임베딩을 concat 256x3으로 early fusion 활성화
# - 실시간 모드에서 거울 임베딩 캐시를 두고 N 프레임마다만 갱신

import os, sys, time, glob, csv, argparse, select
from pathlib import Path
import numpy as np
import cv2

# ONNXRuntime
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

# 전역 기본값
PRETRAINED = True
PRETRAINED_MODE = "onnx"              
PRETRAINED_PATH = "mobilenetv3_small_emb.onnx"

EMBED_DIM = 256
INPUT_SIZE = 224
WIDTH_SCALE = 1.0
USE_FP16 = False                 
CHANNELS_LAST = False               
FRAME_SKIP_N = 1
CUDNN_BENCHMARK = False          
WARMUP_STEPS = 0                    
NO_DEPTHWISE = False               
NO_BN = False                       
USE_PINNED = False          

# ROI 기본값  단일 ROI 경로용
ROI = (280, 96, 288, 288)
ROI_PX = None
ROI_OFF = None

# 3뷰 기본 ROI
# tuple  w h dx dy hflip
CENTER_DEFAULT = (540, 540, +103, -260, 0)
LEFT_DEFAULT   = (240, 460, -380, -170, 1)
RIGHT_DEFAULT  = (270, 440, +630, -170, 1)

# 시간/로그
T0 = time.perf_counter()
VERBOSE = False
TIME_LOG = True
PROFILE = False
GARGS = None  

# 로깅 유틸  verbose일 때 경과시간 포함 출력
def vlog(msg: str):
    if VERBOSE:
        if TIME_LOG: print(f"[{time.perf_counter()-T0:7.3f}s] {msg}", flush=True)
        else:        print(msg, flush=True)

# 전처리
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# L2 정규화 수행
def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    return x / (n + eps)

# 안전한 사각형 크롭 
def _safe_crop_xywh(img, x, y, w, h):
    H, W = img.shape[:2]
    x2, y2 = x + w, y + h
    x  = max(0, x);  y  = max(0, y)
    x2 = min(W, x2); y2 = min(H, y2)
    if x >= x2 or y >= y2:
        raise ValueError(f"ROI out of bounds: img=({W}x{H}), roi={(x,y,w,h)}")
    return img[y:y2, x:x2]

# 중심과 오프셋 기준으로 ROI 좌표 계산
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

# ROI를 shrink 비율만큼 안쪽으로 줄이기
def _shrink_xywh(x, y, w, h, shrink=0.0):
    if shrink <= 0:
        return x, y, w, h
    sx = int(w * shrink / 2.0)
    sy = int(h * shrink / 2.0)
    x2, y2 = x + w - sx, y + h - sy
    x, y = x + sx, y + sy
    w, h = max(1, x2 - x), max(1, y2 - y)
    return x, y, w, h

# 절대 ROI가 주어지면 해당 영역 크롭
def safe_crop(img, roi):
    if roi is None:
        return img
    x, y, w, h = map(int, roi)
    return _safe_crop_xywh(img, x, y, w, h)

# RGB 전처리
def _preprocess_rgb_chw(img_rgb, size=INPUT_SIZE):
    if (img_rgb.shape[1], img_rgb.shape[0]) != (size, size):
        img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0).astype(np.float32)

# BGR 입력을 ROI 처리 후 RGB 전처리 수행
def preprocess_bgr(img_bgr, size=INPUT_SIZE):
    if ROI_PX is not None and ROI_OFF is not None:
        x, y, w, h = _compute_center_roi_xy(img_bgr, ROI_PX, ROI_OFF)
        img = _safe_crop_xywh(img_bgr, x, y, w, h)
    elif ROI is not None:
        img = safe_crop(img_bgr, ROI)
    else:
        img = img_bgr
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _preprocess_rgb_chw(img, size=size)

# 중심기준 ROI와 shrink hflip 옵션으로 전처리 수행
def preprocess_bgr_with_center_roi(img_bgr, px_wh, off_xy, hflip=False, size=INPUT_SIZE, shrink=0.0):
    x, y, w, h = _compute_center_roi_xy(img_bgr, px_wh, off_xy)
    x, y, w, h = _shrink_xywh(x, y, w, h, shrink=shrink)
    crop = _safe_crop_xywh(img_bgr, x, y, w, h)
    if hflip:
        crop = cv2.flip(crop, 1)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return _preprocess_rgb_chw(rgb, size=size)

# 파일 경로에서 BGR 이미지 읽기
def imread_bgr(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None: img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img

# 글롭 패턴들에서 이미지 경로 수집
def list_images(patterns):
    paths = []
    for pat in patterns: paths.extend(glob.glob(pat))
    return sorted({p for p in paths if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))})

# 임베딩 행렬과 인덱스 CSV 저장
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

# ONNXRuntime 세션을 생성하고 단일 배치, 또는 배치 입력의 임베딩을 추출하는 래퍼 클래스
class OnnxEmbedder:
    def __init__(self, onnx_path, size=INPUT_SIZE, out_dim=EMBED_DIM):
        if not onnx_path or not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 파일이 없습니다: {onnx_path}")
        self.size = int(size)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1 
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )
        self.inp = self.session.get_inputs()[0].name
        self.out = self.session.get_outputs()[0].name
        self.out_dim = int(out_dim)

    # 내부 추론 실행
    def _run(self, X: np.ndarray) -> np.ndarray:
        y = self.session.run([self.out], {self.inp: X})[0]
        return y

    # 단일 이미지 BGR에서 전처리 후 추론과 L2 정규화 수행
    def embed_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        X = preprocess_bgr(img_bgr, size=self.size)
        t1 = time.perf_counter()
        f = self._run(X)
        t2 = time.perf_counter()
        v = l2_normalize(f[0])
        t3 = time.perf_counter()

        if PROFILE:
            print("[profile]",
                  f"preproc_ms={(t1-t0)*1000:.1f}",
                  f"forward_ms={(t2-t1)*1000:.1f}",
                  f"norm_ms={(t3-t2)*1000:.1f}",
                  f"total_ms={(t3-t0)*1000:.1f}",
                  flush=True)
        return v.astype(np.float32)

    # 여러 배치 입력을 한 번에 추론하고 L2 정규화 수행
    def embed_batch_np(self, batch_np) -> np.ndarray:
        X = np.concatenate(batch_np, 0)
        f = self._run(X)
        return l2_normalize(f).astype(np.float32)

# 중심기준 ROI 전역 설정
def set_center_roi(px_wh, offset_xy):
    global ROI_PX, ROI_OFF, ROI
    ROI_PX = tuple(map(int, px_wh)) if px_wh is not None else None
    ROI_OFF = tuple(map(int, offset_xy)) if offset_xy is not None else None
    if ROI_PX is not None and ROI_OFF is not None:
        ROI = None
    vlog(f"set_center_roi px={ROI_PX} off={ROI_OFF}")

# 플래그 기반 임베더 생성
def build_embedder_from_flags():
    if not PRETRAINED:
        raise RuntimeError("--pretrained=1 과 --pretrained_path=<onnx> 를 지정하세요.")
    return OnnxEmbedder(PRETRAINED_PATH, size=INPUT_SIZE, out_dim=EMBED_DIM)

# 인자에서 3개 뷰 ROI와 옵션을 구성
def build_3view_rois_from_args(args):
    c_w, c_h, c_dx, c_dy, c_hf = CENTER_DEFAULT
    l_w, l_h, l_dx, l_dy, l_hf = LEFT_DEFAULT
    r_w, r_h, r_dx, r_dy, r_hf = RIGHT_DEFAULT

    if args.center_px is not None: c_w, c_h = map(int, args.center_px)
    if args.center_off is not None: c_dx, c_dy = map(int, args.center_off)
    if args.left_px is not None:   l_w, l_h = map(int, args.left_px)
    if args.left_off is not None:  l_dx, l_dy = map(int, args.left_off)
    if args.right_px is not None:  r_w, r_h = map(int, args.right_px)
    if args.right_off is not None: r_dx, r_dy = map(int, args.right_off)
    if args.center_hflip is not None: c_hf = int(args.center_hflip)
    if args.left_hflip is not None:   l_hf = int(args.left_hflip)
    if args.right_hflip is not None:  r_hf = int(args.right_hflip)

    sh_center = float(args.center_shrink or 0.0)
    sh_mirror = float(args.mirror_shrink or 0.08)

    return [
        (c_w, c_h, c_dx, c_dy, c_hf, sh_center),
        (l_w, l_h, l_dx, l_dy, l_hf, sh_mirror),
        (r_w, r_h, r_dx, r_dy, r_hf, sh_mirror),
    ]

# 세 뷰를 모두 신규 추출해 L2 정규화 후 블록 스케일로 결합
def embed_3view_concat(emb, img_bgr, views, block_scale=True):
    batch = []
    for (w,h,dx,dy,hf,sh) in views:
        X = preprocess_bgr_with_center_roi(img_bgr, (w,h), (dx,dy), hflip=bool(hf), size=emb.size, shrink=float(sh))
        batch.append(X)
    feats = emb.embed_batch_np(batch)  # [3 D]
    if block_scale:
        feats = feats / np.sqrt(len(views))
    return feats.reshape(-1).astype(np.float32)

# 좌우 거울 임베딩을 캐시에 저장해 주기적으로만 갱신
def embed_concat_cached(emb, img_bgr, views, cache: "MirrorCache", frame_idx: int, mirror_period: int = 3, block_scale=True):
    (cw,ch,cdx,cdy,chf,csh), (lw,lh,ldx,ldy,lhf,lsh), (rw,rh,rdx,rdy,rhf,rsh) = views
    Xc = preprocess_bgr_with_center_roi(img_bgr, (cw,ch), (cdx,cdy), hflip=bool(chf), size=emb.size, shrink=float(csh))
    need_refresh = (cache.left is None or cache.right is None or (frame_idx % max(1, mirror_period) == 0))
    if need_refresh:
        Xl = preprocess_bgr_with_center_roi(img_bgr, (lw,lh), (ldx,ldy), hflip=bool(lhf), size=emb.size, shrink=float(lsh))
        Xr = preprocess_bgr_with_center_roi(img_bgr, (rw,rh), (rdx,rdy), hflip=bool(rhf), size=emb.size, shrink=float(rsh))
        feats = emb.embed_batch_np([Xl, Xr])
        cache.left, cache.right = feats[0], feats[1]
        cache.frame_idx = frame_idx
    fc = emb.embed_batch_np([Xc])[0]
    stack = np.vstack([fc, cache.left, cache.right])
    if block_scale:
        stack = stack / np.sqrt(3.0)
    return stack.reshape(-1).astype(np.float32)

# 캐시 구조체
class MirrorCache:
    def __init__(self):
        self.left = None
        self.right = None
        self.frame_idx = -1

# GStreamer 파이프라인 문자열 구성
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

# 카메라 열기 (V4L2 시도, 실패 시 GStreamer로 대체)
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

# 표준입력에서 논블록으로 한 줄 읽기
def stdin_readline_nonblock(timeout_sec=0.1):
    r,_,_ = select.select([sys.stdin], [], [], timeout_sec)
    if r: return sys.stdin.readline().strip()
    return None

# 임베딩 요약 출력
def print_embed(v, ms=None, print_k=8):
    l2 = float(np.linalg.norm(v))
    head = np.round(v[:max(0,print_k)], 4) if print_k>0 else np.array([])
    if ms is None:
        print(f"shape={v.shape}, L2={l2:.6f}, first{print_k}={head}", flush=True)
    else:
        print(f"shape={v.shape}, L2={l2:.6f}, time_ms={ms:.1f}, first{print_k}={head}", flush=True)

# 현재 카메라 프레임에서 ROI 사각형들을 그려 저장
def roi_snapshot_from_cap(cap, out_path="ROI_snapshot.jpg", annotate=True):
    if not cap or not cap.isOpened():
        print("[roi_snap] 카메라 열기 실패", file=sys.stderr); return False
    ok, img = cap.read()
    if not ok:
        print("[roi_snap] 프레임 획득 실패", file=sys.stderr); return False
    draw = img.copy()
    info_list = []
    if GARGS and getattr(GARGS, "concat3", False):
        views = build_3view_rois_from_args(GARGS)
        colors = [(0,255,0),(255,0,0),(0,0,255)]
        names  = ["CENTER","LEFT","RIGHT"]
        for i,(w,h,dx,dy,hf,sh) in enumerate(views):
            x, y, _, _ = _compute_center_roi_xy(img, (w,h), (dx,dy))
            x, y, ww, hh = _shrink_xywh(x, y, w, h, shrink=sh)
            cv2.rectangle(draw, (x,y), (x+ww,y+hh), colors[i%3], 2)
            info_list.append(f"{names[i]} px=({w},{h}) off=({dx},{dy}) flip={hf} shrink={sh}")
        info = " | ".join(info_list)
    else:
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

# 환경 진단 정보 출력
def diag_env():
    print("[env]", "onnxruntime:", ort.__version__)
    try:
        print("[env]", "providers:", ort.get_available_providers())
    except Exception:
        pass
    print("[env]", "cv2:", cv2.__version__)
    print("[env]", "ROI_XYWH:", ROI, "ROI_CENTER(px/off):", ROI_PX, ROI_OFF)

# 헤드리스 FPS 측정
def diag_fps_headless(emb, cap, frames=60):
    if not cap or not cap.isOpened():
        print("[fps] 카메라 열기 실패", file=sys.stderr); return
    t0 = time.perf_counter(); n = 0
    cache = MirrorCache()
    views = build_3view_rois_from_args(GARGS) if (GARGS and getattr(GARGS, "concat3", False)) else None
    while n < frames:
        ok, bgr = cap.read()
        if not ok: break
        if FRAME_SKIP_N <= 1 or (n % FRAME_SKIP_N == 0):
            if views is None:
                _ = emb.embed_bgr(bgr)
            else:
                if getattr(GARGS, "use_mirror_cache", True):
                    _ = embed_concat_cached(emb, bgr, views, cache, n, mirror_period=GARGS.mirror_period, block_scale=True)
                else:
                    _ = embed_3view_concat(emb, bgr, views, block_scale=True)
        n += 1
    fps = n / (time.perf_counter() - t0 + 1e-6)
    mode = "concat3" if views is not None else "single"
    print(f"[fps] frames={n}, mode={mode}, avg_fps={fps:.2f} (FRAME_SKIP_N={FRAME_SKIP_N})", flush=True)


# 카메라 기반 엔드투엔드 워밍업 실행
def e2e_warmup(emb, cap, n=30, pregrab=5):
    if not cap or not cap.isOpened() or n <= 0: return
    for _ in range(max(0, pregrab)):
        cap.grab()
    t0 = time.perf_counter(); cnt = 0
    cache = MirrorCache()
    views = build_3view_rois_from_args(GARGS) if (GARGS and getattr(GARGS, "concat3", False)) else None
    while cnt < n:
        ok, bgr = cap.read()
        if not ok: break
        if views is None:
            _ = emb.embed_bgr(bgr)
        else:
            if getattr(GARGS, "use_mirror_cache", True):
                _ = embed_concat_cached(emb, bgr, views, cache, cnt, mirror_period=GARGS.mirror_period, block_scale=True)
            else:
                _ = embed_3view_concat(emb, bgr, views, block_scale=True)
        cnt += 1
    dt = (time.perf_counter()-t0)/max(1,cnt)
    print(f"[warmup] e2e frames={cnt}, avg_ms={dt*1000:.1f}", flush=True)

# 수동 모드
def manual_demo_headless(emb, cap, args, print_k=8, save_npy=None, save_img=None):
    if not cap or not cap.isOpened():
        print("카메라/비디오 열기 실패", file=sys.stderr); return
    print("headless manual: 's'+Enter=infer, 'q'+Enter=quit", flush=True)
    if save_npy: Path(save_npy).mkdir(parents=True, exist_ok=True)
    if save_img: Path(save_img).mkdir(parents=True, exist_ok=True)
    views = build_3view_rois_from_args(args) if args.concat3 else None
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
            if views is None:
                v = emb.embed_bgr(bgr)
            else:
                v = embed_3view_concat(emb, bgr, views, block_scale=True)
            ms = (time.perf_counter()-t0)*1000.0
            print_embed(v, ms=ms, print_k=print_k)
            if save_npy:
                ts = time.strftime("%Y%m%d_%H%M%S")
                np.save(str(Path(save_npy) / f"emb_{ts}.npy"), v.astype(np.float32))
            if save_img:
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(Path(save_img) / f"img_{ts}.jpg"), bgr)

# 단발 추론
def oneshot(emb, cap, args, print_k=8, save_npy=None, save_img=None):
    if not cap or not cap.isOpened():
        print("카메라/비디오 열기 실패", file=sys.stderr); return
    ok, bgr = cap.read()
    if not ok:
        print("프레임 획득 실패", file=sys.stderr); return
    t0 = time.perf_counter()
    if args.concat3:
        views = build_3view_rois_from_args(args)
        v = embed_3view_concat(emb, bgr, views, block_scale=True)
    else:
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

# 실시간 헤드리스 루프
def realtime_demo_headless(emb, cap, args, print_every=30):
    if not cap or not cap.isOpened():
        print("카메라/비디오 열기 실패", file=sys.stderr); return
    t0 = time.perf_counter(); n=0
    cache = MirrorCache()
    views = build_3view_rois_from_args(args) if args.concat3 else None
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if FRAME_SKIP_N<=1 or (n%FRAME_SKIP_N==0):
            if views is None:
                _ = emb.embed_bgr(bgr)
            else:
                if args.use_mirror_cache:
                    _ = embed_concat_cached(emb, bgr, views, cache, n, mirror_period=args.mirror_period, block_scale=True)
                else:
                    _ = embed_3view_concat(emb, bgr, views, block_scale=True)
        n+=1
        if n % print_every == 0:
            fps = n / (time.perf_counter()-t0+1e-6)
            mode = "concat3" if views is not None else "single"
            print(f"fps≈{fps:.2f} ({mode})", flush=True)

# 엔트리 포인트 
def main():
    global PRETRAINED, PRETRAINED_MODE, PRETRAINED_PATH
    global INPUT_SIZE, EMBED_DIM, WIDTH_SCALE, USE_FP16, CHANNELS_LAST, FRAME_SKIP_N
    global VERBOSE, CUDNN_BENCHMARK, WARMUP_STEPS, TIME_LOG, PROFILE
    global NO_DEPTHWISE, NO_BN, USE_PINNED
    global ROI, ROI_PX, ROI_OFF, GARGS

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
    # ROI 옵션  단일 ROI 경로
    ap.add_argument("--roi", type=int, nargs=4, default=None, help="절대좌표 ROI: x y w h")
    ap.add_argument("--roi_px", type=int, nargs=2, default=None, help="중심기준 ROI 폭높이: w h")
    ap.add_argument("--roi_off", type=int, nargs=2, default=None, help="중심기준 ROI 오프셋: dx dy")
    # 3뷰 concat 옵션
    ap.add_argument("--concat3", action="store_true", help="3뷰 센터와 좌우 거울 임베딩을 concat 768D로 출력")
    ap.add_argument("--mirror_period", type=int, default=3, help="realtime에서 거울 임베딩 갱신 주기 프레임")
    ap.add_argument("--use_mirror_cache", action="store_true", help="realtime에서 거울 캐시 사용")
    ap.add_argument("--no_mirror_cache", action="store_true", help="realtime에서 거울 캐시 비활성화")
    ap.add_argument("--mirror_shrink", type=float, default=0.08, help="거울뷰 ROI shrink 비율 0에서 0.3 권장")
    ap.add_argument("--center_shrink", type=float, default=0.00, help="센터뷰 ROI shrink 비율")
    # 각 뷰 ROI 오버라이드
    ap.add_argument("--center_px", type=int, nargs=2, default=None)
    ap.add_argument("--center_off", type=int, nargs=2, default=None)
    ap.add_argument("--left_px", type=int, nargs=2, default=None)
    ap.add_argument("--left_off", type=int, nargs=2, default=None)
    ap.add_argument("--right_px", type=int, nargs=2, default=None)
    ap.add_argument("--right_off", type=int, nargs=2, default=None)
    ap.add_argument("--center_hflip", type=int, default=None)
    ap.add_argument("--left_hflip", type=int, default=None)
    ap.add_argument("--right_hflip", type=int, default=None)
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
    GARGS = args

    if args.no_mirror_cache:
        args.use_mirror_cache = False
    elif args.use_mirror_cache:
        args.use_mirror_cache = True
    else:
        args.use_mirror_cache = True if args.concat3 else False

    # 전역에 반영
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

    # 무시 옵션 경고 안내
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
        t0 = time.perf_counter(); pairs=[]
        for p in paths:
            img = imread_bgr(p)
            if img is None:
                print(f"로드 실패: {p}", file=sys.stderr); pairs.append((p,None)); continue
            try:
                if args.concat3:
                    views = build_3view_rois_from_args(args)
                    v = embed_3view_concat(emb, img, views, block_scale=True)
                else:
                    v = emb.embed_bgr(img)
                pairs.append((p, v))
            except Exception as e:
                print(f"전처리/임베딩 실패 - {p} | {e}", file=sys.stderr); pairs.append((p,None))
        print(f"임베딩 완료: {sum(1 for _,e in pairs if e is not None)} / {len(pairs)} | {(time.perf_counter()-t0):.3f}s")
        save_matrix_and_index(pairs, args.outdir)

    if need_cam:
        if args.fps_test > 0:
            diag_fps_headless(emb, cap, frames=args.fps_test)
        if args.oneshot:
            oneshot(emb, cap, args, print_k=args.print_k, save_npy=args.save_npy, save_img=args.save_img)
        elif args.manual:
            manual_demo_headless(emb, cap, args, print_k=args.print_k, save_npy=args.save_npy, save_img=args.save_img)
        elif args.realtime:
            realtime_demo_headless(emb, cap, args)
        cap.release()

    if not (args.images or need_cam or args.env_check):
        print("동작이 지정되지 않았습니다. --camera 와 --manual/--oneshot/--realtime/--roi_snap 또는 --images 등을 지정하세요.")

if __name__ == "__main__":
    main()
