# img2emb.py
# Jetson Nano: OpenCV CPU + PyTorch CUDA
# 최적화 포인트:
#  - torch.inference_mode()
#  - GPU 입력 버퍼 사전할당 + pinned CPU 메모리 옵션(--pinned)
#  - cuDNN 비활성 옵션(--no_cudnn)
#  - depthwise 비활성 옵션(--no_depthwise)
#  - BatchNorm 제거 옵션(--no_bn)
#  - 내부 구간 프로파일(--profile)
#  - ROI 기본값=(280,96,288,288)
#  - 엔드투엔드 워밍업(--e2e_warmup), 사전 grab(--pregrab)
#  - OpenCV 스레드 초기화 비용 축소(cv2.setNumThreads(1))

import os, sys, time, glob, csv, argparse, select
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn

# OpenCV 스레드 풀 초기화 고정
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# ========= 전역 기본값 =========
PRETRAINED = False
PRETRAINED_MODE = "jit"                 # "jit" or "pth"
PRETRAINED_PATH = "mobilenetv3_small_emb.ts"

EMBED_DIM = 256
INPUT_SIZE = 224
WIDTH_SCALE = 1.0
USE_FP16 = True
CHANNELS_LAST = False
FRAME_SKIP_N = 1
CUDNN_BENCHMARK = False
WARMUP_STEPS = 0
NO_DEPTHWISE = False
NO_BN = False
USE_PINNED = False

# ROI 기본값
ROI = (280, 96, 288, 288)

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

def safe_crop(img, roi):
    if roi is None: return img
    H, W = img.shape[:2]
    x, y, w, h = map(int, roi)
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)
    x2, y2 = min(W, x2), min(H, y2)
    if x >= x2 or y >= y2:
        raise ValueError(f"ROI out of bounds: img=({W}x{H}), roi={roi}")
    return img[y:y2, x:x2]

def preprocess_bgr(img_bgr, size=INPUT_SIZE):
    img = safe_crop(img_bgr, ROI)
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

# ========= 모델 =========
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, g=1, act=True, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, k//2, groups=g, bias=(not use_bn))
        self.bn   = nn.BatchNorm2d(out_ch) if use_bn else None
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None: x = self.bn(x)
        x = self.act(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand, use_depthwise=True, use_bn=True):
        super().__init__()
        hidden = int(round(in_ch * expand))
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand != 1.0:
            layers.append(ConvBNAct(in_ch, hidden, k=1, s=1, use_bn=use_bn))
        g = hidden if use_depthwise else 1
        layers += [
            ConvBNAct(hidden, hidden, k=3, s=stride, g=g, use_bn=use_bn),
            ConvBNAct(hidden, out_ch, k=1, s=1, act=False, use_bn=use_bn)
        ]
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out

class TinyMobileNet(nn.Module):
    def __init__(self, out_dim=EMBED_DIM, width=WIDTH_SCALE, use_depthwise=True, use_bn=True):
        super().__init__()
        def c(ch): return max(8, int(ch * width))
        self.stem = ConvBNAct(3, c(16), k=3, s=2, use_bn=use_bn)
        self.layer1 = InvertedResidual(c(16),  c(24), 2, 2.0, use_depthwise, use_bn)
        self.layer2 = InvertedResidual(c(24),  c(24), 1, 2.0, use_depthwise, use_bn)
        self.layer3 = InvertedResidual(c(24),  c(40), 2, 2.5, use_depthwise, use_bn)
        self.layer4 = InvertedResidual(c(40),  c(40), 1, 2.5, use_depthwise, use_bn)
        self.layer5 = InvertedResidual(c(40),  c(80), 2, 2.5, use_depthwise, use_bn)
        self.layer6 = InvertedResidual(c(80),  c(80), 1, 2.5, use_depthwise, use_bn)
        self.head   = ConvBNAct(c(80), c(128), k=1, s=1, use_bn=use_bn)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(c(128), out_dim)
        self.out_dim = out_dim
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        nn.init.normal_(self.fc.weight, 0, 0.01); nn.init.zeros_(self.fc.bias)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.layer5(x); x = self.layer6(x)
        x = self.head(x); x = self.pool(x).flatten(1)
        return self.fc(x)

# ========= Embedders =========
class TorchTinyMNetEmbedder:
    def __init__(self, out_dim=EMBED_DIM, width=WIDTH_SCALE, size=INPUT_SIZE,
                 fp16=USE_FP16, weights_path=None, channels_last=CHANNELS_LAST,
                 cudnn_benchmark=CUDNN_BENCHMARK, warmup_steps=WARMUP_STEPS,
                 use_depthwise=(not NO_DEPTHWISE), use_bn=(not NO_BN),
                 pinned=USE_PINNED):
        if not torch.cuda.is_available(): raise RuntimeError("CUDA 필요")
        self.device="cuda"; self.size=int(size); self.fp16=bool(fp16)
        self.channels_last = bool(channels_last)
        self.pinned = bool(pinned)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

        vlog("TinyMobileNet build")
        self.model = TinyMobileNet(out_dim=out_dim, width=width,
                                   use_depthwise=use_depthwise, use_bn=use_bn).to(self.device).eval()
        if weights_path and os.path.exists(weights_path):
            vlog(f"load weights {weights_path}"); t0=time.perf_counter()
            state=torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state: state=state["state_dict"]
            if isinstance(state, dict): state={k.replace("module.",""):v for k,v in state.items()}
            self.model.load_state_dict(state, strict=False)
            vlog(f"weights loaded ({time.perf_counter()-t0:.3f}s)")
        else:
            vlog("no pretrained weights")

        # dtype/메모리 포맷
        self.dtype = torch.float16 if self.fp16 else torch.float32
        if self.fp16: self.model.half(); vlog("half()")
        if self.channels_last:
            try: self.model.to(memory_format=torch.channels_last); vlog("channels_last")
            except Exception: vlog("channels_last skipped")

        # GPU 입력 버퍼 사전할당
        self.x_gpu = torch.empty(1,3,self.size,self.size, device=self.device, dtype=self.dtype)
        if self.channels_last:
            try: self.x_gpu = self.x_gpu.to(memory_format=torch.channels_last)
            except Exception: pass

        # CPU pinned 버퍼 사전할당(+NumPy 뷰) → 매 호출 pin_memory() 비용 제거
        self.x_cpu = None
        self.x_cpu_np = None
        if self.pinned:
            self.x_cpu = torch.empty(1,3,self.size,self.size, dtype=torch.float32, pin_memory=True)
            try:
                self.x_cpu_np = self.x_cpu.numpy()
            except Exception:
                self.x_cpu_np = None

        self.amp = torch.cuda.amp.autocast
        if warmup_steps>0:
            vlog(f"warmup({warmup_steps})...")
            with torch.inference_mode():
                for _ in range(warmup_steps):
                    with self.amp(enabled=self.fp16): _=self.model(self.x_gpu)
                torch.cuda.synchronize()
            vlog("warmup done")

    @torch.inference_mode()
    def embed_bgr(self, img_bgr):
        t0 = time.perf_counter()
        X = preprocess_bgr(img_bgr, size=self.size)                # numpy [1,3,H,W]
        t1 = time.perf_counter()

        # H2D: 사전할당 pinned 버퍼가 있으면 그쪽으로 복사 후 GPU로 비동기 전송
        if self.pinned and (self.x_cpu is not None) and (self.x_cpu_np is not None):
            np.copyto(self.x_cpu_np, X, casting="no")              # CPU memcpy 고정 크기
            t2 = time.perf_counter()
            self.x_gpu.copy_(self.x_cpu.to(self.dtype), non_blocking=True)
        else:
            t_cpu = torch.from_numpy(X)                            # fallback
            if self.pinned: t_cpu = t_cpu.pin_memory()
            t2 = time.perf_counter()
            self.x_gpu.copy_(t_cpu.to(self.dtype), non_blocking=self.pinned)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        with self.amp(enabled=self.fp16):
            f = self.model(self.x_gpu)
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        y = f.squeeze(0).float().cpu().numpy()                     # D2H
        t5 = time.perf_counter()
        v = l2_normalize(y)
        t6 = time.perf_counter()

        if PROFILE:
            print("[profile]",
                  f"preproc_ms={(t1-t0)*1000:.1f}",
                  f"cpu_tensor_ms={(t2-t1)*1000:.1f}",
                  f"H2D_ms={(t3-t2)*1000:.1f}",
                  f"forward_ms={(t4-t3)*1000:.1f}",
                  f"D2H_ms={(t5-t4)*1000:.1f}",
                  f"norm_ms={(t6-t5)*1000:.1f}",
                  f"total_ms={(t6-t0)*1000:.1f}",
                  flush=True)
        return v

    @torch.inference_mode()
    def embed_batch_np(self, batch_np):
        X = np.concatenate(batch_np, 0)
        t_cpu = torch.from_numpy(X)
        if self.pinned: t_cpu = t_cpu.pin_memory()
        x = torch.empty_like(t_cpu, device=self.device, dtype=self.dtype)
        x.copy_(t_cpu.to(self.dtype), non_blocking=self.pinned)
        with self.amp(enabled=self.fp16): f = self.model(x)
        return l2_normalize(f.float().cpu().numpy())

# ========= Embedder factory =========
def build_embedder_from_flags():
    return TorchTinyMNetEmbedder(out_dim=EMBED_DIM, width=WIDTH_SCALE, size=INPUT_SIZE,
                                 fp16=USE_FP16, weights_path=(PRETRAINED_PATH if PRETRAINED else None),
                                 channels_last=CHANNELS_LAST, cudnn_benchmark=CUDNN_BENCHMARK,
                                 warmup_steps=WARMUP_STEPS, use_depthwise=(not NO_DEPTHWISE),
                                 use_bn=(not NO_BN), pinned=USE_PINNED)

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
    if ROI is not None:
        x, y, w, h = map(int, ROI)
        cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if annotate:
        H, W = img.shape[:2]
        cv2.putText(draw, f"{W}x{H} ROI={ROI}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imwrite(out_path, draw)
    print(f"[roi_snap] 저장 완료: {out_path}")
    return True

def diag_env():
    print("[env]", "torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available(): print("[env]", "device:", torch.cuda.get_device_name(0))
    print("[env]", "cudnn.enabled:", torch.backends.cudnn.enabled)
    print("[env]", "cudnn.benchmark:", torch.backends.cudnn.benchmark)
    print("[env]", "cv2:", cv2.__version__)
    print("[env]", "ROI:", ROI)

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
    torch.cuda.synchronize()
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
            torch.cuda.synchronize()
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
    torch.cuda.synchronize()
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

# ========= Embedder factory와 I/O 엔트리 =========
def build_embedder_from_flags():
    return TorchTinyMNetEmbedder(out_dim=EMBED_DIM, width=WIDTH_SCALE, size=INPUT_SIZE,
                                 fp16=USE_FP16, weights_path=(PRETRAINED_PATH if PRETRAINED else None),
                                 channels_last=CHANNELS_LAST, cudnn_benchmark=CUDNN_BENCHMARK,
                                 warmup_steps=WARMUP_STEPS, use_depthwise=(not NO_DEPTHWISE),
                                 use_bn=(not NO_BN), pinned=USE_PINNED)

# ========= CLI =========
def main():
    global PRETRAINED, PRETRAINED_MODE, PRETRAINED_PATH
    global INPUT_SIZE, EMBED_DIM, WIDTH_SCALE, USE_FP16, CHANNELS_LAST, FRAME_SKIP_N
    global VERBOSE, CUDNN_BENCHMARK, WARMUP_STEPS, TIME_LOG, PROFILE
    global NO_DEPTHWISE, NO_BN, USE_PINNED

    ap = argparse.ArgumentParser()
    # 로깅/성능
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--time_log", action="store_true")
    ap.add_argument("--no_time_log", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--cudnn_benchmark", action="store_true")
    ap.add_argument("--no_cudnn", action="store_true", help="cuDNN 비활성화")
    ap.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    # 모델/전역
    ap.add_argument("--pretrained", type=int, default=int(PRETRAINED))
    ap.add_argument("--pretrained_mode", type=str, default=PRETRAINED_MODE)
    ap.add_argument("--pretrained_path", type=str, default=PRETRAINED_PATH)
    ap.add_argument("--size", type=int, default=INPUT_SIZE)
    ap.add_argument("--out_dim", type=int, default=EMBED_DIM)
    ap.add_argument("--width", type=float, default=WIDTH_SCALE)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--channels_last", action="store_true")
    ap.add_argument("--frame_skip", type=int, default=FRAME_SKIP_N)
    ap.add_argument("--no_depthwise", action="store_true", help="depthwise conv 비활성화")
    ap.add_argument("--no_bn", action="store_true", help="BatchNorm 제거")
    ap.add_argument("--pinned", action="store_true", help="CPU pinned 메모리 사용")
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
    WIDTH_SCALE = float(args.width); USE_FP16 = not args.no_fp16
    CHANNELS_LAST = bool(args.channels_last); FRAME_SKIP_N = max(1, int(args.frame_skip))
    NO_DEPTHWISE = bool(args.no_depthwise)
    NO_BN = bool(args.no_bn)
    USE_PINNED = bool(args.pinned)

    # cuDNN on/off
    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
        vlog("cuDNN disabled")
    else:
        torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK

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
