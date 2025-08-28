# core/utils.py
import os, sys, time, select
import numpy as np
import cv2

def ts_wall():
    return time.strftime("%H:%M:%S.%f", time.localtime())[:-3]

def t_now():
    return time.perf_counter()

def ms(dt):
    return f"{dt*1000:.2f} ms"

def stdin_readline_nonblock(timeout_sec=0.05):
    r,_,_ = select.select([sys.stdin], [], [], timeout_sec)
    if r: return sys.stdin.readline().strip()
    return None

def maybe_run_jetson_perf():
    for cmd in ["sudo nvpmodel -m 0", "sudo jetson_clocks"]:
        os.system(cmd + " >/dev/null 2>&1")

def warmup_opencv_kernels():
    print("[warmup] OpenCV start")
    dummy = (np.random.rand(256, 256).astype(np.float32) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    _ = cv2.morphologyEx(dummy, cv2.MORPH_OPEN, k, iterations=1)
    _ = cv2.Canny(dummy, 40, 120)
    print("[warmup] OpenCV done")

def warmup_torch_cuda():
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[warmup] Torch start (device={dev})")
        x = torch.randn(1, 3, 64, 64, device=dev)
        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 16)
        ).to(dev).eval()
        with torch.inference_mode():
            for _ in range(2):
                _ = m(x)
                if dev == "cuda": torch.cuda.synchronize()
        print("[warmup] Torch done")
    except Exception as e:
        print(f"[warmup] Torch error: {e}")

def l2_normalize(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x)
    return x / (n + eps) if n > 0 else x

def same_device(a, b):
    def norm(x):
        if isinstance(x, int): return f"/dev/video{x}"
        if isinstance(x, str) and x.isdigit(): return f"/dev/video{int(x)}"
        return x
    return norm(a) == norm(b)
