# core/utils.py  (Python 3.6 compatible)
from __future__ import print_function
import os
import sys
import time
import select
import subprocess
import numpy as np

try:
    import torch
except Exception:
    torch = None


def ts_wall():
    """로컬시간 HH:MM:SS 문자열."""
    return time.strftime("%H:%M:%S", time.localtime())


def t_now():
    """고해상도 단조 증가 시간(초)."""
    try:
        return time.perf_counter()
    except AttributeError:
        return time.time()


def ms(dt):
    """초 → 'xx.xx ms' 문자열."""
    try:
        return "{:.2f} ms".format(float(dt) * 1000.0)
    except Exception:
        return "n/a"


def stdin_readline_nonblock(timeout=0.05):
    """블로킹하지 않고 한 줄 읽기. 입력 없으면 None."""
    try:
        rlist, _, _ = select.select([sys.stdin], [], [], float(timeout))
        if rlist:
            line = sys.stdin.readline()
            # EOF면 빈 문자열이 들어옴
            return line if line != "" else None
        return None
    except Exception:
        return None


def maybe_run_jetson_perf():
    """Jetson이면 성능고정 도구를 best-effort로 호출 (없으면 무시)."""
    try:
        if os.path.exists("/usr/bin/jetson_clocks"):
            subprocess.Popen(
                ["/usr/bin/jetson_clocks", "--store"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except Exception:
        pass


def warmup_opencv_kernels():
    """OpenCV 커널 간단 워밍업(있으면)."""
    try:
        import cv2
        img = np.zeros((64, 64, 3), np.uint8)
        _ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _ = cv2.GaussianBlur(img, (3, 3), 0)
    except Exception:
        pass


def warmup_torch_cuda():
    """CUDA 컨텍스트 초기화(있으면)."""
    try:
        if (torch is not None) and torch.cuda.is_available():
            x = torch.zeros(1, device="cuda")
            _ = (x + 1.0).sum().item()
            torch.cuda.synchronize()
    except Exception:
        pass


def l2_normalize(x, eps=1e-8):
    """L2 정규화."""
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n < eps:
        return x
    return (x / (n + eps)).astype(np.float32)


def same_device(a, b):
    """
    카메라 디바이스 동일성 비교.
    - 정수 2 ↔ '/dev/video2' 같은 케이스를 동일 취급.
    """
    def _norm(x):
        if isinstance(x, int):
            return "/dev/video{}".format(x)
        if isinstance(x, str):
            return x
        return str(x)
    return _norm(a) == _norm(b)
