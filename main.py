# main.py
# - 시작 시 OpenCV/torch/RealSense/임베더 e2e 워밍업 수행
# - datamatrix.py의 DMatrixWatcher(camera=/dev/video2)로 코드 인식
# - 감지되면 depth로 L/W/H(mm) 측정 → img2emb 임베딩(128차원) → [L,W,H]+emb(128)=131차원 전체 print

import os
import time
import numpy as np
import cv2
import pyrealsense2 as rs
import torch

from datamatrix import DMatrixWatcher   # datamatrix.py
from depth import DepthEstimator
import img2emb as embmod               # img2emb.py 모듈

# numpy 출력 옵션: 절대 생략 없이 전부 출력
np.set_printoptions(suppress=True, linewidth=100000, threshold=np.inf, precision=4)

# ===== 설정 =====
EMB_CAM_DEV   = "/dev/video2"   # 인덱스 2
EMB_CAM_PIX   = "YUYV"          # "YUYV" | "MJPG"
EMB_CAM_W     = 848
EMB_CAM_H     = 480
EMB_CAM_FPS   = 6

EMB_INPUT_SIZE = 128
EMB_OUT_DIM    = 128            # 임베딩 128차원
EMB_WIDTH      = 0.35
EMB_USE_FP16   = False
EMB_USE_DW     = False
EMB_USE_BN     = False
EMB_PINNED     = False

# e2e 워밍업(시작 즉시 1회 수행)
E2E_WARMUP_FRAMES = 60
E2E_PREGRAB       = 8

def maybe_run_jetson_perf():
    for cmd in ["sudo nvpmodel -m 0", "sudo jetson_clocks"]:
        os.system(cmd + " >/dev/null 2>&1")

def warmup_opencv_kernels():
    print("[warmup] OpenCV start")
    dummy = (np.random.rand(256, 256).astype(np.float32) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    _ = cv2.morphologyEx(dummy, cv2.MORPH_OPEN, k, iterations=1)
    _ = cv2.Canny(dummy, 40, 120)
    ys, xs = np.where(dummy > 128)
    if xs.size > 10:
        pts = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
        rect = cv2.minAreaRect(pts)
        _ = cv2.boxPoints(rect)
    print("[warmup] OpenCV done")

def warmup_torch_cuda():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[warmup] Torch start (device={dev})")
    try:
        x = torch.randn(1, 3, 128, 128, device=dev)
        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 64)
        ).to(dev).eval()
        with torch.inference_mode():
            for _ in range(3):
                _ = m(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
        print("[warmup] Torch done")
    except Exception as e:
        print(f"[warmup] Torch error: {e}")

def warmup_realsense_and_calib(depth: DepthEstimator, rs_warmup_s=1.5, calib_s=3.0):
    print("[warmup] RealSense pipeline start")
    depth.start()
    t0 = time.time()
    frames = depth.warmup(seconds=rs_warmup_s)
    print(f"[warmup] RealSense done (frames={frames}, {time.time()-t0:.2f}s)")
    print("[calib] floor plane estimation...")
    t1 = time.time()
    ok = depth.calibrate(max_seconds=calib_s)
    if ok:
        print(f"[calib] success ({time.time()-t1:.2f}s)")
    else:
        print("[calib] failed (timeout) — 바닥만 보이게 하고 다시 시도 권장")
    return ok

def build_embedder_only():
    print("[warmup] img2emb: build embedder (no camera)")
    emb = embmod.TorchTinyMNetEmbedder(
        out_dim=EMB_OUT_DIM,
        width=EMB_WIDTH,
        size=EMB_INPUT_SIZE,
        fp16=EMB_USE_FP16,
        weights_path=None,
        channels_last=False,
        cudnn_benchmark=False,
        warmup_steps=3,          # 모델/GPU 워밍업
        use_depthwise=EMB_USE_DW,
        use_bn=EMB_USE_BN,
        pinned=EMB_PINNED
    )
    print("[warmup] img2emb: embedder ready")
    return emb

def open_embed_camera():
    cap = embmod.open_camera(
        EMB_CAM_DEV,
        backend="auto",
        w=EMB_CAM_W, h=EMB_CAM_H, fps=EMB_CAM_FPS,
        pixfmt=EMB_CAM_PIX
    )
    if not cap or not cap.isOpened():
        raise RuntimeError(f"임베딩 카메라 열기 실패: {EMB_CAM_DEV}")
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("임베딩 카메라 첫 프레임 실패")
    return cap

def e2e_warmup_now(emb):
    print(f"[warmup] img2emb: e2e warmup {E2E_WARMUP_FRAMES} frames (pregrab={E2E_PREGRAB})")
    cap = open_embed_camera()
    try:
        embmod.e2e_warmup(emb, cap, n=E2E_WARMUP_FRAMES, pregrab=E2E_PREGRAB)
    finally:
        try: cap.release()
        except Exception: pass
    print("[warmup] img2emb: e2e done & camera released")

def embed_one_frame(emb, pregrab=3):
    cap = open_embed_camera()
    try:
        for _ in range(max(0, pregrab)):
            cap.grab()
        ok, bgr = cap.read()
        if not ok:
            return None
        v = emb.embed_bgr(bgr)               # numpy 128-dim, L2-normalized
        torch.cuda.synchronize()
        return v.astype(np.float32)
    finally:
        try: cap.release()
        except Exception: pass

def main():
    t_all = time.time()
    print("[init] imports ready")
    maybe_run_jetson_perf()
    warmup_opencv_kernels()
    warmup_torch_cuda()

    # 깊이 파이프라인 웜업+캘리브
    depth = DepthEstimator()
    ok_calib = warmup_realsense_and_calib(depth, rs_warmup_s=1.5, calib_s=3.0)
    if not ok_calib:
        print("[fatal] depth calib 실패. 바닥만 보이게 하고 재실행하세요.")
        return

    # 임베더 모델 준비 후 즉시 e2e 워밍업
    emb = build_embedder_only()
    e2e_warmup_now(emb)

    print(f"[ready] total init {time.time()-t_all:.2f}s")

    # DataMatrix 감시 시작(카메라=/dev/video2)
    watcher = DMatrixWatcher(camera=EMB_CAM_DEV, prefer_res=(1920,1080), prefer_fps=6)
    watcher.start()
    print("[loop] datamatrix 감시 중. 인식되면 L/W/H 측정 후 임베딩 추출 및 전체 벡터 print. Ctrl+C 종료")

    try:
        while True:
            event = watcher.get_detection(timeout=0.5)   # (ts, payloads) or None
            if event is None:
                continue

            ts, payloads = event
            label = payloads[0] if payloads else ""
            print(f"[event] ts={ts:.3f}, payloads={payloads}")

            if hasattr(watcher, "pause"):
                watcher.pause()   # 루프에서 카메라 해제됨

            # 0.7초 윈도우 측정
            print("[measure] 0.7s window (L/W/H)")
            Lm, Wm, Hm = depth.measure_dimensions(duration_s=0.7)
            if any(v is None for v in (Lm, Wm, Hm)):
                print("[measure] None (no object/unstable)")
                if hasattr(watcher, "resume"):
                    watcher.resume()
                continue
            print(f"[measure] L={Lm:.1f} mm, W={Wm:.1f} mm, H={Hm:.1f} mm, label='{label}'")

            # 임베딩(128차원) 추출
            vec = embed_one_frame(emb, pregrab=3)
            if vec is None:
                print("[embed] 실패")
                if hasattr(watcher, "resume"):
                    watcher.resume()
                continue

            # [L,W,H] + emb(128) = 131차원 결합 벡터 생성 및 전체 출력
            meta = np.array([Lm, Wm, Hm], dtype=np.float32)
            full_vec = np.concatenate([meta, vec], axis=0)  # shape (131,)
            if vec.shape[0] != EMB_OUT_DIM:
                print(f"[warn] 임베딩 차원 불일치: got {vec.shape[0]}, expect {EMB_OUT_DIM}")

            print(f"[record] label={label}")
            print(f"[record] vector_dim={full_vec.shape[0]}")
            print(full_vec)  # numpy 출력 옵션으로 모든 값이 한 번에 표시됨

            if hasattr(watcher, "resume"):
                watcher.resume()

    except KeyboardInterrupt:
        print("[exit] keyboard interrupt")
    finally:
        try: watcher.stop()
        except Exception: pass
        try: depth.stop()
        except Exception: pass
        print("[cleanup] stopped")

if __name__ == "__main__":
    main()
