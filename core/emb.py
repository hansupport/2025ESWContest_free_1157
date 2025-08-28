# core/emb.py  (Python 3.6 compatible: no future annotations)
import time
from typing import Optional

import cv2
import numpy as np
import torch

import img2emb as embmod
from datamatrix import read_frame_nonblocking as dm_read_frame
from .utils import same_device


def apply_roi_from_cfg(S):
    try:
        roi_px = S.embedding.roi_px
        roi_off = S.embedding.roi_offset
        if roi_px is not None and hasattr(embmod, "set_center_roi"):
            embmod.set_center_roi(roi_px, roi_off)
            print("[img2emb.cfg] ROI(center) px=%s off=%s" % (str(roi_px), str(roi_off)))
            return
        if roi_px is not None:
            w, h = int(roi_px[0]), int(roi_px[1])
            dx, dy = int(roi_off[0]), int(roi_off[1])
            W, H = int(S.embedding.width), int(S.embedding.height)
            cx, cy = W // 2 + dx, H // 2 + dy
            x = max(0, min(W - w, cx - w // 2))
            y = max(0, min(H - h, cy - h // 2))
            embmod.ROI = (x, y, w, h)
            print("[img2emb.cfg] ROI(xywh)=%s (fallback)" % (str((x, y, w, h))))
    except Exception as e:
        print("[img2emb.cfg] ROI 적용 실패:", e)


def build_embedder(S):
    apply_roi_from_cfg(S)
    print("[warmup] img2emb: build embedder")
    emb = embmod.TorchTinyMNetEmbedder(
        out_dim=S.embedding.out_dim,
        width=S.embedding.width_scale,
        size=S.embedding.input_size,
        fp16=S.embedding.fp16,
        weights_path=(str(S.paths.weights) if S.paths.weights else None),
        channels_last=False,
        cudnn_benchmark=False,
        warmup_steps=3,
        use_depthwise=S.embedding.use_depthwise,
        use_bn=S.embedding.use_bn,
        pinned=S.embedding.pinned,
    )
    print("[warmup] img2emb: embedder ready")
    return emb


def open_camera(S):
    cap = embmod.open_camera(
        S.embedding.cam_dev,
        backend="auto",
        w=S.embedding.width,
        h=S.embedding.height,
        fps=S.embedding.fps,
        pixfmt=S.embedding.pixfmt,
    )
    if not cap or not cap.isOpened():
        raise RuntimeError("임베딩 카메라 열기 실패: %s" % str(S.embedding.cam_dev))
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("임베딩 카메라 첫 프레임 실패")
    return cap


def _warmup_cv2_cap(cap, seconds=0.4):
    t0 = time.time()
    n = 0
    while time.time() - t0 < max(0.0, seconds):
        ok, _ = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        n += 1
        time.sleep(0.005)
    return n


def warmup_shared(emb, S, dm_handle, dm_lock, frames: int, pregrab: int):
    if same_device(S.dm.camera, S.embedding.cam_dev) and (dm_handle is not None):
        print("[warmup] img2emb: shared e2e warmup %d frames (pregrab=%d) via DM persistent cam" % (frames, pregrab))
        with dm_lock:
            for _ in range(max(0, pregrab)):
                _ = dm_read_frame(dm_handle)
        t0 = time.time()
        n_ok = 0
        try:
            # torch.inference_mode는 torch 1.10에서 사용 가능
            ctx = torch.inference_mode()
        except AttributeError:
            # PyTorch 구버전 대비
            class _NullCtx(object):
                def __enter__(self): pass
                def __exit__(self, exc_type, exc, tb): pass
            ctx = _NullCtx()

        with ctx:
            for _ in range(max(1, frames)):
                with dm_lock:
                    bgr = dm_read_frame(dm_handle)
                if bgr is None:
                    time.sleep(0.003)
                    continue
                _ = emb.embed_bgr(bgr)
                n_ok += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("[warmup] img2emb: shared e2e done, ok_frames=%d, elapsed=%.2fs" % (n_ok, time.time() - t0))
        return

    print("[warmup] img2emb: separate device warmup path")
    cap = open_camera(S)
    try:
        _ = _warmup_cv2_cap(cap, seconds=0.4)
        embmod.e2e_warmup(emb, cap, n=frames, pregrab=pregrab)
    finally:
        try:
            cap.release()
        except Exception:
            pass
    print("[warmup] img2emb: e2e done")


def embed_one_frame_shared(emb, S, dm_handle, dm_lock, pregrab=3) -> Optional[np.ndarray]:
    if same_device(S.dm.camera, S.embedding.cam_dev) and (dm_handle is not None):
        with dm_lock:
            for _ in range(max(0, pregrab)):
                _ = dm_read_frame(dm_handle)
            bgr = dm_read_frame(dm_handle)
        if bgr is None:
            return None
        v = emb.embed_bgr(bgr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return v.astype(np.float32)

    # fallback (separate device)
    cap = open_camera(S)
    try:
        for _ in range(max(0, pregrab)):
            cap.grab()
        ok, bgr = cap.read()
        if not ok:
            return None
        v = emb.embed_bgr(bgr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return v.astype(np.float32)
    finally:
        try:
            cap.release()
        except Exception:
            pass
