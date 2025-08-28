# core/dm.py
from __future__ import print_function
import time
import threading
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

from .utils import ts_wall, t_now, ms

# datamatrix 유틸 (fast4만 사용)
from datamatrix import (
    open_camera as dm_open_camera,
    read_frame_nonblocking as dm_read_frame,
    crop_roi_center as dm_crop_roi,
    decode_payloads_fast4 as dm_decode_fast4,
)

try:
    from datamatrix import DEFAULT_ROIS as DM_DEFAULT_ROIS
except Exception:
    # Fallback 기본 ROI 3개
    DM_DEFAULT_ROIS = [
        dict(name="ROI1", size=[260, 370], offset=[-380, 100], hflip=True),
        dict(name="ROI2", size=[300, 400], offset=[ 610, 110], hflip=True),
        dict(name="ROI3", size=[480, 340], offset=[ 120,  70], hflip=False),
    ]

# 공유 락 및 퍼시스턴트 핸들
DM_LOCK = threading.Lock()
_PERSIST_HANDLE = None  # ("realsense", (pipeline, rs)) 또는 ("cv2", cap)


def _roi_get(obj, key, default=None):
    """dict와 SimpleNamespace 모두에서 안전하게 속성/키를 가져온다."""
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    except Exception:
        return default


def open_persistent(camera, prefer_res, prefer_fps):
    # type: (Any, Any, Any) -> Optional[tuple]
    """DM 카메라 퍼시스턴트 오픈(+프리웜). 이미 열려 있으면 그대로 반환."""
    global _PERSIST_HANDLE
    if _PERSIST_HANDLE is not None:
        return _PERSIST_HANDLE
    try:
        w = int(prefer_res[0]) if isinstance(prefer_res, (list, tuple)) and len(prefer_res) >= 2 else 1920
        h = int(prefer_res[1]) if isinstance(prefer_res, (list, tuple)) and len(prefer_res) >= 2 else 1080
        fps = int(prefer_fps)
        _PERSIST_HANDLE = dm_open_camera(camera, (w, h), fps)

        # prewarm
        t0 = time.time()
        read = 0
        while time.time() - t0 < 0.2:
            with DM_LOCK:
                f = dm_read_frame(_PERSIST_HANDLE)
            if f is not None:
                read += 1
            time.sleep(0.005)
        print("[dm.persist] opened {} and prewarmed, frames={}".format(camera, read))
    except Exception as e:
        _PERSIST_HANDLE = None
        print("[dm.persist] open failed:", e)
    return _PERSIST_HANDLE


def close_persistent():
    """퍼시스턴트 핸들 닫기."""
    global _PERSIST_HANDLE
    if _PERSIST_HANDLE is None:
        return
    try:
        kind = _PERSIST_HANDLE[0]
        if kind == "realsense":
            pipeline, rs = _PERSIST_HANDLE[1]
            pipeline.stop()
        else:
            cap = _PERSIST_HANDLE[1]
            try:
                cap.release()
            except Exception:
                pass
    except Exception:
        pass
    _PERSIST_HANDLE = None
    print("[dm.persist] closed")


def read_frame(handle):
    """퍼시스턴트 핸들에서 프레임 읽기(논블로킹)."""
    if handle is None:
        return None
    with DM_LOCK:
        return dm_read_frame(handle)


def scan_fast4(
    handle,
    rois,              # type: Optional[List[Dict[str, Any]]]
    timeout_s,         # type: float
    debug=False,       # type: bool
    trace_id=None,     # type: Optional[int]
):
    # type: (...) -> Optional[str]
    """
    이미 열린 handle에서만 스캔.
    fast 경로(0/90/180/270)만 수행. 전처리/헤비 디코드 없음.
    """
    tag = "D#{}".format(trace_id) if trace_id is not None else "D"
    if handle is None:
        if debug:
            print("[{}][{}] cam=None (열리지 않음)".format(tag, ts_wall()))
        return None

    # non-mirror 먼저 → mirror 나중 (dict/namespace 모두 지원)
    cfg_rois_raw = rois if (rois and isinstance(rois, list)) else DM_DEFAULT_ROIS
    rois_nm = [r for r in cfg_rois_raw if not bool(_roi_get(r, "hflip", False))]
    rois_m  = [r for r in cfg_rois_raw if     bool(_roi_get(r, "hflip", False))]
    cfg_rois = rois_nm + rois_m

    t0 = t_now()
    timeout_s = max(0.1, float(timeout_s))
    deadline = t0 + timeout_s
    SAFETY_MS = 40.0
    FAST_BUDGET_MS = 70.0  # ROI당 최대 예산(ms)

    frames = 0
    reads_null = 0
    if debug:
        print("[{}][{}] scan_start timeout={:.2f}s rois={}".format(tag, ts_wall(), timeout_s, len(cfg_rois)))

    try:
        while True:
            now = t_now()
            if now >= deadline:
                break

            left_ms = (deadline - now) * 1000.0
            if left_ms < (FAST_BUDGET_MS + SAFETY_MS):
                if debug:
                    print("[{}][{}] stop_before_frame left≈{:.1f}ms".format(tag, ts_wall(), left_ms))
                break

            # 프레임 읽기
            t_read0 = t_now()
            with DM_LOCK:
                frame = dm_read_frame(handle)
            t_read1 = t_now()
            if frame is None:
                reads_null += 1
                if debug and reads_null <= 5:
                    print("[{}][{}] frame=None read_cost={} elapsed={}".format(
                        tag, ts_wall(), ms(t_read1 - t_read0), ms(t_read1 - t0)))
                time.sleep(0.005)
                continue

            frames += 1
            if debug and frames == 1:
                print("[{}][{}] first_frame read_cost={} T0→first_frame={}".format(
                    tag, ts_wall(), ms(t_read1 - t_read0), ms(t_read1 - t0)))

            # ROI 루프 (fast4만)
            for r in cfg_rois:
                now = t_now()
                left_ms = (deadline - now) * 1000.0
                if left_ms < (FAST_BUDGET_MS + SAFETY_MS):
                    if debug:
                        print("[{}][{}] stop_before_roi left≈{:.1f}ms".format(tag, ts_wall(), left_ms))
                    return None

                name = _roi_get(r, "name", "ROI")
                size = _roi_get(r, "size", [0, 0]) or [0, 0]
                offs = _roi_get(r, "offset", [0, 0]) or [0, 0]
                hflip = bool(_roi_get(r, "hflip", False))

                try:
                    rw, rh = int(size[0]), int(size[1])
                except Exception:
                    rw, rh = 0, 0
                try:
                    dx, dy = int(offs[0]), int(offs[1])
                except Exception:
                    dx, dy = 0, 0

                t_roi0 = t_now()
                roi = dm_crop_roi(frame, int(rw), int(rh), int(dx), int(dy))
                if roi.size == 0:
                    continue
                if hflip:
                    roi = cv2.flip(roi, 1)
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                t_roi1 = t_now()
                if debug:
                    print("[{}][{}] {} roi={}x{} roi_prep={}".format(
                        tag, ts_wall(), name, rw, rh, ms(t_roi1 - t_roi0)))

                fast_budget = int(min(FAST_BUDGET_MS, max(5.0, (deadline - t_now()) * 1000.0 - SAFETY_MS)))
                t_f0 = t_now()
                res = dm_decode_fast4(gray, max_count=3, time_budget_ms=fast_budget)
                t_f1 = t_now()
                if debug:
                    print("[{}][{}] {} fast4_decode={}".format(tag, ts_wall(), name, ms(t_f1 - t_f0)))

                if res:
                    if debug:
                        print("[{}][{}] HIT fast name={} T0→hit={} payload={}".format(
                            tag, ts_wall(), name, ms(t_now() - t0), res[0]))
                    return res[0]

            time.sleep(0.003)

        if debug:
            print("[{}][{}] TIMEOUT frames={} null_reads={} total_elapsed={}".format(
                tag, ts_wall(), frames, reads_null, ms(t_now() - t0)))
        return None

    except Exception as e:
        print("[{}] [dm.persist] scan error:".format(tag), e)
        return None
