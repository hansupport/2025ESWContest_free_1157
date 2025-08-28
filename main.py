# main.py
# - YAML/JSON 설정 로딩
# - depth/img2emb/datamatrix에 설정값 주입
# - SQLite 로깅(모든 샘플 기록) — DB 오픈 가속(스키마 1회/종료 시 WAL truncate)
# - D+Enter: DataMatrix 스캔(빠른 4방향만) + 1초 치수 측정 + 임베딩
# - L+Enter: LGBM만 학습 트리거 (--type lgbm)
# - C+Enter: Centroid만 학습 트리거 (--type centroid)
# - DM 카메라 persistent 공유 + 락
# - 모델: centroid + LGBM 동시 지원(파일 변경 자동 리로드, config.model.type로 추론 우선순위 선택)
# - 추론: 학습과 동일한 L2정규화, top-1 확률 임계 미달 시 3~5프레임 이동평균/다수결로 확정
# - (FIX) centroid 확률: softmax(top-k) 대신 margin(sigmoid)로 ‘확신도’ 산출

import os
import sys
import time
import json
import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Deque, Tuple

import numpy as np
import cv2
import torch
from collections import deque, Counter

# ===== 경로 세팅 =====
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

# 로컬 모듈
import depth as depthmod
from depth import DepthEstimator
import img2emb as embmod

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
    DM_DEFAULT_ROIS = [
        dict(name="ROI1", size=[260, 370], offset=[-380, 100], hflip=True),
        dict(name="ROI2", size=[300, 400], offset=[ 610, 110], hflip=True),
        dict(name="ROI3", size=[480, 340], offset=[ 120,  70], hflip=False),
    ]

np.set_printoptions(suppress=True, linewidth=100000, threshold=np.inf, precision=4)

# =========================
# Config 로딩
# =========================
def load_config():
    """
    우선순위:
      1) <script_stem>.yaml / .yml / .json
      2) config.yaml / config.json
    """
    stem = Path(__file__).stem
    cand = [
        ROOT / f"{stem}.yaml", ROOT / f"{stem}.yml", ROOT / f"{stem}.json",
        ROOT / "config.yaml", ROOT / "config.json"
    ]
    cfg, used = {}, None
    for p in cand:
        if not p.exists():
            continue
        try:
            if p.suffix.lower() in (".yaml", ".yml"):
                import yaml
                with p.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                used = p; break
            if p.suffix.lower() == ".json":
                with p.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                used = p; break
        except Exception as e:
            print(f"[config] 로드 실패: {p} | {e}")
    if used is None:
        print("[config] 파일 없음. 기본값 사용")
    else:
        print(f"[config] 사용 파일: {used}")
    return cfg, used

CFG, CONFIG_PATH = load_config()

# ========= 기본값 + config 적용 =========
# embedding/img2emb
EMB_CAM_DEV   = CFG.get("embedding",{}).get("cam_dev", "/dev/video2")
EMB_CAM_PIX   = CFG.get("embedding",{}).get("pixfmt", "YUYV")
EMB_CAM_W     = int(CFG.get("embedding",{}).get("width", 848))
EMB_CAM_H     = int(CFG.get("embedding",{}).get("height", 480))
EMB_CAM_FPS   = int(CFG.get("embedding",{}).get("fps", 6))
EMB_INPUT_SIZE= int(CFG.get("embedding",{}).get("input_size", 128))
EMB_OUT_DIM   = int(CFG.get("embedding",{}).get("out_dim", 128))
EMB_WIDTH     = float(CFG.get("embedding",{}).get("width_scale", 0.35))
EMB_USE_FP16  = bool(CFG.get("embedding",{}).get("fp16", False))
EMB_USE_DW    = bool(CFG.get("embedding",{}).get("use_depthwise", False))
EMB_USE_BN    = bool(CFG.get("embedding",{}).get("use_bn", False))
EMB_PINNED    = bool(CFG.get("embedding",{}).get("pinned", False))
EMB_WEIGHTS_PATH  = CFG.get("embedding",{}).get("weights_path", None)
E2E_WARMUP_FRAMES = int(CFG.get("embedding",{}).get("e2e_warmup_frames", 60))
E2E_PREGRAB       = int(CFG.get("embedding",{}).get("e2e_pregrab", 8))

# img2emb ROI(중앙 기준)
EMB_ROI_PX   = CFG.get("embedding",{}).get("roi_px", None)
EMB_ROI_OFF  = CFG.get("embedding",{}).get("roi_offset", [0,0])

# depth
DEPTH_W       = int(CFG.get("depth",{}).get("width", 1280))
DEPTH_H       = int(CFG.get("depth",{}).get("height", 720))
DEPTH_FPS     = int(CFG.get("depth",{}).get("fps", 6))
DEPTH_ROI_PX  = CFG.get("depth",{}).get("roi_px", [260,260])
DEPTH_ROI_OFF = CFG.get("depth",{}).get("roi_offset", [20,-100])

# depth 모듈 파라미터 오버라이드(옵션)
for k in ["DECIM","PLANE_TAU","H_MIN_BASE","H_MAX","MIN_OBJ_PIX",
          "BOTTOM_ROI_RATIO","HOLE_FILL","CORE_MARGIN_PX","P_LO","P_HI"]:
    if k in CFG.get("depth",{}):
        setattr(depthmod, k, CFG["depth"][k])
        print(f"[depth.cfg] set {k} = {CFG['depth'][k]}")

# datamatrix (persistent open 전제)
DM_CAMERA: Union[int,str] = CFG.get("datamatrix",{}).get("camera", 2)
DM_RES        = CFG.get("datamatrix",{}).get("prefer_res", [1920,1080])
DM_FPS        = int(CFG.get("datamatrix",{}).get("prefer_fps", 6))
DM_ROIS       = CFG.get("datamatrix",{}).get("rois", None)
DM_SCAN_TIMEOUT_S = float(CFG.get("datamatrix",{}).get("scan_timeout_s", 2.0))

# 품질/저장/모델
Q_WARN   = float(CFG.get("quality",{}).get("q_warn", 0.30))
DB_PATH  = ROOT / CFG.get("storage",{}).get("sqlite_path", "pack.db")
CENTROID_MODEL_PATH = ROOT / CFG.get("model",{}).get("centroids_path", "centroids.npz")
CENTROID_TOPK = int(CFG.get("model",{}).get("topk", 3))

# 스무딩/임계/온도
TOP1_THRESHOLD   = float(CFG.get("model",{}).get("top1_threshold", 0.40))
SMOOTH_WINDOW    = int(CFG.get("model",{}).get("smooth_window", 3))   # 3~5 권장
SMOOTH_MIN_VOTES = int(CFG.get("model",{}).get("smooth_min_votes", max(2, int(round(SMOOTH_WINDOW*0.6)))))
MIN_MARGIN       = float(CFG.get("model",{}).get("min_margin", 0.02))  # 상위-차상위 최소 격차(확률 기준)

# LGBM 모델
MODEL_TYPE = CFG.get("model", {}).get("type", "centroid").lower()  # "centroid"|"lgbm"
LGBM_MODEL_PATH = ROOT / CFG.get("model", {}).get("lgbm_path", "lgbm.npz")  # 확장자 자유(.npz/.json/.pkl 지원)

# (FIX) centroid margin→sigmoid 스케일
CENTROID_MARGIN_SCALE = float(CFG.get("model",{}).get("centroid_margin_scale", 1500.0))

# ---- DM 디버그 플래그 & 타임스탬프 유틸 ----
DM_DEBUG = bool(CFG.get("debug", {}).get("datamatrix", True))
DM_TRACE_ID = 0

def _ts_wall():
    return time.strftime("%H:%M:%S.%f", time.localtime())[:-3]

def _t_now():
    return time.perf_counter()

def _ms(dt):
    return f"{dt*1000:.2f} ms"

# ==========================
# 공용 유틸
# ==========================
def stdin_readline_nonblock(timeout_sec=0.05):
    import select
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
                if dev == "cuda": torch.cuda.synchronize()
        print("[warmup] Torch done")
    except Exception as e:
        print(f"[warmup] Torch error: {e}")

# ==========================
# img2emb
# ==========================
def apply_img2emb_roi_from_cfg():
    try:
        if EMB_ROI_PX is not None and hasattr(embmod, "set_center_roi"):
            embmod.set_center_roi(EMB_ROI_PX, EMB_ROI_OFF)
            print(f"[img2emb.cfg] ROI(center) px={EMB_ROI_PX} off={EMB_ROI_OFF}")
            return
        if EMB_ROI_PX is not None:
            w, h = int(EMB_ROI_PX[0]), int(EMB_ROI_PX[1])
            dx, dy = int(EMB_ROI_OFF[0]), int(EMB_ROI_OFF[1])
            cx, cy = EMB_CAM_W//2 + dx, EMB_CAM_H//2 + dy
            x = max(0, min(EMB_CAM_W - w,  cx - w//2))
            y = max(0, min(EMB_CAM_H - h,  cy - h//2))
            embmod.ROI = (x, y, w, h)
            print(f"[img2emb.cfg] ROI(xywh)={(x,y,w,h)} (fallback)")
    except Exception as e:
        print("[img2emb.cfg] ROI 적용 실패:", e)

def build_embedder_only():
    apply_img2emb_roi_from_cfg()
    print("[warmup] img2emb: build embedder")
    emb = embmod.TorchTinyMNetEmbedder(
        out_dim=EMB_OUT_DIM,
        width=EMB_WIDTH,
        size=EMB_INPUT_SIZE,
        fp16=EMB_USE_FP16,
        weights_path=EMB_WEIGHTS_PATH,
        channels_last=False,
        cudnn_benchmark=False,
        warmup_steps=3,
        use_depthwise=EMB_USE_DW,
        use_bn=EMB_USE_BN,
        pinned=EMB_PINNED
    )
    print("[warmup] img2emb: embedder ready")
    return emb

def open_embed_camera():
    cap = embmod.open_camera(
        EMB_CAM_DEV, backend="auto",
        w=EMB_CAM_W, h=EMB_CAM_H, fps=EMB_CAM_FPS, pixfmt=EMB_CAM_PIX
    )
    if not cap or not cap.isOpened():
        raise RuntimeError(f"임베딩 카메라 열기 실패: {EMB_CAM_DEV}")
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("임베딩 카메라 첫 프레임 실패")
    return cap

def warmup_cv2_cap(cap, seconds=0.4):
    t0 = time.time(); n = 0
    while time.time() - t0 < max(0.0, seconds):
        ok, _ = cap.read()
        if not ok:
            time.sleep(0.01); continue
        n += 1
        time.sleep(0.005)
    return n

def embed_one_frame(emb, pregrab=3):
    cap = open_embed_camera()
    try:
        for _ in range(max(0, pregrab)): cap.grab()
        ok, bgr = cap.read()
        if not ok: return None
        v = emb.embed_bgr(bgr)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return v.astype(np.float32)
    finally:
        try: cap.release()
        except Exception: pass

# ==========================
# DataMatrix: persistent camera helpers (+ 공유 락)
# ==========================
DM_CAM_PERSIST = None
DM_CAM_LOCK = threading.Lock()

def _same_device(a: Union[int,str], b: Union[int,str]) -> bool:
    def norm(x):
        if isinstance(x, int):
            return f"/dev/video{x}"
        if isinstance(x, str) and x.isdigit():
            return f"/dev/video{int(x)}"
        return x
    return norm(a) == norm(b)

def open_dm_cam_persistent():
    global DM_CAM_PERSIST
    if DM_CAM_PERSIST is not None:
        return DM_CAM_PERSIST
    try:
        DM_CAM_PERSIST = dm_open_camera(DM_CAMERA, (int(DM_RES[0]), int(DM_RES[1])), int(DM_FPS))
        t0 = time.time(); read = 0
        while time.time() - t0 < 0.6:
            with DM_CAM_LOCK:
                f = dm_read_frame(DM_CAM_PERSIST)
            if f is not None:
                read += 1
            time.sleep(0.005)
        print(f"[dm.persist] opened {DM_CAMERA} and prewarmed, frames={read}")
    except Exception as e:
        DM_CAM_PERSIST = None
        print("[dm.persist] open failed:", e)
    return DM_CAM_PERSIST

def close_dm_cam_persistent():
    global DM_CAM_PERSIST
    if DM_CAM_PERSIST is None:
        return
    try:
        if DM_CAM_PERSIST[0] == "realsense":
            pipeline, rs = DM_CAM_PERSIST[1]; pipeline.stop()
        else:
            DM_CAM_PERSIST[1].release()
    except Exception:
        pass
    DM_CAM_PERSIST = None
    print("[dm.persist] closed")

def datamatrix_scan_on_cam(
    cam,
    rois: Optional[List[Dict[str,Any]]],
    timeout_s: float,
    debug: bool = False,
    trace_id: Optional[int] = None,
) -> Optional[str]:
    """
    이미 열린 cam에서만 스캔.
    요청 반영: fast 경로(0/90/180/270)만 수행. 전처리/헤비 디코드 없음.
    """
    tag = f"D#{trace_id}" if trace_id is not None else "D"
    if cam is None:
        if debug: print(f"[{tag}][{_ts_wall()}] cam=None (열리지 않음)")
        return None

    # non-mirror 먼저 → mirror 나중
    cfg_rois_raw = rois if (rois and isinstance(rois, list)) else DM_DEFAULT_ROIS
    rois_nm = [r for r in cfg_rois_raw if not bool(r.get("hflip", False))]
    rois_m  = [r for r in cfg_rois_raw if     bool(r.get("hflip", False))]
    cfg_rois = rois_nm + rois_m

    t0 = _t_now()
    deadline = t0 + max(0.1, float(timeout_s))
    SAFETY_MS = 40.0
    FAST_BUDGET_MS = 70.0  # ROI당 최대 예산

    frames = 0
    reads_null = 0
    if debug:
        print(f"[{tag}][{_ts_wall()}] scan_start timeout={timeout_s:.2f}s rois={len(cfg_rois)}")

    try:
        while True:
            now = _t_now()
            if now >= deadline:
                break

            left_ms = (deadline - now) * 1000.0
            if left_ms < (FAST_BUDGET_MS + SAFETY_MS):
                if debug:
                    print(f"[{tag}][{_ts_wall()}] stop_before_frame left≈{left_ms:.1f}ms")
                break

            # 프레임 읽기
            t_read0 = _t_now()
            with DM_CAM_LOCK:
                frame = dm_read_frame(cam)
            t_read1 = _t_now()
            if frame is None:
                reads_null += 1
                if debug and reads_null <= 5:
                    print(f"[{tag}][{_ts_wall()}] frame=None read_cost={_ms(t_read1 - t_read0)} "
                          f"elapsed={_ms(t_read1 - t0)}")
                time.sleep(0.005)
                continue

            frames += 1
            if debug and frames == 1:
                print(f"[{tag}][{_ts_wall()}] first_frame read_cost={_ms(t_read1 - t_read0)} "
                      f"T0→first_frame={_ms(t_read1 - t0)}")

            # ROI 루프 (fast4만)
            for r in cfg_rois:
                now = _t_now()
                left_ms = (deadline - now) * 1000.0
                if left_ms < (FAST_BUDGET_MS + SAFETY_MS):
                    if debug:
                        print(f"[{tag}][{_ts_wall()}] stop_before_roi left≈{left_ms:.1f}ms")
                    return None

                name = r.get("name","ROI")
                (rw, rh) = r.get("size",[0,0])
                (dx, dy) = r.get("offset",[0,0])
                hflip = bool(r.get("hflip", False))

                t_roi0 = _t_now()
                roi = dm_crop_roi(frame, int(rw), int(rh), int(dx), int(dy))
                if roi.size == 0:
                    continue
                if hflip:
                    roi = cv2.flip(roi, 1)
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                t_roi1 = _t_now()
                if debug:
                    print(f"[{tag}][{_ts_wall()}] {name} roi={rw}x{rh} roi_prep={_ms(t_roi1 - t_roi0)}")

                fast_budget = int(min(FAST_BUDGET_MS, max(5.0, (deadline - _t_now())*1000.0 - SAFETY_MS)))
                t_f0 = _t_now()
                res = dm_decode_fast4(gray, max_count=3, time_budget_ms=fast_budget)
                t_f1 = _t_now()
                if debug:
                    print(f"[{tag}][{_ts_wall()}] {name} fast4_decode={_ms(t_f1 - t_f0)}")

                if res:
                    if debug:
                        print(f"[{tag}][{_ts_wall()}] HIT fast name={name} "
                              f"T0→hit={_ms(_t_now() - t0)} payload={res[0]}")
                    return res[0]

            time.sleep(0.003)

        if debug:
            print(f"[{tag}][{_ts_wall()}] TIMEOUT frames={frames} null_reads={reads_null} "
                  f"total_elapsed={_ms(_t_now() - t0)}")
        return None

    except Exception as e:
        print(f"[{tag}] [dm.persist] scan error:", e)
        return None

def datamatrix_scan_persistent(timeout_s: float, debug: bool = False, trace_id: Optional[int] = None) -> Optional[str]:
    cam = open_dm_cam_persistent()
    return datamatrix_scan_on_cam(cam, DM_ROIS, timeout_s, debug=debug, trace_id=trace_id)

def e2e_warmup_now_shared(emb, frames: int = E2E_WARMUP_FRAMES, pregrab: int = E2E_PREGRAB):
    if _same_device(DM_CAMERA, EMB_CAM_DEV) and (DM_CAM_PERSIST is not None):
        print(f"[warmup] img2emb: shared e2e warmup {frames} frames (pregrab={pregrab}) via DM persistent cam")
        with DM_CAM_LOCK:
            for _ in range(max(0, pregrab)):
                _ = dm_read_frame(DM_CAM_PERSIST)
        t0 = time.time(); n_ok = 0
        with torch.inference_mode():
            for _ in range(max(1, frames)):
                with DM_CAM_LOCK:
                    bgr = dm_read_frame(DM_CAM_PERSIST)
                if bgr is None:
                    time.sleep(0.003)
                    continue
                _ = emb.embed_bgr(bgr)
                n_ok += 1
        if torch.cuda.is_available(): torch.cuda.synchronize()
        print(f"[warmup] img2emb: shared e2e done, ok_frames={n_ok}, elapsed={time.time()-t0:.2f}s")
        return
    print("[warmup] img2emb: separate device warmup path")
    cap = open_embed_camera()
    try:
        _ = warmup_cv2_cap(cap, seconds=0.4)
        embmod.e2e_warmup(emb, cap, n=frames, pregrab=pregrab)
    finally:
        try: cap.release()
        except Exception: pass
    print("[warmup] img2emb: e2e done")

def embed_one_frame_shared(emb, pregrab=3):
    if _same_device(DM_CAMERA, EMB_CAM_DEV) and (DM_CAM_PERSIST is not None):
        with DM_CAM_LOCK:
            for _ in range(max(0, pregrab)):
                _ = dm_read_frame(DM_CAM_PERSIST)
            bgr = dm_read_frame(DM_CAM_PERSIST)
        if bgr is None:
            return None
        v = emb.embed_bgr(bgr)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return v.astype(np.float32)
    return embed_one_frame(emb, pregrab=pregrab)

# ==========================
# centroid 모델
# ==========================
def load_centroid_model(path=CENTROID_MODEL_PATH):
    if not path.exists(): return None, None
    z = np.load(path, allow_pickle=True)
    C = z["C"].astype(np.float32)
    labels = z["labels"]
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True)+1e-8)
    return Cn, labels

def predict_with_centroid(x143, Cn, labels, topk=CENTROID_TOPK):
    xx = x143 / (np.linalg.norm(x143)+1e-8)
    sims = Cn @ xx
    idx = np.argsort(-sims)[:topk]
    return [(str(labels[i]), float(sims[i])) for i in idx]

# (FIX) centroid 확신도: margin(sigmoid)
def centroid_conf_from_topk(preds: List[Tuple[str, float]], scale: float = 1500.0) -> Tuple[str, float, float, float]:
    """
    preds: [(label, sim)] (정렬 여부 무관)
    반환: (top_label, conf_p, margin, second_sim)
      - conf_p = sigmoid(scale * (sim1 - sim2))
      - gap_prob = 2*conf_p - 1 와 동일(스무딩용)
    """
    if not preds:
        return "", 0.0, 0.0, 0.0
    arr = sorted(preds, key=lambda x: x[1], reverse=True)
    lab1, s1 = arr[0]
    s2 = arr[1][1] if len(arr) > 1 else -1.0
    margin = float(s1 - s2)
    conf = float(1.0 / (1.0 + np.exp(-scale * margin))) if len(arr) > 1 else 1.0
    return str(lab1), conf, margin, s2

# ==========================
# LGBM 모델 (신/구 포맷 자동 호환: .npz/.json/.pkl)
# ==========================
def _try_load_lgbm_npz(path: Path):
    try:
        z = np.load(str(path), allow_pickle=True)
        booster_str = z["booster_str"].item() if hasattr(z["booster_str"], "item") else z["booster_str"]
        if isinstance(booster_str, (bytes, bytearray)):
            booster_str = booster_str.decode("utf-8")
        best_it = int(z["best_iteration"]) if "best_iteration" in z else None
        classes = z["classes_"]
        return booster_str, classes, best_it
    except Exception:
        return None

def _try_load_lgbm_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        booster_str = obj.get("booster_str", None)
        best_it = obj.get("best_iteration", None)
        classes = obj.get("classes_", None)
        return booster_str, classes, best_it
    except Exception:
        return None

def _try_load_lgbm_joblib(path: Path):
    try:
        from joblib import load
        obj = load(str(path))
        if isinstance(obj, dict) and "booster_str" in obj:
            return obj["booster_str"], obj.get("classes_", None), obj.get("best_iteration", None)
        # 구 포맷(래퍼)
        clf = obj.get("model", None) if isinstance(obj, dict) else None
        classes = obj.get("classes_", getattr(clf, "classes_", None) if clf else None) if isinstance(obj, dict) else None
        best_it = getattr(clf, "best_iteration_", None) if clf else None
        return clf, classes, best_it  # 주의: 이 경우 booster_str이 아니라 wrapper 반환
    except Exception:
        return None

def load_lgbm_model(path=LGBM_MODEL_PATH):
    if not path.exists():
        return None, None, None
    # 1) NPZ (피클-프리 권장)
    if path.suffix.lower() == ".npz":
        triple = _try_load_lgbm_npz(path)
        if triple:
            booster_str, classes, best_it = triple
            import lightgbm as lgb
            booster = lgb.Booster(model_str=booster_str)
            return booster, np.array(classes), best_it
    # 2) JSON
    if path.suffix.lower() == ".json":
        triple = _try_load_lgbm_json(path)
        if triple:
            booster_str, classes, best_it = triple
            import lightgbm as lgb
            booster = lgb.Booster(model_str=booster_str)
            return booster, np.array(classes), best_it
    # 3) Joblib (레거시 호환)
    jl = _try_load_lgbm_joblib(path)
    if jl:
        if isinstance(jl[0], str):
            import lightgbm as lgb
            booster = lgb.Booster(model_str=jl[0])
            return booster, np.array(jl[1]), jl[2]
        # 구 포맷의 sklearn wrapper
        return jl[0], np.array(jl[1]) if jl[1] is not None else None, jl[2]

    # 4) 마지막 폴백: 내용 추론 시도
    try:
        txt = path.read_text(encoding="utf-8")
        if txt.strip().startswith("tree"):
            import lightgbm as lgb
            booster = lgb.Booster(model_str=txt)
            return booster, None, None
    except Exception:
        pass

    print("[model] lgbm load failed (unsupported format)")
    return None, None, None

# ---- (NEW) LGBM 전처리/차원 보정 유틸 ----
def _lgbm_expected_dim(model):
    for attr in ("n_features_", "n_features_in_"):
        if hasattr(model, attr):
            try:
                v = int(getattr(model, attr))
                if v > 0:
                    return v
            except Exception:
                pass
    if hasattr(model, "booster_"):
        try:
            return int(model.booster_.num_feature())
        except Exception:
            pass
    if hasattr(model, "num_feature"):
        try:
            return int(model.num_feature())
        except Exception:
            pass
    return None

def _ensure_feat_dim(model, x: np.ndarray) -> np.ndarray:
    exp = _lgbm_expected_dim(model)
    if exp is None:
        return x
    cur = x.shape[1]
    if cur == exp:
        return x
    if cur > exp:
        print(f"[infer] warn: feature_dim {cur} > expected {exp} → slice")
        return x[:, :exp]
    # cur < exp
    print(f"[infer] warn: feature_dim {cur} < expected {exp} → pad zeros")
    pad = np.zeros((x.shape[0], exp - cur), dtype=x.dtype)
    return np.hstack([x, pad])

def lgbm_predict_proba(model, classes, vec143: np.ndarray, best_iteration=None) -> np.ndarray:
    x = vec143.reshape(1, -1).astype(np.float32)
    x = _ensure_feat_dim(model, x)
    if hasattr(model, "predict_proba"):  # sklearn wrapper
        probs = model.predict_proba(x)[0]
    else:  # lightgbm.Booster
        num_it = int(best_iteration) if (best_iteration is not None and int(best_iteration) > 0) else None
        probs = model.predict(x, num_iteration=num_it)[0]
    return np.asarray(probs, dtype=np.float32)

# ==========================
# (NEW) 스무더(확률 기반 다수결)
# ==========================
class ProbSmoother:
    """
    최근 N 프레임의 top-1 후보를 수집해 다수결(+평균 확률)로 확정.
    - window: 버퍼 길이
    - min_votes: 다수결 최소 표수
    """
    def __init__(self, window=3, min_votes=2):
        self.window = int(window)
        self.min_votes = int(min_votes)
        self.buf: List[Tuple[str,float]] = []  # [(label, prob)]

    def push(self, label, prob):
        self.buf.append((str(label), float(prob)))
        if len(self.buf) > self.window:
            self.buf.pop(0)

    def status(self):
        from collections import Counter
        cnt = Counter([lab for lab, _ in self.buf])
        if not cnt:
            return None, 0, 0.0
        top_lab, votes = cnt.most_common(1)[0]
        avg_p = float(np.mean([p for lab, p in self.buf if lab == top_lab]))
        return top_lab, votes, avg_p

    def maybe_decide(self, threshold=0.40):
        if len(self.buf) < self.window:
            return None
        lab, votes, avg_p = self.status()
        if votes >= self.min_votes and avg_p >= threshold:
            self.buf.clear()
            return (lab, avg_p)
        return None

# ==========================
# SQLite 스토리지 (가속 패치 반영)
# ==========================
def open_db(db_path: Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=5.0)
    t1 = time.perf_counter()

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=268435456")  # 256MB

    ver = conn.execute("PRAGMA user_version").fetchone()[0]

    if ver == 0:
        t_schema0 = time.perf_counter()
        conn.execute("""
        CREATE TABLE IF NOT EXISTS sample_log (
          sample_id INTEGER PRIMARY KEY,
          ts_unix   REAL NOT NULL,
          product_id TEXT,
          has_label  INTEGER NOT NULL DEFAULT 0,
          d1 REAL, d2 REAL, d3 REAL,
          mad1 REAL, mad2 REAL, mad3 REAL,
          r1 REAL, r2 REAL, r3 REAL,
          sr1 REAL, sr2 REAL, sr3 REAL,
          logV REAL, logsV REAL, q REAL,
          emb BLOB NOT NULL,
          origin TEXT
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sample_ts ON sample_log(ts_unix)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sample_label ON sample_log(product_id, has_label)")
        conn.execute("PRAGMA user_version=1")
        t_schema1 = time.perf_counter()
        print(f"[db] schema init {(t_schema1 - t_schema0)*1000:.1f} ms")

    print(f"[db] connect {(t1 - t0)*1000:.1f} ms  | total open {(time.perf_counter()-t0)*1000:.1f} ms")
    return conn

def emb_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes(order="C")

def insert_sample(conn, feat15: dict, emb128: np.ndarray, product_id, has_label: int, origin: str):
    vals = (
        time.time(), product_id, int(has_label),
        float(feat15["d1"]),  float(feat15["d2"]),  float(feat15["d3"]),
        float(feat15["mad1"]), float(feat15["mad2"]), float(feat15["mad3"]),
        float(feat15["r1"]),  float(feat15["r2"]),  float(feat15["r3"]),
        float(feat15["sr1"]), float(feat15["sr2"]), float(feat15["sr3"]),
        float(feat15["logV"]), float(feat15["logsV"]), float(feat15["q"]),
        emb_to_blob(emb128), origin
    )
    conn.execute("""
    INSERT INTO sample_log(
      ts_unix, product_id, has_label,
      d1,d2,d3,mad1,mad2,mad3,r1,r2,r3,sr1,sr2,sr3,logV,logsV,q,
      emb, origin
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, vals)

def on_sample_record(conn, feat15: dict, emb128: np.ndarray, product_id, has_label: int, origin: str):
    insert_sample(conn, feat15, emb128, product_id, has_label, origin)
    meta = [
        feat15["d1"], feat15["d2"], feat15["d3"],
        feat15["mad1"], feat15["mad2"], feat15["mad3"],
        feat15["r1"], feat15["r2"], feat15["r3"],
        feat15["sr1"], feat15["sr2"], feat15["sr3"],
        feat15["logV"], feat15["logsV"], feat15["q"]
    ]
    full_vec = np.concatenate([np.array(meta, np.float32), emb128], axis=0)
    print(f"[record] origin={origin} has_label={has_label} product_id={product_id}")
    print(f"[vector] dim={full_vec.shape[0]}")
    print(full_vec)

# ==========================
# 학습 트리거
# ==========================
def run_training_now(config_path: Optional[Path], force_type: Optional[str] = None):
    train_py = MODEL_DIR / "train.py"
    args = [sys.executable, str(train_py)]
    if config_path is not None:
        args += ["--config", str(config_path)]
    if force_type in ("lgbm", "centroid"):
        args += ["--type", force_type]
    print("[train] 시작:", " ".join(args))
    try:
        r = subprocess.run(args, check=False)
        print(f"[train] 종료 코드={r.returncode}")
    except Exception as e:
        print("[train] 실행 실패:", e)

# ==========================
# 모델 즉시 로더
# ==========================
last_mtime_cent = 0.0
last_mtime_lgbm = 0.0
Cn, labels = (None, None)
lgbm_model, lgbm_classes, lgbm_best_it = (None, None, None)

def ensure_lgbm_loaded():
    global lgbm_model, lgbm_classes, lgbm_best_it, last_mtime_lgbm
    if lgbm_model is not None and lgbm_classes is not None:
        return True
    if LGBM_MODEL_PATH.exists():
        last_mtime_lgbm = LGBM_MODEL_PATH.stat().st_mtime
        lgbm_model, lgbm_classes, lgbm_best_it = load_lgbm_model()
        return (lgbm_model is not None) and (lgbm_classes is not None)
    return False

def ensure_centroid_loaded():
    global Cn, labels, last_mtime_cent
    if Cn is not None and labels is not None:
        return True
    if CENTROID_MODEL_PATH.exists():
        last_mtime_cent = CENTROID_MODEL_PATH.stat().st_mtime
        Cn, labels = load_centroid_model()
        return (Cn is not None) and (labels is not None)
    return False

# ==========================
# 메인
# ==========================
def main():
    t_all = time.time()
    print("[init] start")
    maybe_run_jetson_perf()
    warmup_opencv_kernels()
    warmup_torch_cuda()

    # DB
    conn = open_db(DB_PATH)

    # DepthEstimator 구성값 주입(roi_px+roi_offset)
    try:
        roi_px = (int(DEPTH_ROI_PX[0]), int(DEPTH_ROI_PX[1]))
    except Exception:
        roi_px = (260,260)
    try:
        roi_off = (int(DEPTH_ROI_OFF[0]), int(DEPTH_ROI_OFF[1]))
    except Exception:
        roi_off = (20,-100)

    depth = DepthEstimator(
        width=DEPTH_W, height=DEPTH_H, fps=DEPTH_FPS,
        roi_px=roi_px, roi_offset=roi_off
    )
    depth.start()
    frames = depth.warmup(seconds=1.5)
    print(f"[warmup] RealSense frames={frames}")
    ok_calib = depth.calibrate(max_seconds=3.0)
    if not ok_calib:
        print("[fatal] depth calib 실패. 바닥만 보이게 하고 재실행하세요.")
        try: depth.stop()
        except: pass
        try: conn.close()
        except: pass
        return

    # DataMatrix 카메라 persistent open
    open_dm_cam_persistent()

    # 임베더 준비
    if _same_device(DM_CAMERA, EMB_CAM_DEV):
        print(f"[warn] DM_CAMERA({DM_CAMERA})와 EMB_CAM_DEV({EMB_CAM_DEV}) 동일. shared persistent handle + lock 사용")
    emb = build_embedder_only()
    print(f"[img2emb.cfg] dev={EMB_CAM_DEV} {EMB_CAM_W}x{EMB_CAM_H}@{EMB_CAM_FPS} pixfmt={EMB_CAM_PIX}")

    # e2e 워밍업
    e2e_warmup_now_shared(emb, frames=E2E_WARMUP_FRAMES, pregrab=E2E_PREGRAB)

    # 초기 모델 로드(있으면)
    global Cn, labels, lgbm_model, lgbm_classes, lgbm_best_it, last_mtime_cent, last_mtime_lgbm
    if CENTROID_MODEL_PATH.exists():
        last_mtime_cent = CENTROID_MODEL_PATH.stat().st_mtime
        Cn, labels = load_centroid_model()
        if labels is not None:
            print(f"[model] centroid 로드: {len(labels)} classes")
    if LGBM_MODEL_PATH.exists():
        last_mtime_lgbm = LGBM_MODEL_PATH.stat().st_mtime
        lgbm_model, lgbm_classes, lgbm_best_it = load_lgbm_model()
        if lgbm_model is not None:
            print(f"[model] lgbm 로드: classes={len(lgbm_classes)} best_it={lgbm_best_it}")

    # 스무더 준비
    smoother = ProbSmoother(window=SMOOTH_WINDOW, min_votes=SMOOTH_MIN_VOTES)

    print("[ready] total init %.2fs" % (time.time()-t_all))
    print("[hint] D + Enter = 수동 측정 1초 & DataMatrix 스캔")
    print("[hint] L + Enter = LGBM 학습(model/train.py --type lgbm)")
    print("[hint] C + Enter = Centroid 학습(model/train.py --type centroid)")

    try:
        while True:
            # 모델 업데이트 감지
            if CENTROID_MODEL_PATH.exists():
                m = CENTROID_MODEL_PATH.stat().st_mtime
                if m > last_mtime_cent:
                    last_mtime_cent = m
                    Cn, labels = load_centroid_model()
                    if labels is not None:
                        print(f"[update] centroid 업데이트: {len(labels)} classes 로드")

            if LGBM_MODEL_PATH.exists():
                m2 = LGBM_MODEL_PATH.stat().st_mtime
                if m2 > last_mtime_lgbm:
                    last_mtime_lgbm = m2
                    lgbm_model, lgbm_classes, lgbm_best_it = load_lgbm_model()
                    if lgbm_model is not None:
                        print(f"[update] lgbm 업데이트: classes={len(lgbm_classes)} best_it={lgbm_best_it}")

            # 수동 트리거 / 학습 트리거
            cmd = stdin_readline_nonblock(0.05)
            if not cmd:
                continue

            uc = cmd.strip().upper()
            if uc == "T":
                print("[hint] 이제는 L/C로 모델별 학습이 가능합니다. (L=LGBM, C=Centroid)")
                run_training_now(CONFIG_PATH, force_type=None)
                # 학습 후 재로딩
                if CENTROID_MODEL_PATH.exists():
                    last_mtime_cent = CENTROID_MODEL_PATH.stat().st_mtime
                    Cn, labels = load_centroid_model()
                    if labels is not None:
                        print(f"[model] centroid reloaded: {len(labels)} classes")
                if LGBM_MODEL_PATH.exists():
                    last_mtime_lgbm = LGBM_MODEL_PATH.stat().st_mtime
                    lgbm_model, lgbm_classes, lgbm_best_it = load_lgbm_model()
                    if lgbm_model is not None:
                        print(f"[model] lgbm reloaded: classes={len(lgbm_classes)} best_it={lgbm_best_it}")
                continue

            if uc == "L":
                run_training_now(CONFIG_PATH, force_type="lgbm")
                if LGBM_MODEL_PATH.exists():
                    last_mtime_lgbm = LGBM_MODEL_PATH.stat().st_mtime
                    lgbm_model, lgbm_classes, lgbm_best_it = load_lgbm_model()
                    if lgbm_model is not None:
                        print(f"[model] lgbm reloaded: classes={len(lgbm_classes)} best_it={lgbm_best_it}")
                continue

            if uc == "C":
                run_training_now(CONFIG_PATH, force_type="centroid")
                if CENTROID_MODEL_PATH.exists():
                    last_mtime_cent = CENTROID_MODEL_PATH.stat().st_mtime
                    Cn, labels = load_centroid_model()
                    if labels is not None:
                        print(f"[model] centroid reloaded: {len(labels)} classes")
                continue

            if uc == "D":
                global DM_TRACE_ID
                DM_TRACE_ID += 1
                tid = DM_TRACE_ID
                t_press = _t_now()
                print(f"[D#{tid}][{_ts_wall()}] key_down → scan_call")

                # 1) DataMatrix 스캔 (fast4만)
                t_s0 = _t_now()
                payload = datamatrix_scan_persistent(DM_SCAN_TIMEOUT_S, debug=DM_DEBUG, trace_id=tid)
                t_s1 = _t_now()
                print(f"[D#{tid}][{_ts_wall()}] scan_return elapsed={_ms(t_s1 - t_s0)} "
                      f"Tpress→scan_return={_ms(t_s1 - t_press)} payload={'YES' if payload else 'NO'}")
                if payload:
                    print(f"[dm] payload={payload}")
                else:
                    print("[dm] payload 없음")

                # 2) 1초 치수 측정
                t_depth0 = _t_now()
                feat = depth.measure_dimensions(duration_s=1.0, n_frames=10)
                t_depth1 = _t_now()
                print(f"[D#{tid}][{_ts_wall()}] depth_measure elapsed={_ms(t_depth1 - t_depth0)}")
                if feat is None:
                    print("[manual] 측정 실패"); continue

                # 3) 임베딩 1프레임
                t_emb0 = _t_now()
                vec = embed_one_frame_shared(emb, pregrab=3)
                t_emb1 = _t_now()
                print(f"[D#{tid}][{_ts_wall()}] embed_one_frame elapsed={_ms(t_emb1 - t_emb0)}")
                if vec is None:
                    print("[manual] 임베딩 실패"); continue

                # 4) 품질 경고
                if feat["q"] < Q_WARN:
                    print(f"[notify] 품질 경고: q={feat['q']:.2f} (임계 {Q_WARN:.2f})")

                # 5) 저장
                if payload:
                    on_sample_record(conn, feat, vec, product_id=payload, has_label=1, origin="manual_dm")
                    # 확정 라벨 들어오면 스무더 초기화
                    smoother.buf.clear()
                    continue
                else:
                    on_sample_record(conn, feat, vec, product_id=None, has_label=0, origin="manual_no_dm")

                # === 학습과 동일한 L2 정규화로 full_vec 구성 ===
                meta = np.array([
                    feat["d1"], feat["d2"], feat["d3"],
                    feat["mad1"], feat["mad2"], feat["mad3"],
                    feat["r1"], feat["r2"], feat["r3"],
                    feat["sr1"], feat["sr2"], feat["sr3"],
                    feat["logV"], feat["logsV"], feat["q"]
                ], np.float32)
                full_vec = np.concatenate([meta, vec], axis=0)
                full_vec = np.where(np.isfinite(full_vec), full_vec, 0.0).astype(np.float32)
                norm = float(np.linalg.norm(full_vec))
                full_vec = full_vec / (norm + 1e-8)
                print(f"[debug] ||full_vec||={np.linalg.norm(full_vec):.6f}")

                # ---- 추론 실행 (우선순위: config.model.type) ----
                ran = False
                top_lab, top_p, gap = None, 0.0, 0.0

                if MODEL_TYPE == "lgbm":
                    # LGBM 우선
                    if ensure_lgbm_loaded():
                        try:
                            probs = lgbm_predict_proba(lgbm_model, lgbm_classes, full_vec, best_iteration=lgbm_best_it)
                            idx = np.argsort(-probs)
                            top_lab = str(lgbm_classes[idx[0]])
                            p1 = float(probs[idx[0]])
                            p2 = float(probs[idx[1]]) if len(idx) > 1 else 0.0
                            top_p = p1
                            gap = p1 - p2
                            print(f"[infer] LGBM top1={top_lab} p={p1:.3f} gap={gap:.4f}")
                            ran = True
                        except Exception as e:
                            print("[infer] lgbm error:", e)

                    # 폴백: centroid
                    if not ran and ensure_centroid_loaded():
                        try:
                            preds = predict_with_centroid(full_vec, Cn, labels, topk=CENTROID_TOPK)
                            print("[infer] centroid top-k:", preds)
                            lab, conf, margin, s2 = centroid_conf_from_topk(preds, scale=CENTROID_MARGIN_SCALE)
                            top_lab, top_p = lab, conf
                            gap = 2.0*conf - 1.0  # (p1 - p2) with two-class view
                            print(f"[infer] centroid_conf: top={lab} conf={conf:.3f} margin={margin:.6f}")
                            ran = True
                        except Exception as e:
                            print("[infer] centroid error:", e)

                else:
                    # centroid 우선
                    if ensure_centroid_loaded():
                        try:
                            preds = predict_with_centroid(full_vec, Cn, labels, topk=CENTROID_TOPK)
                            print("[infer] centroid top-k:", preds)
                            lab, conf, margin, s2 = centroid_conf_from_topk(preds, scale=CENTROID_MARGIN_SCALE)
                            top_lab, top_p = lab, conf
                            gap = 2.0*conf - 1.0
                            print(f"[infer] centroid_conf: top={lab} conf={conf:.3f} margin={margin:.6f}")
                            ran = True
                        except Exception as e:
                            print("[infer] centroid error:", e)

                    # 폴백: LGBM
                    if not ran and ensure_lgbm_loaded():
                        try:
                            probs = lgbm_predict_proba(lgbm_model, lgbm_classes, full_vec, best_iteration=lgbm_best_it)
                            idx = np.argsort(-probs)
                            top_lab = str(lgbm_classes[idx[0]])
                            p1 = float(probs[idx[0]])
                            p2 = float(probs[idx[1]]) if len(idx) > 1 else 0.0
                            top_p = p1
                            gap = p1 - p2
                            print(f"[infer] LGBM top1={top_lab} p={p1:.3f} gap={gap:.4f}")
                            ran = True
                        except Exception as e:
                            print("[infer] lgbm error:", e)

                if not ran or top_lab is None:
                    print("[infer] 모델 없음(파일 미존재 또는 로드 실패)")
                    continue

                # ---- 임계치/마진/스무딩 ----
                # 1) 마진(확률 차) 게이트
                if gap < MIN_MARGIN:
                    smoother.push(top_lab, top_p)
                    print(f"[smooth] hold: small_margin gap={gap:.4f} (<{MIN_MARGIN:.3f}), "
                          f"len={len(smoother.buf)}/{SMOOTH_WINDOW}, top={top_lab} p={top_p:.2f}")
                    continue

                # 2) top-1 확률 임계 미만이면 보류
                if top_p < TOP1_THRESHOLD:
                    smoother.push(top_lab, top_p)
                    print(f"[smooth] hold: len={len(smoother.buf)}/{SMOOTH_WINDOW}, "
                          f"top={top_lab} p={top_p:.2f} (<{TOP1_THRESHOLD:.2f})")
                    continue

                # 3) 스무더로 확정 시도
                smoother.push(top_lab, top_p)
                decided = smoother.maybe_decide(threshold=TOP1_THRESHOLD)
                if decided is None:
                    lab, votes, avgp = smoother.status()
                    if lab is None:
                        print(f"[smooth] hold: len={len(smoother.buf)}/{SMOOTH_WINDOW}")
                    else:
                        print(f"[smooth] hold: len={len(smoother.buf)}/{SMOOTH_WINDOW}, "
                              f"lead={lab} votes={votes} avgp={avgp:.2f}")
                else:
                    lab, avgp = decided
                    print(f"[decision] smoothed: {lab} p={avgp:.2f}")

                continue

    except KeyboardInterrupt:
        print("[exit] keyboard interrupt")
    finally:
        # 종료 시 WAL 비우기 → 다음 오픈 빨라짐
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA optimize")
        except Exception:
            pass
        try:
            depth.stop()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        try:
            close_dm_cam_persistent()
        except Exception:
            pass
        print("[cleanup] stopped")

if __name__ == "__main__":
    main()
