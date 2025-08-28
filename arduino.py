# -*- coding: utf-8 -*-
# main.py
# - YAML/JSON 설정 로딩
# - depth/img2emb/datamatrix에 설정값 주입
# - SQLite 로깅(모든 샘플 기록) — DB 오픈 가속(스키마 1회/종료 시 WAL truncate)
# - D+Enter or Arduino Signal "1": DataMatrix 스캔(빠른 4방향만) + 1초 치수 측정 + 임베딩
# - T+Enter: 학습 트리거
# - DM 카메라 persistent 공유 + 락
# - 모델: centroid + LGBM 동시 지원(파일 변경 자동 리로드, config.model.type로 추론 우선순위 선택)
import os
import sys
import time
import json
import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import serial  # 아두이노 시리얼 통신을 위한 라이브러리
import numpy as np
import cv2
import torch

# ===== 경로 세팅 =====
ROOT = Path(__file__).resolve().parentcd. cd. 
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
DM_RES           = CFG.get("datamatrix",{}).get("prefer_res", [1920,1080])
DM_FPS           = int(CFG.get("datamatrix",{}).get("prefer_fps", 6))
DM_ROIS          = CFG.get("datamatrix",{}).get("rois", None)
DM_SCAN_TIMEOUT_S = float(CFG.get("datamatrix",{}).get("scan_timeout_s", 2.0))

# 품질/저장/모델
Q_WARN  = float(CFG.get("quality",{}).get("q_warn", 0.30))
DB_PATH = ROOT / CFG.get("storage",{}).get("sqlite_path", "pack.db")
CENTROID_MODEL_PATH = ROOT / CFG.get("model",{}).get("centroids_path", "centroids.npz")
CENTROID_TOPK = int(CFG.get("model",{}).get("topk", 3))

# LGBM 모델
MODEL_TYPE = CFG.get("model", {}).get("type", "centroid").lower()  # "centroid"|"lgbm"
LGBM_MODEL_PATH = ROOT / CFG.get("model", {}).get("lgbm_path", "lgbm.pkl")

# 아두이노 시리얼 통신 설정
# 예: "/dev/ttyACM0" (리눅스), "COM3" (윈도우)
SERIAL_PORT = CFG.get("arduino", {}).get("port", None)
SERIAL_BAUDRATE = int(CFG.get("arduino", {}).get("baudrate", 9600))

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

    cfg_rois_raw = rois if (rois and isinstance(rois, list)) else DM_DEFAULT_ROIS
    rois_nm = [r for r in cfg_rois_raw if not bool(r.get("hflip", False))]
    rois_m  = [r for r in cfg_rois_raw if     bool(r.get("hflip", False))]
    cfg_rois = rois_nm + rois_m

    t0 = _t_now()
    deadline = t0 + max(0.1, float(timeout_s))
    SAFETY_MS = 40.0
    FAST_BUDGET_MS = 70.0

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
    z = np.load(path)
    C = z["C"].astype(np.float32)
    labels = z["labels"]
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True)+1e-8)
    return Cn, labels

def predict_with_centroid(x143, Cn, labels, topk=CENTROID_TOPK):
    xx = x143 / (np.linalg.norm(x143)+1e-8)
    sims = Cn @ xx
    idx = np.argsort(-sims)[:topk]
    return [(str(labels[i]), float(sims[i])) for i in idx]

# ==========================
# LGBM 모델 (신/구 포맷 자동 호환)
# ==========================
def load_lgbm_model(path=LGBM_MODEL_PATH):
    if not path.exists():
        return None, None, None
    try:
        from joblib import load
        obj = load(str(path))
        if isinstance(obj, dict) and "booster_str" in obj:
            import lightgbm as lgb
            booster = lgb.Booster(model_str=obj["booster_str"])
            classes = obj.get("classes_", None)
            best_it = obj.get("best_iteration", None)
            return booster, classes, best_it
        clf = obj.get("model", None)
        classes = obj.get("classes_", getattr(clf, "classes_", None) if clf else None)
        best_it = getattr(clf, "best_iteration_", None)
        return clf, classes, best_it
    except Exception as e:
        print("[model] lgbm load error:", e)
        return None, None, None

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
    print(f"[infer] warn: feature_dim {cur} < expected {exp} → pad zeros")
    pad = np.zeros((x.shape[0], exp - cur), dtype=x.dtype)
    return np.hstack([x, pad])

def predict_lgbm_topk(model, classes, vec143: np.ndarray, topk=3, best_iteration=None):
    x = vec143.reshape(1, -1).astype(np.float32)
    x = _ensure_feat_dim(model, x)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
    else:
        num_it = int(best_iteration) if (best_iteration is not None and int(best_iteration) > 0) else 0
        probs = model.predict(x, num_iteration=num_it)[0]

    if np.allclose(probs, probs[0], rtol=0, atol=1e-7):
        print("[infer] warn: flat probabilities from LGBM — check L2 normalization & feature-dimension match")

    if classes is None:
        classes = np.arange(len(probs))
    idx = np.argsort(-probs)[:topk]
    return [(str(classes[i]), float(probs[i])) for i in idx]

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
    conn.execute("PRAGMA mmap_size=268435456")

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
def run_training_now(config_path: Optional[Path]):
    train_py = MODEL_DIR / "train.py"
    args = [sys.executable, str(train_py)]
    if config_path is not None:
        args += ["--config", str(config_path)]
    print("[train] 시작:", " ".join(args))
    try:
        r = subprocess.run(args, check=False)
        print(f"[train] 종료 코드={r.returncode}")
    except Exception as e:
        print("[train] 실행 실패:", e)

# ==========================
# 모델 즉시 로더 유틸
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
    if CENTROID_MODEL_PATH.exists():
        global last_mtime_cent, Cn, labels
        last_mtime_cent = CENTROID_MODEL_PATH.stat().st_mtime
        Cn, labels = load_centroid_model()
        if labels is not None:
            print(f"[model] centroid 로드: {len(labels)} classes")
    if LGBM_MODEL_PATH.exists():
        global last_mtime_lgbm, lgbm_model, lgbm_classes, lgbm_best_it
        last_mtime_lgbm = LGBM_MODEL_PATH.stat().st_mtime
        lgbm_model, lgbm_classes, lgbm_best_it = load_lgbm_model()
        if lgbm_model is not None:
            print(f"[model] lgbm 로드: classes={len(lgbm_classes)} best_it={lgbm_best_it}")

    # 시리얼 포트 열기
    ser = None
    if SERIAL_PORT:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=0.01)
            time.sleep(2) # 아두이노 리셋 및 안정화 대기
            print(f"[arduino] 시리얼 포트 연결 성공: {SERIAL_PORT}")
        except Exception as e:
            print(f"[arduino] 시리얼 포트 연결 실패: {e}")
            ser = None
    
    print("[ready] total init %.2fs" % (time.time()-t_all))
    print("[hint] D+Enter:수동측정 / T+Enter:학습 / Arduino 신호 '1':수동측정")

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

            # 수동/학습 트리거 (시리얼 입력 추가)
            # 1. 키보드 입력 확인
            cmd = stdin_readline_nonblock(0.02)

            # 2. 시리얼 입력 확인 (키보드 입력이 없을 때만)
            if not cmd and ser and ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line == "1":
                        cmd = "D" # 아두이노 신호 "1"을 내부 명령어 "D"로 변환
                        print(f"[{_ts_wall()}] [arduino] 신호 '1' 수신 -> 측정 시작")
                except Exception as e:
                    print(f"[{_ts_wall()}] [arduino] 시리얼 읽기 오류: {e}")

            # 3. 명령어 처리
            if not cmd:
                continue
            
            uc = cmd.strip().upper()
            if uc == "T":
                run_training_now(CONFIG_PATH)
                # 학습 후 두 모델 모두 재로딩 시도
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

            if uc == "D":
                global DM_TRACE_ID
                DM_TRACE_ID += 1
                tid = DM_TRACE_ID
                t_press = _t_now()
                print(f"[D#{tid}][{_ts_wall()}] trigger -> scan_call")

                # 1) DataMatrix 스캔 (fast4만)
                t_s0 = _t_now()
                payload = datamatrix_scan_persistent(DM_SCAN_TIMEOUT_S, debug=DM_DEBUG, trace_id=tid)
                t_s1 = _t_now()
                print(f"[D#{tid}][{_ts_wall()}] scan_return elapsed={_ms(t_s1 - t_s0)} "
                      f"T_trigger→scan_return={_ms(t_s1 - t_press)} payload={'YES' if payload else 'NO'}")
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

                # 5) 저장 + (옵션) 추론 출력
                if payload:
                    on_sample_record(conn, feat, vec, product_id=payload, has_label=1, origin="manual_dm")
                else:
                    on_sample_record(conn, feat, vec, product_id=None, has_label=0, origin="manual_no_dm")

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

                    ran = False
                    if MODEL_TYPE == "lgbm":
                        if ensure_lgbm_loaded():
                            try:
                                topk = int(CFG.get("model", {}).get("topk", 3))
                                preds = predict_lgbm_topk(lgbm_model, lgbm_classes, full_vec, topk=topk, best_iteration=lgbm_best_it)
                                print("[infer] LGBM top-k:", preds)
                                ran = True
                            except Exception as e:
                                print("[infer] lgbm error:", e)
                        if not ran and ensure_centroid_loaded():
                            try:
                                preds = predict_with_centroid(full_vec, Cn, labels, topk=CENTROID_TOPK)
                                print("[infer] centroid top-k:", preds)
                                ran = True
                            except Exception as e:
                                print("[infer] centroid error:", e)
                    else:
                        if ensure_centroid_loaded():
                            try:
                                preds = predict_with_centroid(full_vec, Cn, labels, topk=CENTROID_TOPK)
                                print("[infer] centroid top-k:", preds)
                                ran = True
                            except Exception as e:
                                print("[infer] centroid error:", e)
                        if not ran and ensure_lgbm_loaded():
                            try:
                                topk = int(CFG.get("model", {}).get("topk", 3))
                                preds = predict_lgbm_topk(lgbm_model, lgbm_classes, full_vec, topk=topk, best_iteration=lgbm_best_it)
                                print("[infer] LGBM top-k:", preds)
                                ran = True
                            except Exception as e:
                                print("[infer] lgbm error:", e)
                    if not ran:
                        print("[infer] 모델 없음(파일 미존재 또는 로드 실패)")
                continue

    except KeyboardInterrupt:
        print("\n[exit] keyboard interrupt")
    finally:
        # 종료 시 WAL 비우기 → 다음 오픈 빨라짐
        try:
            if conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("PRAGMA optimize")
        except Exception:
            pass
        try:
            if 'depth' in locals() and depth.is_running():
                depth.stop()
        except Exception:
            pass
        try:
            if conn:
                conn.close()
        except Exception:
            pass
        try:
            close_dm_cam_persistent()
        except Exception:
            pass
        
        # 시리얼 포트 닫기
        try:
            if ser and ser.is_open:
                ser.close()
                print("[cleanup] serial port closed")
        except Exception:
            pass

        print("[cleanup] stopped")

if __name__ == "__main__":
    main()