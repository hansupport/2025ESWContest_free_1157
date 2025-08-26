# main.py
# - YAML/JSON 설정 로딩: <script>.yaml|yml|json → config.yaml|json
# - depth/img2emb/datamatrix에 설정값 동적 주입(roi_px+roi_offset 통일)
# - SQLite 로깅(모든 샘플 기록)
# - D+Enter 수동 트리거(초음파 대체) 1초 측정
# - DataMatrix 유무 무관 기록, q 알림, centroids.npz 변경 감지 알림
# - T+Enter 학습 트리거(model/train.py 호출)

import os
import sys
import time
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch

# ===== 경로 세팅: model 디렉토리를 파이썬 경로에 추가 =====
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

# 로컬 모듈
import depth as depthmod
from depth import DepthEstimator
import img2emb as embmod
from datamatrix import DMatrixWatcher

np.set_printoptions(suppress=True, linewidth=100000, threshold=np.inf, precision=4)

# =========================
# Config 로딩 (파일별 적용)
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
                import yaml  # 선택 의존성
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
EMB_CAM_PIX   = CFG.get("embedding",{}).get("pixfmt", "YUYV")   # "YUYV"|"MJPG"
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
EMB_ROI_PX   = CFG.get("embedding",{}).get("roi_px", None)      # [w,h] or null
EMB_ROI_OFF  = CFG.get("embedding",{}).get("roi_offset", [0,0]) # [dx,dy]

# depth
DEPTH_W       = int(CFG.get("depth",{}).get("width", 1280))
DEPTH_H       = int(CFG.get("depth",{}).get("height", 720))
DEPTH_FPS     = int(CFG.get("depth",{}).get("fps", 6))
DEPTH_ROI_PX  = CFG.get("depth",{}).get("roi_px", [230,230])     # [w,h]
DEPTH_ROI_OFF = CFG.get("depth",{}).get("roi_offset", [20,-210]) # [dx,dy]

# depth 모듈 파라미터 오버라이드(옵션)
for k in ["DECIM","PLANE_TAU","H_MIN_BASE","H_MAX","MIN_OBJ_PIX",
          "BOTTOM_ROI_RATIO","HOLE_FILL","CORE_MARGIN_PX","P_LO","P_HI"]:
    if k in CFG.get("depth",{}):
        setattr(depthmod, k, CFG["depth"][k])
        print(f"[depth.cfg] set {k} = {CFG['depth'][k]}")

# datamatrix
DM_CAMERA     = CFG.get("datamatrix",{}).get("camera", "auto")
DM_RES        = CFG.get("datamatrix",{}).get("prefer_res", [1920,1080])
DM_FPS        = int(CFG.get("datamatrix",{}).get("prefer_fps", 6))
DM_INTERVAL   = float(CFG.get("datamatrix",{}).get("decode_interval", 0.20))
DM_LOG_EVERY  = int(CFG.get("datamatrix",{}).get("log_every_decode", 1))
DM_MAX_BACKOFF= float(CFG.get("datamatrix",{}).get("max_backoff", 1.0))
DM_ROIS       = CFG.get("datamatrix",{}).get("rois", None)  # list of {name,size,offset,hflip}

# 품질/저장/모델
Q_WARN   = float(CFG.get("quality",{}).get("q_warn", 0.30))
DB_PATH  = ROOT / CFG.get("storage",{}).get("sqlite_path", "pack.db")
CENTROID_MODEL_PATH = ROOT / CFG.get("model",{}).get("centroids_path", "centroids.npz")
CENTROID_TOPK = int(CFG.get("model",{}).get("topk", 3))

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
# img2emb 빌더/카메라
# ==========================
def apply_img2emb_roi_from_cfg():
    """img2emb에 ROI(중앙 기준)를 주입. set_center_roi가 있으면 그걸 우선 사용."""
    try:
        if EMB_ROI_PX is not None and hasattr(embmod, "set_center_roi"):
            embmod.set_center_roi(EMB_ROI_PX, EMB_ROI_OFF)
            print(f"[img2emb.cfg] ROI(center) px={EMB_ROI_PX} off={EMB_ROI_OFF}")
            return
        # 폴백: 절대 좌표 계산(해상도 기반)
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

def e2e_warmup_now(emb):
    print(f"[warmup] img2emb: e2e warmup {E2E_WARMUP_FRAMES} frames (pregrab={E2E_PREGRAB})")
    cap = open_embed_camera()
    try:
        embmod.e2e_warmup(emb, cap, n=E2E_WARMUP_FRAMES, pregrab=E2E_PREGRAB)
    finally:
        try: cap.release()
        except Exception: pass
    print("[warmup] img2emb: e2e done")

def embed_one_frame(emb, pregrab=3):
    cap = open_embed_camera()
    try:
        for _ in range(max(0, pregrab)): cap.grab()
        ok, bgr = cap.read()
        if not ok: return None
        v = emb.embed_bgr(bgr)  # (EMB_OUT_DIM,)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return v.astype(np.float32)
    finally:
        try: cap.release()
        except Exception: pass

# ==========================
# centroid 모델
# ==========================
def load_centroid_model(path=CENTROID_MODEL_PATH):
    if not path.exists(): return None, None
    z = np.load(path)
    C = z["C"].astype(np.float32)       # (n_cls, 143)
    labels = z["labels"]
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True)+1e-8)
    return Cn, labels

def predict_with_centroid(x143, Cn, labels, topk=CENTROID_TOPK):
    xx = x143 / (np.linalg.norm(x143)+1e-8)
    sims = Cn @ xx
    idx = np.argsort(-sims)[:topk]
    return [(str(labels[i]), float(sims[i])) for i in idx]

# ==========================
# SQLite 스토리지
# ==========================
def open_db(db_path: Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
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
    print(f"[db] open: {db_path}")
    return conn

def emb_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes(order="C")

def insert_sample(conn, feat15: dict, emb128: np.ndarray, product_id, has_label: int, origin: str):
    vals = (
        time.time(), product_id, int(has_label),
        float(feat15["d1"]), float(feat15["d2"]), float(feat15["d3"]),
        float(feat15["mad1"]), float(feat15["mad2"]), float(feat15["mad3"]),
        float(feat15["r1"]), float(feat15["r2"]), float(feat15["r3"]),
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
        roi_px = (230,230)
    try:
        roi_off = (int(DEPTH_ROI_OFF[0]), int(DEPTH_ROI_OFF[1]))
    except Exception:
        roi_off = (20,-210)

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

    # 임베더 준비 + e2e 워밍업
    emb = build_embedder_only()
    print(f"[img2emb.cfg] dev={EMB_CAM_DEV} {EMB_CAM_W}x{EMB_CAM_H}@{EMB_CAM_FPS} pixfmt={EMB_CAM_PIX}")
    e2e_warmup_now(emb)

    # DataMatrix watcher (다중 ROI 지원 / 구버전 폴백)
    try:
        w,h = int(DM_RES[0]), int(DM_RES[1])
    except Exception:
        w,h = 1920,1080
    try:
        watcher = DMatrixWatcher(
            camera=DM_CAMERA,
            prefer_res=(w,h),
            prefer_fps=DM_FPS,
            rois=DM_ROIS,
            decode_interval=DM_INTERVAL,
            log_every_decode=DM_LOG_EVERY,
            max_backoff=DM_MAX_BACKOFF
        )
    except TypeError:
        # 구버전 시그니처 폴백
        watcher = DMatrixWatcher(
            camera=DM_CAMERA,
            prefer_res=(w,h),
            prefer_fps=DM_FPS
        )
    watcher.start()

    # centroid 모델 변경 감지
    last_mtime = 0.0
    Cn, labels = (None, None)
    if CENTROID_MODEL_PATH.exists():
        last_mtime = CENTROID_MODEL_PATH.stat().st_mtime
        Cn, labels = load_centroid_model()
        if labels is not None:
            print(f"[model] centroid 로드: {len(labels)} classes")

    print("[ready] total init %.2fs" % (time.time()-t_all))
    print("[hint] D + Enter = 수동 측정 1초 / T + Enter = 학습 실행(model/train.py)")

    try:
        while True:
            # 모델 업데이트 감지
            if CENTROID_MODEL_PATH.exists():
                m = CENTROID_MODEL_PATH.stat().st_mtime
                if m > last_mtime:
                    last_mtime = m
                    Cn, labels = load_centroid_model()
                    if labels is not None:
                        print(f"[update] 모델 업데이트 감지: {len(labels)} classes 로드")

            # 수동 트리거(초음파 대체) / 학습 트리거
            cmd = stdin_readline_nonblock(0.05)
            if cmd:
                uc = cmd.strip().upper()
                if uc == "T":
                    if hasattr(watcher, "pause"): watcher.pause()
                    run_training_now(CONFIG_PATH)
                    if CENTROID_MODEL_PATH.exists():
                        last_mtime = CENTROID_MODEL_PATH.stat().st_mtime
                        Cn, labels = load_centroid_model()
                        if labels is not None:
                            print(f"[model] reloaded: {len(labels)} classes")
                    if hasattr(watcher, "resume"): watcher.resume()
                    continue
                if uc == "D":
                    if hasattr(watcher, "pause"): watcher.pause()
                    print("[manual] D 트리거 → 1초 측정 시작")
                    feat = depth.measure_dimensions(duration_s=1.0, n_frames=10)
                    if feat is None:
                        print("[manual] 측정 실패")
                        if hasattr(watcher, "resume"): watcher.resume()
                        continue
                    vec = embed_one_frame(emb, pregrab=3)
                    if vec is None:
                        print("[manual] 임베딩 실패")
                        if hasattr(watcher, "resume"): watcher.resume()
                        continue
                    if feat["q"] < Q_WARN:
                        print(f"[notify] 품질 경고: q={feat['q']:.2f} (임계 {Q_WARN:.2f})")

                    on_sample_record(conn, feat, vec, product_id=None, has_label=0, origin="manual")

                    meta = np.array([
                        feat["d1"], feat["d2"], feat["d3"],
                        feat["mad1"], feat["mad2"], feat["mad3"],
                        feat["r1"], feat["r2"], feat["r3"],
                        feat["sr1"], feat["sr2"], feat["sr3"],
                        feat["logV"], feat["logsV"], feat["q"]
                    ], np.float32)
                    full_vec = np.concatenate([meta, vec], axis=0)
                    if Cn is not None and labels is not None:
                        print("[infer] (print) centroid top-k:",
                              predict_with_centroid(full_vec, Cn, labels, topk=CENTROID_TOPK))
                    else:
                        print("[infer] (print) centroid 모델 없음")
                    if hasattr(watcher, "resume"): watcher.resume()
                    continue

            # DataMatrix 이벤트
            event = watcher.get_detection(timeout=0.5)
            if event is None:
                continue

            ts, payloads = event
            payload = payloads[0] if (payloads and len(payloads)>0) else None
            print(f"[event] ts={ts:.3f}, payload={payload}")
            if hasattr(watcher, "pause"): watcher.pause()

            feat = depth.measure_dimensions(duration_s=1.0, n_frames=10)
            if feat is None:
                print("[measure] 실패(물체/안정성)")
                if hasattr(watcher, "resume"): watcher.resume()
                continue

            vec = embed_one_frame(emb, pregrab=3)
            if vec is None:
                print("[embed] 실패")
                if hasattr(watcher, "resume"): watcher.resume()
                continue

            if feat["q"] < Q_WARN:
                print(f"[notify] 품질 경고: q={feat['q']:.2f} (임계 {Q_WARN:.2f})")

            # 라벨 유무와 관계없이 기록
            if payload:
                on_sample_record(conn, feat, vec, product_id=payload, has_label=1, origin="datamatrix")
            else:
                on_sample_record(conn, feat, vec, product_id=None, has_label=0, origin="datamatrix_empty")
                # 추론(print 대체)
                meta = np.array([
                    feat["d1"], feat["d2"], feat["d3"],
                    feat["mad1"], feat["mad2"], feat["mad3"],
                    feat["r1"], feat["r2"], feat["r3"],
                    feat["sr1"], feat["sr2"], feat["sr3"],
                    feat["logV"], feat["logsV"], feat["q"]
                ], np.float32)
                full_vec = np.concatenate([meta, vec], axis=0)
                if Cn is not None and labels is not None:
                    print("[infer] (print) centroid top-k:",
                          predict_with_centroid(full_vec, Cn, labels, topk=CENTROID_TOPK))
                else:
                    print("[infer] (print) centroid 모델 없음")

            if hasattr(watcher, "resume"): watcher.resume()

    except KeyboardInterrupt:
        print("[exit] keyboard interrupt")
    finally:
        try: watcher.stop()
        except Exception: pass
        try: depth.stop()
        except Exception: pass
        try: conn.close()
        except Exception: pass
        print("[cleanup] stopped")

if __name__ == "__main__":
    main()
