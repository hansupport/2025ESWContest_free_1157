# core/config.py  (Python 3.6 compatible)
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple, Optional

def _as_ns(d):
    """dict/list → dot-access SimpleNamespace (재귀)."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _as_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_as_ns(x) for x in d]
    return d

def load_config():
    # type: () -> Tuple[dict, Optional[Path]]
    """
    우선순위:
      1) <script_stem>.yaml / .yml / .json
      2) config.yaml / config.json
    """
    import sys
    stem = Path(sys.argv[0]).resolve().stem
    root = Path(sys.argv[0]).resolve().parent
    cand = [
        root / "{}.yaml".format(stem), root / "{}.yml".format(stem), root / "{}.json".format(stem),
        root / "config.yaml",          root / "config.json"
    ]
    cfg, used = {}, None
    for p in cand:
        if not p.exists():
            continue
        try:
            suf = p.suffix.lower()
            if suf in (".yaml", ".yml"):
                try:
                    import yaml  # optional
                except Exception as e:
                    print("[config] PyYAML 미설치 또는 오류:", e)
                    continue
                text = p.read_text(encoding="utf-8")
                cfg = (yaml.safe_load(text) or {})
                used = p
                break
            if suf == ".json":
                cfg = json.loads(p.read_text(encoding="utf-8"))
                used = p
                break
        except Exception as e:
            print("[config] 로드 실패: {} | {}".format(p, e))
    if used is None:
        print("[config] 파일 없음. 기본값 사용")
    else:
        print("[config] 사용 파일: {}".format(used))
    return cfg, used

def get_settings(cfg):
    # type: (dict) -> SimpleNamespace
    ROOT = Path(__file__).resolve().parents[1]  # project root

    emb = cfg.get("embedding", {}) or {}
    EMB = {
        "cam_dev":     emb.get("cam_dev", "/dev/video2"),
        "pixfmt":      emb.get("pixfmt", "YUYV"),
        "width":       int(emb.get("width", 848)),
        "height":      int(emb.get("height", 480)),
        "fps":         int(emb.get("fps", 6)),
        "input_size":  int(emb.get("input_size", 128)),
        "out_dim":     int(emb.get("out_dim", 128)),
        "width_scale": float(emb.get("width_scale", 0.35)),
        "fp16":        bool(emb.get("fp16", False)),
        "use_depthwise": bool(emb.get("use_depthwise", False)),
        "use_bn":      bool(emb.get("use_bn", False)),
        "pinned":      bool(emb.get("pinned", False)),
        "weights_path": emb.get("weights_path", None),
        "e2e_warmup_frames": int(emb.get("e2e_warmup_frames", 60)),
        "e2e_pregrab": int(emb.get("e2e_pregrab", 8)),
        "roi_px":      emb.get("roi_px", None),
        "roi_offset":  emb.get("roi_offset", [0, 0]),
    }

    dep = cfg.get("depth", {}) or {}
    DEP = {
        "width":  int(dep.get("width", 1280)),
        "height": int(dep.get("height", 720)),
        "fps":    int(dep.get("fps", 6)),
        "roi_px": dep.get("roi_px", [260, 260]),
        "roi_offset": dep.get("roi_offset", [20, -100]),
        "overrides": {k: dep[k] for k in [
            "DECIM","PLANE_TAU","H_MIN_BASE","H_MAX","MIN_OBJ_PIX",
            "BOTTOM_ROI_RATIO","HOLE_FILL","CORE_MARGIN_PX","P_LO","P_HI"
        ] if k in dep}
    }

    dm = cfg.get("datamatrix", {}) or {}
    DM = {
        "camera": dm.get("camera", 2),
        "prefer_res": dm.get("prefer_res", [1920, 1080]),
        "prefer_fps": int(dm.get("prefer_fps", 6)),
        "rois": dm.get("rois", None),
        "scan_timeout_s": float(dm.get("scan_timeout_s", 2.0)),
    }

    mdl = cfg.get("model", {}) or {}
    MODEL = {
        "type": (mdl.get("type", "centroid") or "centroid").lower(),
        "topk": int(mdl.get("topk", 3)),
        "prob_threshold": float(mdl.get("prob_threshold", 0.40)),
        "smooth_window": int(mdl.get("smooth_window", 5)),
        "smooth_min": int(mdl.get("smooth_min", 3)),
        "min_margin": float(mdl.get("min_margin", 0.02)),
        "centroid_margin_scale": float(mdl.get("centroid_margin_scale", 8.0)),
        "centroid_margin_bias": float(mdl.get("centroid_margin_bias", 0.0)),
        "centroids_path": mdl.get("centroids_path", "centroids.npz"),
        "lgbm_path": mdl.get("lgbm_path", "lgbm.npz"),
    }

    storage = cfg.get("storage", {}) or {}
    STG = {
        "sqlite_path": storage.get("sqlite_path", "pack.db"),
    }
    qual = cfg.get("quality", {}) or {}
    Q = {"q_warn": float(qual.get("q_warn", 0.30))}
    dbg = cfg.get("debug", {}) or {}
    DEBUG = {"datamatrix": bool(dbg.get("datamatrix", True))}

    PATHS = {
        "root": ROOT,
        "db":   (ROOT / STG["sqlite_path"]).resolve(),
        "centroids": (ROOT / MODEL["centroids_path"]).resolve(),
        "lgbm": (ROOT / MODEL["lgbm_path"]).resolve(),
        "model_dir": (ROOT / "model").resolve(),
    }
    PATHS["weights"] = (ROOT / EMB["weights_path"]).resolve() if EMB["weights_path"] else None

    # 스무딩 파라미터 보정
    sw = max(3, min(5, MODEL["smooth_window"]))
    sm = max(3, min(sw, MODEL["smooth_min"]))
    MODEL["smooth_window"] = sw
    MODEL["smooth_min"] = sm

    return _as_ns({
        "embedding": EMB,
        "depth":     DEP,
        "dm":        DM,
        "model":     MODEL,
        "storage":   STG,
        "quality":   Q,
        "debug":     DEBUG,
        "paths":     PATHS,
    })

def apply_depth_overrides(depth_module, settings):
    """depth 모듈 상수 오버라이드. settings.depth.overrides 가 dict/Namespace 모두 안전 처리."""
    try:
        overrides = getattr(settings.depth, "overrides", {}) or {}
    except Exception:
        overrides = {}

    # dict vs SimpleNamespace 모두 대응
    try:
        items = overrides.items() if hasattr(overrides, "items") else vars(overrides).items()
    except Exception:
        items = []

    for k, v in items:
        try:
            setattr(depth_module, k, v)
            print("[depth.cfg] set {} = {}".format(k, v))
        except Exception as e:
            print("[depth.cfg] set {} 실패: {}".format(k, e))

