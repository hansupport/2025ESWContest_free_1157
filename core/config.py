# core/config.py
import json
from pathlib import Path
from types import SimpleNamespace

def _sn(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _sn(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_sn(x) for x in d]
    return d

def load_config():
    """
    우선순위:
      1) main.yaml / main.yml / main.json
      2) config.yaml / config.json
    """
    root = Path(__file__).resolve().parents[1]
    stem = "main"
    cand = [
        root / f"{stem}.yaml", root / f"{stem}.yml", root / f"{stem}.json",
        root / "config.yaml", root / "config.json"
    ]
    cfg, used = {}, None
    for p in cand:
        if not p.exists():
            continue
        try:
            if p.suffix.lower() in (".yaml", ".yml"):
                import yaml
                cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                used = p; break
            if p.suffix.lower() == ".json":
                cfg = json.loads(p.read_text(encoding="utf-8"))
                used = p; break
        except Exception as e:
            print(f"[config] 로드 실패: {p} | {e}")
    if used is None:
        print("[config] 파일 없음. 기본값 사용")
    else:
        print(f"[config] 사용 파일: {used}")
    return cfg, used

def get_settings(CFG: dict):
    root = Path(__file__).resolve().parents[1]

    emb = CFG.get("embedding", {})
    depth = CFG.get("depth", {})
    dm = CFG.get("datamatrix", {})
    quality = CFG.get("quality", {})
    storage = CFG.get("storage", {})
    model = CFG.get("model", {})

    # 기본값들
    S = {
        "paths": {
            "root": str(root),
            "db": str(root / storage.get("sqlite_path", "pack.db")),
            "centroids": str(root / model.get("centroids_path", "centroids.npz")),
            "lgbm": str(root / model.get("lgbm_path", "lgbm.npz")),
        },
        "embedding": {
            "cam_dev": emb.get("cam_dev", "/dev/video2"),
            "pixfmt": emb.get("pixfmt", "YUYV"),
            "width": int(emb.get("width", 848)),
            "height": int(emb.get("height", 480)),
            "fps": int(emb.get("fps", 6)),
            "input_size": int(emb.get("input_size", 128)),
            "out_dim": int(emb.get("out_dim", 128)),
            "width_scale": float(emb.get("width_scale", 0.35)),
            "fp16": bool(emb.get("fp16", False)),  # ONNX CPU에선 무시
            "use_depthwise": bool(emb.get("use_depthwise", False)),
            "use_bn": bool(emb.get("use_bn", False)),
            "pinned": bool(emb.get("pinned", False)),  # ONNX CPU에선 무시
            "roi_px": emb.get("roi_px", None),
            "roi_offset": emb.get("roi_offset", [0, 0]),
            "weights_path": str(emb.get("weights_path", "model/weights/tinymnet_emb_128d_w035.onnx")),
            "e2e_warmup_frames": int(emb.get("e2e_warmup_frames", 12)),
            "e2e_pregrab": int(emb.get("e2e_pregrab", 2)),
        },
        "depth": {
            "width": int(depth.get("width", 1280)),
            "height": int(depth.get("height", 720)),
            "fps": int(depth.get("fps", 6)),
            "roi_px": depth.get("roi_px", [260, 260]),
            "roi_offset": depth.get("roi_offset", [20, -100]),
            "overrides": depth
        },
        "dm": {
            "camera": dm.get("camera", 2),
            "prefer_res": dm.get("prefer_res", [1920, 1080]),
            "prefer_fps": int(dm.get("prefer_fps", 6)),
            "rois": dm.get("rois", None),
            "scan_timeout_s": float(dm.get("scan_timeout_s", 2.0))
        },
        "quality": {
            "q_warn": float(quality.get("q_warn", 0.30)),
        },
        "model": {
            "type": model.get("type", "centroid").lower(),
            "topk": int(model.get("topk", 3)),
            "prob_threshold": float(model.get("top1_threshold", 0.40)),
            "smooth_window": int(model.get("smooth_window", 3)),
            "smooth_min": int(model.get("smooth_min_votes", max(2, int(round(int(model.get("smooth_window", 3))*0.6))))),
            "min_margin": float(model.get("min_margin", 0.02)),
            "centroid_margin_scale": float(model.get("centroid_margin_scale", 1500.0)),
        }
    }
    return _sn(S)

def apply_depth_overrides(depthmod, S):
    # depth 섹션의 특정 키가 있으면 depth 모듈 상수 덮어쓰기
    keys = ["DECIM","PLANE_TAU","H_MIN_BASE","H_MAX","MIN_OBJ_PIX",
            "BOTTOM_ROI_RATIO","HOLE_FILL","CORE_MARGIN_PX","P_LO","P_HI"]
    d = getattr(S.depth, "overrides", {})  # SimpleNamespace 또는 dict
    if d is None:
        return
    # SimpleNamespace → dict 로 통일
    if not isinstance(d, dict):
        try:
            d = vars(d)
        except Exception:
            d = {}
    for k in keys:
        if k in d:
            try:
                setattr(depthmod, k, d[k])
                print(f"[depth.cfg] set {k} = {d[k]}")
            except Exception as e:
                print(f"[depth.cfg] set {k} 실패: {e}")
