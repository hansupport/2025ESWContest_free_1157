# core/config.py
import os
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
            # 프로젝트 구조에 맞춰 기본값을 pkl로 조정
            "lgbm": str(root / model.get("lgbm_path", "lgbm.pkl")),
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

            # ==== 3-뷰 concat 임베딩 관련 신규 옵션들 ====
            # config.yaml에 있으면 그대로 반영, 없으면 apply_embedding_overrides에서 디폴트/ENV로 채움
            "concat3": bool(emb.get("concat3", False)),
            "mirror_period": int(emb.get("mirror_period", 3)),
            "center_shrink": float(emb.get("center_shrink", 0.0)),
            "mirror_shrink": float(emb.get("mirror_shrink", 0.08)),
            "rois3": emb.get("rois3", None),  # [[w,h,dx,dy,hflip], ...] 최대 3개
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

# ===== 신규: embedding 옵션 오버레이(3뷰 concat) =====
def apply_embedding_overrides(S):
    """
    S.embedding에 3-뷰(concat) 관련 기본값/환경변수/설정값을 주입.
    config.yaml에 없거나 일부만 있어도 안전하게 동작하도록 보강.
    """
    if not hasattr(S, 'embedding') or S.embedding is None:
        S.embedding = SimpleNamespace()

    E = S.embedding

    # 필수 기본값(기존 값이 없으면 채움)
    if not hasattr(E, 'input_size') or E.input_size is None:
        E.input_size = 224
    if not hasattr(E, 'out_dim') or E.out_dim is None:
        E.out_dim = 128
    if not hasattr(E, 'roi_px') or E.roi_px is None:
        E.roi_px = [540, 540]
    if not hasattr(E, 'roi_offset') or E.roi_offset is None:
        E.roi_offset = [103, -260]

    # ----- 새 옵션 주입 -----
    # 켜기/끄기 (ENV가 있으면 우선)
    concat3_env = os.getenv("EMB_CONCAT3", None)
    if concat3_env is not None:
        try:
            E.concat3 = bool(int(concat3_env))
        except Exception:
            E.concat3 = True
    else:
        if not hasattr(E, 'concat3'):
            E.concat3 = False
        else:
            E.concat3 = bool(E.concat3)

    # 미러뷰 최신화용 프레임 스텝
    try:
        E.mirror_period = int(os.getenv("EMB_MIRROR_PERIOD", str(getattr(E, 'mirror_period', 3))))
    except Exception:
        E.mirror_period = 3

    # ROI 테두리 노이즈 잘라내기 (0.0~0.4 권장)
    try:
        E.center_shrink = float(os.getenv("EMB_CENTER_SHRINK", str(getattr(E, 'center_shrink', 0.0))))
    except Exception:
        E.center_shrink = 0.0
    try:
        E.mirror_shrink = float(os.getenv("EMB_MIRROR_SHRINK", str(getattr(E, 'mirror_shrink', 0.08))))
    except Exception:
        E.mirror_shrink = 0.08

    # 3뷰 ROI (w,h,dx,dy,hflip) 리스트
    default_rois3 = [
        [540, 540, +103, -260, 0],  # center
        [240, 460, -380, -170, 1],  # left mirror  (hflip)
        [270, 440, +630, -170, 1],  # right mirror (hflip)
    ]
    if not hasattr(E, 'rois3') or not E.rois3:
        E.rois3 = default_rois3
    # 타입 정리(문자→정수 캐스팅, 길이 3까지만)
    try:
        cleaned = []
        for item in E.rois3[:3]:
            w,h,dx,dy,hf = item
            cleaned.append([int(w), int(h), int(dx), int(dy), int(hf)])
        if len(cleaned) < 3:
            # 부족하면 기본으로 채우기
            cleaned = (cleaned + default_rois3)[:3]
        E.rois3 = cleaned
    except Exception:
        E.rois3 = default_rois3
