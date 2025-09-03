import os
import json
from pathlib import Path
from types import SimpleNamespace

# dict/list를 재귀적으로 SimpleNamespace로 변환
def _sn(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _sn(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_sn(x) for x in d]
    return d

# 설정 파일 탐색/로드: main.* 우선, 없으면 config.*
def load_config():
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

# CFG에 기본값을 병합해 실행 설정 S를 만들고, 경로를 루트 기준 절대 경로로 확정
def get_settings(CFG: dict):
    root = Path(__file__).resolve().parents[1]

    emb = CFG.get("embedding", {})
    depth = CFG.get("depth", {})
    dm = CFG.get("datamatrix", {})
    quality = CFG.get("quality", {})
    storage = CFG.get("storage", {})
    model = CFG.get("model", {})

    S = {
        # 경로
        "paths": {
            "root": str(root),
            "db": str(root / storage.get("sqlite_path", "pack.db")),
            "centroids": str(root / model.get("centroids_path", "centroids.npz")),
            "lgbm": str(root / model.get("lgbm_path", "lgbm.pkl")),
        },
        # 임베딩
        "embedding": {
            "cam_dev": emb.get("cam_dev", "/dev/video2"),
            "pixfmt": emb.get("pixfmt", "YUYV"),
            "width": int(emb.get("width", 848)),
            "height": int(emb.get("height", 480)),
            "fps": int(emb.get("fps", 6)),
            "input_size": int(emb.get("input_size", 128)),
            "out_dim": int(emb.get("out_dim", 128)),
            "width_scale": float(emb.get("width_scale", 0.35)),
            "fp16": bool(emb.get("fp16", False)),
            "use_depthwise": bool(emb.get("use_depthwise", False)),
            "use_bn": bool(emb.get("use_bn", False)),
            "pinned": bool(emb.get("pinned", False)),
            "roi_px": emb.get("roi_px", None),
            "roi_offset": emb.get("roi_offset", [0, 0]),
            "weights_path": str(emb.get("weights_path", "model/weights/tinymnet_emb_128d_w035.onnx")),
            "e2e_warmup_frames": int(emb.get("e2e_warmup_frames", 12)),
            "e2e_pregrab": int(emb.get("e2e_pregrab", 2)),

            "concat3": bool(emb.get("concat3", False)),
            "mirror_period": int(emb.get("mirror_period", 3)),
            "center_shrink": float(emb.get("center_shrink", 0.0)),
            "mirror_shrink": float(emb.get("mirror_shrink", 0.08)),
            "rois3": emb.get("rois3", None),
        },
        # Depth 카메라
        "depth": {
            "width": int(depth.get("width", 1280)),
            "height": int(depth.get("height", 720)),
            "fps": int(depth.get("fps", 6)),
            "roi_px": depth.get("roi_px", [260, 260]),
            "roi_offset": depth.get("roi_offset", [20, -100]),
            "overrides": depth
        },
        # DATAMATRIX 스캔
        "dm": {
            "camera": dm.get("camera", 2),
            "prefer_res": dm.get("prefer_res", [1920, 1080]),
            "prefer_fps": int(dm.get("prefer_fps", 6)),
            "rois": dm.get("rois", None),
            "scan_timeout_s": float(dm.get("scan_timeout_s", 2.0))
        },
        # 품질 임계
        "quality": {
            "q_warn": float(quality.get("q_warn", 0.30)),
        },
        # 추론/모델
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

# S.depth.overrides에 지정된 키가 있으면 depth 모듈 상수를 런타임에 덮어씀
def apply_depth_overrides(depthmod, S):
    keys = ["DECIM","PLANE_TAU","H_MIN_BASE","H_MAX","MIN_OBJ_PIX",
            "BOTTOM_ROI_RATIO","HOLE_FILL","CORE_MARGIN_PX","P_LO","P_HI"]
    d = getattr(S.depth, "overrides", {})  # SimpleNamespace 또는 dict
    if d is None:
        return
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

# S.embedding에 임베딩 관련 기본값/ 환경 변수/ 설정값을 주입하고 타입 정리
def apply_embedding_overrides(S):
    if not hasattr(S, 'embedding') or S.embedding is None:
        S.embedding = SimpleNamespace()

    E = S.embedding

    if not hasattr(E, 'input_size') or E.input_size is None:
        E.input_size = 224
    if not hasattr(E, 'out_dim') or E.out_dim is None:
        E.out_dim = 128
    if not hasattr(E, 'roi_px') or E.roi_px is None:
        E.roi_px = [540, 540]
    if not hasattr(E, 'roi_offset') or E.roi_offset is None:
        E.roi_offset = [103, -260]

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

    # 거울 ROI 최신화용 프레임 스텝
    try:
        E.mirror_period = int(os.getenv("EMB_MIRROR_PERIOD", str(getattr(E, 'mirror_period', 3))))
    except Exception:
        E.mirror_period = 3

    # ROI 테두리 노이즈 잘라내기
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
            w, h, dx, dy, hf = item
            cleaned.append([int(w), int(h), int(dx), int(dy), int(hf)])
        if len(cleaned) < 3:
            cleaned = (cleaned + default_rois3)[:3]
        E.rois3 = cleaned
    except Exception:
        E.rois3 = default_rois3
