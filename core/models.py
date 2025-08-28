# core/models.py  (Python 3.6 호환)
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# -------- Centroid --------
def load_centroid(path: Path):
    if not path.exists():
        return None, None
    z = np.load(str(path), allow_pickle=True)
    C = z["C"].astype(np.float32)
    labels = z["labels"]
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    return Cn, labels


def predict_centroid_topk(x143: np.ndarray, Cn: np.ndarray, labels, topk: int):
    xx = x143 / (np.linalg.norm(x143) + 1e-8)
    sims = Cn @ xx  # [-1..1] 유사도
    idx = np.argsort(-sims)[:int(topk)]
    return [(str(labels[i]), float(sims[i])) for i in idx], sims


def confidence_from_gap(gap: float, scale: float = 8.0, bias: float = 0.0) -> float:
    """
    gap = s1 - s2 (상위-차상위 유사도 차)
    확신도 = sigmoid(scale*(gap - bias))
    - scale↑ → 더 가파르게
    - bias  → 기준 이동
    """
    y = 1.0 / (1.0 + np.exp(-float(scale) * (float(gap) - float(bias))))
    return float(y)


# -------- LGBM --------
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
    import json
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
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
        return clf, classes, best_it  # 주의: wrapper 반환
    except Exception:
        return None


def load_lgbm(path: Path):
    if not path.exists():
        return None, None, None
    if path.suffix.lower() == ".npz":
        triple = _try_load_lgbm_npz(path)
        if triple:
            booster_str, classes, best_it = triple
            import lightgbm as lgb
            booster = lgb.Booster(model_str=booster_str)
            return booster, np.array(classes), best_it
    if path.suffix.lower() == ".json":
        triple = _try_load_lgbm_json(path)
        if triple:
            booster_str, classes, best_it = triple
            import lightgbm as lgb
            booster = lgb.Booster(model_str=booster_str)
            return booster, np.array(classes), best_it
    jl = _try_load_lgbm_joblib(path)
    if jl:
        if isinstance(jl[0], str):
            import lightgbm as lgb
            booster = lgb.Booster(model_str=jl[0])
            return booster, np.array(jl[1]), jl[2]
        # sklearn wrapper
        return jl[0], np.array(jl[1]) if jl[1] is not None else None, jl[2]

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


def _lgbm_expected_dim(model):
    for attr in ("n_features_", "n_features_in_"):
        if hasattr(model, attr):
            try:
                v = int(getattr(model, attr))  # sklearn
                if v > 0:
                    return v
            except Exception:
                pass
    if hasattr(model, "booster_"):
        try:
            return int(model.booster_.num_feature())  # sklearn wrapper
        except Exception:
            pass
    if hasattr(model, "num_feature"):  # Booster
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
        print("[infer] warn: feature_dim %d > expected %d → slice" % (cur, exp))
        return x[:, :exp]
    # cur < exp
    print("[infer] warn: feature_dim %d < expected %d → pad zeros" % (cur, exp))
    pad = np.zeros((x.shape[0], exp - cur), dtype=x.dtype)
    return np.hstack([x, pad])


def lgbm_predict_proba(model, classes, x143: np.ndarray, best_iteration=None) -> np.ndarray:
    x = x143.reshape(1, -1).astype(np.float32)
    x = _ensure_feat_dim(model, x)
    if hasattr(model, "predict_proba"):  # sklearn wrapper
        probs = model.predict_proba(x)[0]
    else:  # Booster
        num_it = int(best_iteration) if (best_iteration is not None and int(best_iteration) > 0) else None
        probs = model.predict(x, num_iteration=num_it)[0]
    return np.asarray(probs, dtype=np.float32)


# -------- High-level Inference Engine --------
class InferenceEngine(object):
    """
    settings.model.type 순서로 추론:
      - "centroid": centroid → lgbm 폴백
      - "lgbm":     lgbm → centroid 폴백
    centroid 확신도는 gap-based sigmoid를 사용.
    """
    def __init__(self, settings):
        self.S = settings
        self.Cn = None
        self.labels = None
        self.m_cent = 0.0
        self.lgbm = None
        self.lgbm_classes = None
        self.lgbm_best_it = None
        self.m_lgbm = 0.0
        self._load_all()

    def _load_all(self):
        # centroid
        p_cent = Path(self.S.paths.centroids)
        if p_cent.exists():
            self.m_cent = p_cent.stat().st_mtime
            self.Cn, self.labels = load_centroid(p_cent)
            if self.labels is not None:
                print("[model] centroid 로드: %d classes" % (len(self.labels),))
        # lgbm
        p_lgbm = Path(self.S.paths.lgbm)
        if p_lgbm.exists():
            self.m_lgbm = p_lgbm.stat().st_mtime
            self.lgbm, self.lgbm_classes, self.lgbm_best_it = load_lgbm(p_lgbm)
            if (self.lgbm is not None) and (self.lgbm_classes is not None):
                print("[model] lgbm 로드: classes=%d best_it=%s" % (len(self.lgbm_classes), str(self.lgbm_best_it)))

    def reload_if_updated(self):
        p_cent = Path(self.S.paths.centroids)
        if p_cent.exists():
            m = p_cent.stat().st_mtime
            if m > self.m_cent:
                self.m_cent = m
                self.Cn, self.labels = load_centroid(p_cent)
                if self.labels is not None:
                    print("[update] centroid 업데이트: %d classes 로드" % (len(self.labels),))
        p_lgbm = Path(self.S.paths.lgbm)
        if p_lgbm.exists():
            m = p_lgbm.stat().st_mtime
            if m > self.m_lgbm:
                self.m_lgbm = m
                self.lgbm, self.lgbm_classes, self.lgbm_best_it = load_lgbm(p_lgbm)
                if self.lgbm is not None:
                    print("[update] lgbm 업데이트: classes=%d best_it=%s" % (len(self.lgbm_classes), str(self.lgbm_best_it)))

    def _infer_centroid(self, x_norm) -> Optional[Tuple[str, float, float]]:
        if self.Cn is None or self.labels is None:
            return None
        topk = int(self.S.model.topk)
        preds, sims = predict_centroid_topk(x_norm, self.Cn, self.labels, topk)
        # gap = s1 - s2
        s_sorted = np.sort(sims)[::-1]
        gap = float(s_sorted[0] - (s_sorted[1] if len(s_sorted) > 1 else 0.0))
        conf = confidence_from_gap(
            gap,
            scale=float(self.S.model.centroid_margin_scale),
            bias=float(self.S.model.centroid_margin_bias),
        )
        top_lab = preds[0][0]
        return top_lab, conf, gap

    def _infer_lgbm(self, x_norm) -> Optional[Tuple[str, float, float]]:
        if self.lgbm is None or self.lgbm_classes is None:
            return None
        probs = lgbm_predict_proba(self.lgbm, self.lgbm_classes, x_norm, self.lgbm_best_it)
        if np.allclose(probs, probs[0], rtol=0, atol=1e-7):
            print("[infer] warn: flat probabilities from LGBM — check L2 normalization & feature-dimension match")
        idx = int(np.argmax(probs))
        p1 = float(probs[idx])
        p2 = float(np.sort(probs)[-2]) if probs.size >= 2 else 0.0
        return str(self.lgbm_classes[idx]), p1, (p1 - p2)

    def infer(self, x_norm: np.ndarray) -> Tuple[Optional[str], float, float, Optional[str]]:
        """
        반환: (top_label, top_prob, gap, backend)
        backend: "centroid" | "lgbm" | None
        """
        if (self.S.model.type or "centroid") == "lgbm":
            r = self._infer_lgbm(x_norm)
            if r is not None:
                lab, p, gap = r
                return lab, p, gap, "lgbm"
            r2 = self._infer_centroid(x_norm)
            if r2 is not None:
                lab, p, gap = r2
                return lab, p, gap, "centroid"
            return None, 0.0, 0.0, None
        else:
            r = self._infer_centroid(x_norm)
            if r is not None:
                lab, p, gap = r
                return lab, p, gap, "centroid"
            r2 = self._infer_lgbm(x_norm)
            if r2 is not None:
                lab, p, gap = r2
                return lab, p, gap, "lgbm"
            return None, 0.0, 0.0, None
