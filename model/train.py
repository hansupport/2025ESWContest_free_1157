#train.py
# - YAML/JSON 설정 로딩(--config로 명시 가능)
# - SQLite(sample_log)에서 has_label=1인 샘플 로드
# - 15 스칼라 + emb(BLOB) → (15 + out_dim) 벡터 구성
# - 벡터 L2 정규화
#   * model.type == "centroid": 클래스별 평균 → L2 정규화한 centroid 저장(.npz)
#   * model.type == "lgbm"    : LightGBM 학습(.pkl), 간단한 Stratified K-Fold 평가
# - 저장 경로: model.centroids_path 또는 model.lgbm_path

import sys, json, argparse, sqlite3, time
from pathlib import Path
import numpy as np
from typing import Optional

np.set_printoptions(suppress=True, linewidth=100000, threshold=np.inf, precision=4)

# 프로젝트 루트(…/AutoPack)
ROOT = Path(__file__).resolve().parent.parent

# ----------------- config 로더 -----------------
def load_config(cli_path: Optional[str]):
    """--config가 주어지면 그 경로를, 아니면 프로젝트 루트의 config.yaml/json을 우선 탐색"""
    if cli_path:
        p = Path(cli_path)
        if not p.exists():
            print(f"[config] 지정 파일 없음: {p}")
        else:
            try:
                if p.suffix.lower() in (".yaml",".yml"):
                    import yaml
                    with p.open("r", encoding="utf-8") as f:
                        return yaml.safe_load(f) or {}, p
                if p.suffix.lower() == ".json":
                    with p.open("r", encoding="utf-8") as f:
                        return json.load(f), p
            except Exception as e:
                print(f"[config] 로드 실패: {p} | {e}")

    cand = [ROOT/"config.yaml", ROOT/"config.yml", ROOT/"config.json"]
    for p in cand:
        if not p.exists(): continue
        try:
            if p.suffix.lower() in (".yaml",".yml"):
                import yaml
                with p.open("r", encoding="utf-8") as f:
                    return (yaml.safe_load(f) or {}), p
            if p.suffix.lower() == ".json":
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f), p
        except Exception as e:
            print(f"[config] 로드 실패: {p} | {e}")
    print("[config] 파일 없음. 기본값 사용")
    return {}, None

def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)

# ----------------- DB 로딩 -----------------
def open_db(db_path: Path):
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=10.0)
    return conn

def fetch_labeled(conn, min_q=0.0):
    cols = ("d1,d2,d3,mad1,mad2,mad3,r1,r2,r3,sr1,sr2,sr3,logV,logsV,q,emb,product_id")
    cur = conn.cursor()
    if min_q > 0:
        cur.execute(f"SELECT {cols} FROM sample_log WHERE has_label=1 AND q>=?;", (float(min_q),))
    else:
        cur.execute(f"SELECT {cols} FROM sample_log WHERE has_label=1;")
    rows = cur.fetchall()
    return rows

# ----------------- 벡터 구성 -----------------
def emb_from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)

def stack_vectors(rows, expect_emb_dim=None):
    """rows → (X: [N,D], y: [N])  ; 각 벡터는 [15스칼라 + emb]를 L2 정규화"""
    vecs, labels = [], []
    for r in rows:
        *feat14, logsV, q, emb_blob, pid = r  # d1..sr3(14개), logV, q
        feat15 = list(feat14) + [logsV, q]
        emb = emb_from_blob(emb_blob)
        if expect_emb_dim is not None and emb.shape[0] != expect_emb_dim:
            print(f"[warn] emb_dim mismatch: got {emb.shape[0]} expect {expect_emb_dim} → skip")
            continue
        v = np.concatenate([np.array(feat15, np.float32), emb.astype(np.float32)], axis=0)
        v = np.where(np.isfinite(v), v, 0.0).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-8)
        vecs.append(v); labels.append(pid)
    if not vecs:
        return None, None
    return np.stack(vecs, 0).astype(np.float32), np.array(labels, dtype=object)

# ----------------- centroid 학습/평가/저장 -----------------
def train_centroids(X, y, min_count=2):
    classes = {}
    for xi, yi in zip(X, y):
        classes.setdefault(yi, []).append(xi)
    kept = {k: np.stack(v,0) for k,v in classes.items() if len(v) >= int(min_count)}
    if not kept:
        return None, None, None
    lbls, cents, cnts = [], [], []
    for k, M in kept.items():
        mu = M.mean(axis=0)
        mu = mu / (np.linalg.norm(mu)+1e-8)
        lbls.append(k); cents.append(mu.astype(np.float32)); cnts.append(M.shape[0])
    C = np.stack(cents, 0).astype(np.float32)
    labels = np.array(lbls, dtype=object)
    counts = np.array(cnts, dtype=np.int32)
    return C, labels, counts

def eval_top1_loo(X, y, C, labels, counts):
    uniq = list(labels)
    idx_map = {lab:i for i,lab in enumerate(uniq)}
    sums = {lab: np.zeros_like(X[0]) for lab in uniq}
    for xi, yi in zip(X, y):
        if yi in sums: sums[yi] += xi
    correct = 0; total = 0
    for xi, yi in zip(X, y):
        if yi not in idx_map:  # min_count 미만 제외 클래스
            continue
        k = idx_map[yi]
        ny = int(counts[k])
        if ny <= 1:
            continue
        cy = sums[yi] - xi
        cy = cy / (np.linalg.norm(cy)+1e-8)
        sims = []
        for j, lab in enumerate(uniq):
            sims.append(float(np.dot(xi, cy)) if j==k else float(np.dot(xi, C[j])))
        pred = uniq[int(np.argmax(sims))]
        total += 1
        if pred == yi:
            correct += 1
    acc = (correct/total) if total>0 else None
    return acc, total

def save_centroids(path: Path, C, labels, counts, dim, cfg_used):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_unix": time.time(),
        "dim": int(dim),
        "labels_count": int(len(labels)),
        "counts": counts.tolist(),
        "config_used": str(cfg_used) if cfg_used else None,
        "normalize": "l2_fullvec"
    }
    np.savez(str(path), C=C.astype(np.float32), labels=labels, meta=json.dumps(meta))
    print(f"[save] {path}  classes={len(labels)} dim={dim}")

# ----------------- LightGBM 학습/평가/저장 -----------------
def train_lgbm(X, y, params):
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(
        n_estimators=int(params.get("n_estimators", 200)),
        num_leaves=int(params.get("num_leaves", 31)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        min_data_in_leaf=int(params.get("min_data_in_leaf", 10)),
        feature_fraction=float(params.get("feature_fraction", 0.8)),
        bagging_fraction=float(params.get("bagging_fraction", 0.8)),
        bagging_freq=int(params.get("bagging_freq", 1)),
        class_weight=params.get("class_weight", "balanced"),
        random_state=int(params.get("seed", 42)),
        n_jobs=-1
    )
    clf.fit(X, y)
    return clf

def eval_lgbm_cv(X, y, params, k=5):
    """Stratified K-Fold 간이 평가(클래스 최소 샘플수보다 folds가 클 수 없게 조정)"""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    _, cnts = np.unique(y, return_counts=True)
    max_k = int(cnts.min())
    if max_k < 2:
        return None, 0
    k_use = min(k, max_k)
    accs = []
    skf = StratifiedKFold(n_splits=k_use, shuffle=True, random_state=int(params.get("seed", 42)))
    for tr, va in skf.split(X, y):
        clf = train_lgbm(X[tr], y[tr], params)
        pr = clf.predict(X[va])
        accs.append(accuracy_score(y[va], pr))
    return float(np.mean(accs)), len(accs)

def save_lgbm(path: Path, clf):
    from joblib import dump
    path.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": clf, "classes_": getattr(clf, "classes_", None)}, str(path))
    print(f"[save] {path}  classes={len(clf.classes_)}")

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML/JSON 경로(옵션)")
    ap.add_argument("--min_count", type=int, default=None, help="(centroid) 클래스 최소 샘플수(미지정시 config 또는 2)")
    ap.add_argument("--min_q", type=float, default=None, help="q 하한(미지정시 config 또는 0.0)")
    ap.add_argument("--type", type=str, default=None, help="모델 타입 강제: centroid|lgbm")
    args = ap.parse_args()

    cfg, used_cfg = load_config(args.config)

    model_cfg = cfg.get("model", {})
    model_type = (args.type or model_cfg.get("type", "centroid")).lower()
    out_dim = int(cfg.get("embedding", {}).get("out_dim", 128))
    db_path = resolve_path(cfg.get("storage", {}).get("sqlite_path", "pack.db"))
    centroids_path = resolve_path(model_cfg.get("centroids_path", "centroids.npz"))
    lgbm_path = resolve_path(model_cfg.get("lgbm_path", "lgbm.pkl"))
    min_count = args.min_count if args.min_count is not None else int(cfg.get("training", {}).get("min_count", 2))
    min_q = args.min_q if args.min_q is not None else float(cfg.get("training", {}).get("min_q", 0.0))

    print(f"[cfg] type={model_type}  db={db_path}  emb_dim={out_dim}  min_count={min_count}  min_q={min_q}")

    conn = open_db(db_path)
    rows = fetch_labeled(conn, min_q=min_q)
    print(f"[data] labeled rows={len(rows)}")

    X, y = stack_vectors(rows, expect_emb_dim=out_dim)
    if X is None:
        print("[fatal] usable data not found"); return 2
    dim = X.shape[1]
    print(f"[data] X shape={X.shape}  classes={len(set(y))}")

    if model_type == "lgbm":
        # LightGBM 분기
        try:
            import lightgbm  # noqa: F401
            from sklearn import model_selection  # noqa: F401
        except Exception as e:
            print("[fatal] LightGBM/Sklearn 필요: pip install lightgbm scikit-learn joblib")
            print("        error:", e)
            return 5

        params = cfg.get("lgbm", {})
        acc, folds = eval_lgbm_cv(X, y, params, k=5)
        if acc is None:
            print("[eval] not enough per-class samples for CV")
        else:
            print(f"[eval] LGBM {folds}-fold acc={acc*100:.2f}%")
        clf = train_lgbm(X, y, params)
        save_lgbm(lgbm_path, clf)
        return 0

    # 기본: centroid
    C, labels, counts = train_centroids(X, y, min_count=min_count)
    if C is None:
        print("[fatal] no class has enough samples(min_count)"); return 3

    acc, total = eval_top1_loo(X, y, C, labels, counts)
    if acc is None:
        print("[eval] not enough per-class samples for LOO")
    else:
        print(f"[eval] LOO top-1 acc={acc*100:.2f}%  n={total}")

    save_centroids(centroids_path, C, labels, counts, dim, used_cfg)
    return 0

if __name__ == "__main__":
    sys.exit(main())
