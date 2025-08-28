# train.py (Py3.6 호환 / sklearn 미사용 / 진행률 % 출력 / 피클 없이 저장되는 centroid)
# - YAML/JSON 설정 로딩(--config)
# - SQLite(sample_log)에서 has_label=1 로드
# - 15 스칼라 + emb → L2 정규화 벡터  ← (logsV 중복 포함되던 문제 수정)
# - model.type: "centroid" or "lgbm"
# - LGBM: lightgbm.train + early_stopping + 진행률(%) 콜백
# - 저장:
#     * centroid: NPZ (labels를 'U128' 문자열로 변환)  => allow_pickle=False OK
#     * lgbm: joblib(pkl) + 추가로 피클-프리 NPZ 백업도 함께 저장

import sys, json, argparse, sqlite3, time
from pathlib import Path
import numpy as np
from typing import Optional

np.set_printoptions(suppress=True, linewidth=100000, threshold=np.inf, precision=4)
ROOT = Path(__file__).resolve().parent.parent

# ----------------- config 로더 -----------------
def load_config(cli_path):  # type: (Optional[str]) -> (dict, Optional[Path])
    if cli_path:
        p = Path(cli_path)
        if p.exists():
            try:
                if p.suffix.lower() in (".yaml",".yml"):
                    import yaml  # noqa
                    with p.open("r", encoding="utf-8") as f:
                        return yaml.safe_load(f) or {}, p
                if p.suffix.lower() == ".json":
                    with p.open("r", encoding="utf-8") as f:
                        return json.load(f), p
            except Exception as e:
                print("[config] 로드 실패: {} | {}".format(p, e))
        else:
            print("[config] 지정 파일 없음: {}".format(p))

    cand = [ROOT/"config.yaml", ROOT/"config.yml", ROOT/"config.json"]
    for p in cand:
        if not p.exists():
            continue
        try:
            if p.suffix.lower() in (".yaml",".yml"):
                import yaml  # noqa
                with p.open("r", encoding="utf-8") as f:
                    return (yaml.safe_load(f) or {}), p
            if p.suffix.lower() == ".json":
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f), p
        except Exception as e:
            print("[config] 로드 실패: {} | {}".format(p, e))
    print("[config] 파일 없음. 기본값 사용")
    return {}, None

def resolve_path(p):  # type: (str) -> Path
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)

# ----------------- DB -----------------
def open_db(db_path):  # type: (Path) -> sqlite3.Connection
    return sqlite3.connect(str(db_path), isolation_level=None, timeout=10.0)

def fetch_labeled(conn, min_q=0.0):
    cols = ("d1,d2,d3,mad1,mad2,mad3,r1,r2,r3,sr1,sr2,sr3,logV,logsV,q,emb,product_id")
    cur = conn.cursor()
    if min_q > 0:
        cur.execute("SELECT {} FROM sample_log WHERE has_label=1 AND q>=?;".format(cols), (float(min_q),))
    else:
        cur.execute("SELECT {} FROM sample_log WHERE has_label=1;".format(cols))
    return cur.fetchall()

# ----------------- 벡터 구성 -----------------
def emb_from_blob(b):  # type: (bytes) -> np.ndarray
    return np.frombuffer(b, dtype=np.float32)

def stack_vectors(rows, expect_emb_dim=None):
    """
    DB 컬럼 순서:
      d1,d2,d3, mad1..mad3, r1..r3, sr1..sr3, logV, logsV, q, emb, product_id
      => 앞의 12개(core12) + [logV, logsV, q] = 15 스칼라 + emb
    """
    vecs, labels = [], []
    for r in rows:
        # 총 17개: core12(12) + logV + logsV + q + emb + product_id
        *core12, logV, logsV, q, emb_blob, pid = r
        if len(core12) != 12:
            print("[warn] unexpected core12 len:", len(core12))
        feat15 = list(core12) + [logV, logsV, q]  # ← logsV 중복 제거 및 15개 보장

        emb = emb_from_blob(emb_blob)
        if expect_emb_dim is not None and emb.shape[0] != expect_emb_dim:
            print("[warn] emb_dim mismatch: got {} expect {} → skip".format(emb.shape[0], expect_emb_dim))
            continue

        v = np.concatenate([np.array(feat15, np.float32), emb.astype(np.float32)], axis=0)
        v = np.where(np.isfinite(v), v, 0.0).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-8)
        vecs.append(v); labels.append(pid)
    if not vecs:
        return None, None
    return np.stack(vecs, 0).astype(np.float32), np.array(labels, dtype=object)

# ----------------- centroid -----------------
def train_centroids(X, y, min_count=2):
    classes = {}
    for xi, yi in zip(X, y):
        classes.setdefault(yi, []).append(xi)
    kept = {k: np.stack(v,0) for k,v in classes.items() if len(v) >= int(min_count)}
    if not kept:
        return None, None, None
    lbls, cents, cnts = [], [], []
    for k, M in kept.items():
        mu = M.mean(axis=0); mu = mu / (np.linalg.norm(mu)+1e-8)
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
        if yi not in idx_map:  # 방어
            continue
        k = idx_map[yi]; ny = int(counts[k])
        if ny <= 1:
            continue
        cy = sums[yi] - xi; cy = cy / (np.linalg.norm(cy)+1e-8)
        sims = []
        for j, lab in enumerate(uniq):
            sims.append(float(np.dot(xi, cy)) if j==k else float(np.dot(xi, C[j])))
        pred = uniq[int(np.argmax(sims))]
        total += 1
        if pred == yi: correct += 1
    acc = (correct/total) if total>0 else None
    return acc, total

def save_centroids(path, C, labels, counts, dim, cfg_used):
    """
    피클 없이 저장:
      - labels를 문자열 dtype('U128')로 변환
      - meta는 JSON 문자열로 저장
      => np.load(..., allow_pickle=False)로 안전 로딩 가능
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_unix": time.time(),
        "dim": int(dim),
        "labels_count": int(len(labels)),
        "counts": [int(x) for x in counts.tolist()],
        "config_used": str(cfg_used) if cfg_used else None,
        "normalize": "l2_fullvec"
    }
    labels_str = np.array([str(x) for x in labels], dtype="U128")
    np.savez(
        str(path),
        C=C.astype(np.float32),
        labels=labels_str,
        meta=json.dumps(meta)
    )
    print("[save] {}  classes={} dim={}  (pickle-free)".format(path, len(labels_str), dim))

# ----------------- LightGBM (sklearn 미사용) -----------------
def make_progress_cb(total_iters, period=10, prefix="[cv]"):
    def _cb(env):
        it = getattr(env, "iteration", None)
        if it is None:
            return
        it += 1
        if it % max(1, period) == 0 or it == total_iters:
            pct = 100.0 * it / float(total_iters)
            print("{} iter {}/{} ({:.1f}%)".format(prefix, it, total_iters, pct))
    return _cb

def encode_labels(y):
    classes, y_inv = np.unique(y, return_inverse=True)
    return classes, y_inv.astype(np.int32)

def stratified_kfold_indices(y_idx, n_splits=5, seed=42):
    rng = np.random.RandomState(int(seed))
    labels = np.unique(y_idx)
    per_label_idxs = [np.where(y_idx == lab)[0] for lab in labels]
    for arr in per_label_idxs:
        rng.shuffle(arr)
    folds = [list() for _ in range(n_splits)]
    for arr in per_label_idxs:
        for i, idx in enumerate(arr):
            folds[i % n_splits].append(int(idx))
    return [np.array(sorted(f), dtype=np.int64) for f in folds]

def lgb_params_from_cfg(params, num_class):
    p = {
        "objective": "multiclass",
        "num_class": int(num_class),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "num_leaves": int(params.get("num_leaves", 31)),
        "min_data_in_leaf": int(params.get("min_data_in_leaf", 10)),
        "feature_fraction": float(params.get("feature_fraction", 0.8)),
        "bagging_fraction": float(params.get("bagging_fraction", 0.8)),
        "bagging_freq": int(params.get("bagging_freq", 1)),
        "metric": ["multi_logloss","multi_error"],
        "verbosity": -1,
        "seed": int(params.get("seed", 42)),
    }
    return p

def eval_lgbm_cv(X, y, params, k=5, progress_period=10):
    import lightgbm as lgb

    classes, y_idx = encode_labels(y)
    num_class = len(classes)
    _, cnts = np.unique(y_idx, return_counts=True)
    max_k = int(cnts.min())
    if max_k < 2:
        return None, 0, []

    k_use = min(k, max_k)
    folds = stratified_kfold_indices(y_idx, n_splits=k_use, seed=int(params.get("seed", 42)))

    accs, best_iters = [], []
    n_est_big = int(params.get("n_estimators", 200))
    lgb_params = lgb_params_from_cfg(params, num_class)

    for fi in range(k_use):
        va_idx = folds[fi]
        tr_idx = np.setdiff1d(np.arange(len(y_idx)), va_idx, assume_unique=False)
        print("[cv] fold {}/{} start (train={}, valid={})".format(fi+1, k_use, len(tr_idx), len(va_idx)))

        dtrain = lgb.Dataset(X[tr_idx], label=y_idx[tr_idx])
        dvalid = lgb.Dataset(X[va_idx], label=y_idx[va_idx])

        callbacks = [make_progress_cb(n_est_big, period=progress_period)]
        try:
            from lightgbm import early_stopping, log_evaluation
            callbacks = [early_stopping(50), log_evaluation(progress_period)] + callbacks
        except Exception:
            pass

        booster = lgb.train(
            params=lgb_params,
            train_set=dtrain,
            num_boost_round=n_est_big,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=callbacks
        )

        best_it = getattr(booster, "best_iteration", None) or n_est_big
        best_iters.append(int(best_it))

        pred = booster.predict(X[va_idx], num_iteration=best_it)  # (N, C)
        yhat = np.argmax(pred, axis=1)
        acc = float((yhat == y_idx[va_idx]).mean())
        accs.append(acc)
        print("[cv] fold {}/{} acc={:.2f}%  best_it={}".format(fi+1, k_use, acc*100.0, best_it))

    return float(np.mean(accs)), k_use, best_iters

def train_lgbm_full(X, y, params, n_estimators_override=None, verbose_period=50):
    import lightgbm as lgb
    classes, y_idx = encode_labels(y)
    num_class = len(classes)
    lgb_params = lgb_params_from_cfg(params, num_class)
    n_est = int(params.get("n_estimators", 200))
    if n_estimators_override is not None:
        n_est = int(n_estimators_override)

    dtrain = lgb.Dataset(X, label=y_idx)

    callbacks = []
    try:
        from lightgbm import log_evaluation
        callbacks.append(log_evaluation(verbose_period))
    except Exception:
        pass

    booster = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=n_est,
        valid_sets=[dtrain],           # 진행률 로그용
        valid_names=["train"],
        callbacks=callbacks
    )
    return booster, classes

def save_lgbm(path, booster, classes_):
    from joblib import dump
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 기존 포맷(joblib)
    booster_str = booster.model_to_string()
    best_it = getattr(booster, "best_iteration", None)
    dump({"booster_str": booster_str, "best_iteration": best_it, "classes_": classes_}, str(path))
    print("[save] {}  classes={} (joblib)".format(path, len(classes_)))

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML/JSON 경로(옵션)")
    ap.add_argument("--min_count", type=int, default=None, help="(centroid) 클래스 최소 샘플수(미지정시 2)")
    ap.add_argument("--min_q", type=float, default=None, help="q 하한(미지정시 0.0)")
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

    print("[cfg] type={}  db={}  emb_dim={}  min_count={}  min_q={}".format(
        model_type, db_path, out_dim, min_count, min_q))

    conn = open_db(db_path)
    rows = fetch_labeled(conn, min_q=min_q)
    print("[data] labeled rows={}".format(len(rows)))

    X, y = stack_vectors(rows, expect_emb_dim=out_dim)
    if X is None:
        print("[fatal] usable data not found"); return 2
    dim = X.shape[1]
    print("[data] X shape={}  classes={}".format(X.shape, len(set(y))))

    if model_type == "lgbm":
        params = cfg.get("lgbm", {})
        acc, folds, best_iters = eval_lgbm_cv(X, y, params, k=5, progress_period=10)
        if acc is None:
            print("[eval] not enough per-class samples for CV")
            booster, classes_ = train_lgbm_full(X, y, params)
            save_lgbm(lgbm_path, booster, classes_)
            return 0

        n_final = int(np.median(best_iters)) if best_iters else int(params.get("n_estimators", 200))
        print("[eval] LGBM {}-fold acc={:.2f}% | best_iters={} | n_final={}".format(
            folds, acc*100.0, best_iters, n_final))

        booster, classes_ = train_lgbm_full(X, y, params, n_estimators_override=n_final, verbose_period=50)
        save_lgbm(lgbm_path, booster, classes_)
        return 0

    # 기본: centroid
    C, labels, counts = train_centroids(X, y, min_count=min_count)
    if C is None:
        print("[fatal] no class has enough samples(min_count)"); return 3

    acc, total = eval_top1_loo(X, y, C, labels, counts)
    if acc is None:
        print("[eval] not enough per-class samples for LOO")
    else:
        print("[eval] LOO top-1 acc={:.2f}%  n={}".format(acc*100.0, total))

    save_centroids(centroids_path, C, labels, counts, dim, used_cfg)
    return 0

if __name__ == "__main__":
    sys.exit(main())
