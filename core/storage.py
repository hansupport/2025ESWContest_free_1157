# core/storage.py
import time
import sqlite3
from pathlib import Path
import numpy as np


def open_db(db_path: Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    import time as _t
    t0 = _t.perf_counter()
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=5.0)
    t1 = _t.perf_counter()

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=67108864")  # 64MB

    ver = conn.execute("PRAGMA user_version").fetchone()[0]

    if ver == 0:
        t_schema0 = _t.perf_counter()
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
        t_schema1 = _t.perf_counter()
        print(f"[db] schema init {(t_schema1 - t_schema0)*1000:.1f} ms")

    print(f"[db] connect {(t1 - t0)*1000:.1f} ms  | total open {(_t.perf_counter()-t0)*1000:.1f} ms")
    return conn


def _emb_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes(order="C")


def insert_sample(conn, feat15: dict, emb128: np.ndarray, product_id, has_label: int, origin: str):
    vals = (
        time.time(), product_id, int(has_label),
        float(feat15["d1"]),  float(feat15["d2"]),  float(feat15["d3"]),
        float(feat15["mad1"]), float(feat15["mad2"]), float(feat15["mad3"]),
        float(feat15["r1"]),  float(feat15["r2"]),  float(feat15["r3"]),
        float(feat15["sr1"]), float(feat15["sr2"]), float(feat15["sr3"]),
        float(feat15["logV"]), float(feat15["logsV"]), float(feat15["q"]),
        _emb_to_blob(emb128), origin
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
