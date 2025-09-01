# core/lite.py
import os
import threading, time
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2

from .utils import t_now, ms, l2_normalize, same_device

# ======== DataMatrix 래퍼 ========
class DM:
    _persist = None
    _lock = threading.Lock()

    @staticmethod
    def lock():
        return DM._lock

    @staticmethod
    def open_persistent(camera, prefer_res, prefer_fps):
        """
        datamatrix.open_camera() 래핑.
        """
        if DM._persist is not None:
            return DM._persist
        try:
            from datamatrix import open_camera as dm_open
            from datamatrix import read_frame_nonblocking as dm_read

            DM._persist = ("generic", SimpleNamespace(
                open=True, read=dm_read, cap=dm_open(camera, tuple(prefer_res), int(prefer_fps))
            ))
            # prewarm
            t0 = time.time(); read = 0
            while time.time() - t0 < 0.6:
                with DM._lock:
                    f = dm_read(DM._persist[1].cap)
                if f is not None: read += 1
                time.sleep(0.005)
            print(f"[dm.persist] opened {camera} and prewarmed, frames={read}")
        except Exception as e:
            DM._persist = None
            print("[dm.persist] open failed:", e)
        return DM._persist

    @staticmethod
    
    def close_persistent():
        if DM._persist is None:
            return
        try:
            DM._persist[1].cap.release()
        except Exception:
            pass
        DM._persist = None
        print("[dm.persist] closed")

    @staticmethod
    def scan_fast4(handle, rois, timeout_s: float, debug: bool=False, trace_id: Optional[int]=None):
        """
        datamatrix.decode_payloads_fast4() 기반의 빠른 스캔. (0/90/180/270)
        """
        try:
            from datamatrix import (
                read_frame_nonblocking as dm_read,
                crop_roi_center as dm_crop_roi,
                decode_payloads_fast4 as dm_decode_fast4,
            )
            # 기본 ROI: datamatrix.DEFAULT_ROIS
            try:
                from datamatrix import DEFAULT_ROIS as DM_DEFAULT_ROIS
            except Exception:
                DM_DEFAULT_ROIS = [
                    dict(name="ROI1", size=[260, 370], offset=[-380, 100], hflip=True),
                    dict(name="ROI2", size=[300, 400], offset=[ 610, 110], hflip=True),
                    dict(name="ROI3", size=[480, 340], offset=[ 120,  70], hflip=False),
                ]
            cfg_rois_raw = rois if (rois and isinstance(rois, list)) else DM_DEFAULT_ROIS
            rois_nm = [r for r in cfg_rois_raw if not bool(r.get("hflip", False))]
            rois_m  = [r for r in cfg_rois_raw if     bool(r.get("hflip", False))]
            cfg_rois = rois_nm + rois_m

            if handle is None:
                if debug: print("[dm.scan] handle=None")
                return None

            tag = f"D#{trace_id}" if trace_id is not None else "D"
            t0 = t_now()
            deadline = t0 + max(0.1, float(timeout_s))
            SAFETY_MS = 40.0
            FAST_BUDGET_MS = 70.0

            frames = 0
            reads_null = 0
            while True:
                now = t_now()
                if now >= deadline: break
                left_ms = (deadline - now) * 1000.0
                if left_ms < (FAST_BUDGET_MS + SAFETY_MS):
                    break

                with DM._lock:
                    frame = dm_read(handle[1].cap)
                if frame is None:
                    reads_null += 1
                    time.sleep(0.005)
                    continue

                frames += 1
                for r in cfg_rois:
                    now = t_now()
                    left_ms = (deadline - now) * 1000.0
                    if left_ms < (FAST_BUDGET_MS + SAFETY_MS):
                        return None

                    (rw, rh) = r.get("size",[0,0])
                    (dx, dy) = r.get("offset",[0,0])
                    hflip = bool(r.get("hflip", False))

                    roi = dm_crop_roi(frame, int(rw), int(rh), int(dx), int(dy))
                    if roi.size == 0: continue
                    if hflip: roi = cv2.flip(roi, 1)
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    fast_budget = int(min(FAST_BUDGET_MS, max(5.0, (deadline - t_now())*1000.0 - SAFETY_MS)))
                    res = dm_decode_fast4(gray, max_count=3, time_budget_ms=fast_budget)
                    if res:
                        return res[0]
                time.sleep(0.003)
            return None
        except Exception as e:
            print("[dm.scan] error:", e)
            return None

# ======== Embedding (ONNX CPU) ========
class ONNXEmbedder:
    def __init__(self, onnx_path: str, input_size: int, out_dim: int,
                 roi_px: Optional[Tuple[int,int]], roi_off: Tuple[int,int]):
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError("onnxruntime가 필요합니다. pip install onnxruntime") from e

        self.size = int(input_size)
        self.out_dim = int(out_dim)
        self.roi_px = tuple(roi_px) if roi_px is not None else None
        self.roi_off = (int(roi_off[0]), int(roi_off[1])) if roi_off is not None else (0,0)

        sess_opt = ort.SessionOptions()
        # Jetson Nano: 과도한 스레드는 오히려 느려질 수 있음 → 기본 1/1
        sess_opt.intra_op_num_threads = int(os.getenv("ORT_INTRA_THREADS", "1"))
        sess_opt.inter_op_num_threads = int(os.getenv("ORT_INTER_THREADS", "1"))
        # 그래프 최적화 활성화
        try:
            sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        except Exception:
            pass

        self.sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opt, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

        self.mean = np.array([0.485,0.456,0.406], np.float32)
        self.std  = np.array([0.229,0.224,0.225], np.float32)

    def _compute_center_roi(self, img):
        if self.roi_px is None:
            return img
        H, W = img.shape[:2]
        rw, rh = int(self.roi_px[0]), int(self.roi_px[1])
        dx, dy = self.roi_off
        rw = max(1, min(rw, W)); rh = max(1, min(rh, H))
        cx = W//2 + dx; cy = H//2 + dy
        x = max(0, min(W - rw, cx - rw//2))
        y = max(0, min(H - rh, cy - rh//2))
        return img[y:y+rh, x:x+rw]

    def _preprocess(self, bgr):
        img = self._compute_center_roi(bgr)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (img.shape[1], img.shape[0]) != (self.size, self.size):
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2,0,1))  # CHW
        return np.expand_dims(img, 0).astype(np.float32)  # [1,3,H,W]

    def embed_bgr(self, bgr: np.ndarray) -> np.ndarray:
        X = self._preprocess(bgr)
        y = self.sess.run([self.output_name], {self.input_name: X})[0]
        v = y[0].astype(np.float32)
        return l2_normalize(v)

class Emb:
    @staticmethod
    def build_embedder(S):
        # S.embedding.weights_path는 ONNX여야 함
        p = Path(S.embedding.weights_path)
        if not p.exists():
            raise RuntimeError(f"[img2emb] ONNX 가중치가 없습니다: {p}")
        return ONNXEmbedder(
            str(p), S.embedding.input_size, S.embedding.out_dim,
            S.embedding.roi_px, S.embedding.roi_offset
        )

    @staticmethod
    def _open_camera(cam, w=None, h=None, fps=None, pixfmt="YUYV"):
        # 간단 V4L2 오픈 (필요시 GStreamer로 확장)
        cam_id = cam
        if isinstance(cam, str):
            if cam.isdigit():
                cam_id = int(cam)
            elif cam.startswith("/dev/video"):
                try:
                    cam_id = int(cam.replace("/dev/video", ""))
                except Exception:
                    pass
        cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        if w: cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        if h: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
        if fps: cap.set(cv2.CAP_PROP_FPS, int(fps))
        if pixfmt:
            fourcc = cv2.VideoWriter_fourcc(*pixfmt.upper())
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        ret, _ = cap.read()
        if not (cap.isOpened() and ret):
            cap.release()
            raise RuntimeError(f"카메라 열기 실패: {cam}")
        return cap

    @staticmethod
    def warmup_shared(emb, S, dm_handle, lock, frames=12, pregrab=2):
        if same_device(S.dm.camera, S.embedding.cam_dev) and (dm_handle is not None):
            print(f"[warmup] img2emb: shared e2e warmup {frames} frames (pregrab={pregrab}) via DM persistent cam")
            try:
                from datamatrix import read_frame_nonblocking as dm_read
                with lock:
                    for _ in range(max(0, pregrab)):
                        _ = dm_read(dm_handle[1].cap)
                t0 = time.time(); n_ok = 0
                for _ in range(max(1, frames)):
                    with lock:
                        bgr = dm_read(dm_handle[1].cap)
                    if bgr is None:
                        time.sleep(0.003); continue
                    _ = emb.embed_bgr(bgr); n_ok += 1
                print(f"[warmup] img2emb: shared e2e done, ok_frames={n_ok}, elapsed={time.time()-t0:.2f}s")
                return
            except Exception as e:
                print("[warmup] shared path failed:", e)

        print("[warmup] img2emb: separate device warmup path")
        cap = Emb._open_camera(S.embedding.cam_dev, w=S.embedding.width, h=S.embedding.height,
                               fps=S.embedding.fps, pixfmt=S.embedding.pixfmt or "YUYV")
        try:
            t0 = time.time(); n_ok = 0
            for _ in range(max(0, pregrab)):
                cap.grab()
            for _ in range(max(1, frames)):
                ok, bgr = cap.read()
                if not ok: break
                _ = emb.embed_bgr(bgr); n_ok += 1
            print(f"[warmup] img2emb: e2e done, ok_frames={n_ok}, elapsed={time.time()-t0:.2f}s")
        finally:
            cap.release()

    @staticmethod
    def embed_one_frame_shared(emb, S, dm_handle, lock, pregrab=3):
        if same_device(S.dm.camera, S.embedding.cam_dev) and (dm_handle is not None):
            try:
                from datamatrix import read_frame_nonblocking as dm_read
                with lock:
                    for _ in range(max(0, pregrab)):
                        _ = dm_read(dm_handle[1].cap)
                    bgr = dm_read(dm_handle[1].cap)
                if bgr is None:
                    return None
                return emb.embed_bgr(bgr).astype(np.float32)
            except Exception as e:
                print("[embed] shared path error:", e)
                return None
        # separate camera
        try:
            cap = Emb._open_camera(S.embedding.cam_dev, w=S.embedding.width, h=S.embedding.height,
                                   fps=S.embedding.fps, pixfmt=S.embedding.pixfmt or "YUYV")
            try:
                for _ in range(max(0, pregrab)): cap.grab()
                ok, bgr = cap.read()
                if not ok: return None
                return emb.embed_bgr(bgr).astype(np.float32)
            finally:
                cap.release()
        except Exception as e:
            print("[embed] separate device error:", e)
            return None

    # ==============================
    # 3-View Concat 임베딩 (center + mirror L/R)
    # ==============================
    @staticmethod
    def _crop_center_roi(img: np.ndarray, rw: int, rh: int, dx: int, dy: int,
                         hflip: bool=False, shrink: float=0.0) -> np.ndarray:
        """
        이미지 중심 기준 (rw,rh), (dx,dy) 오프셋 ROI 크롭.
        hflip=True면 좌우반전 보정.
        shrink>0이면 ROI 내부를 shrink 비율만큼 가장자리 잘라 중심 재크롭.
        """
        H, W = img.shape[:2]
        rw = max(1, min(int(rw), W))
        rh = max(1, min(int(rh), H))
        cx = W // 2 + int(dx)
        cy = H // 2 + int(dy)
        x = max(0, min(W - rw, cx - rw // 2))
        y = max(0, min(H - rh, cy - rh // 2))
        roi = img[y:y+rh, x:x+rw]
        if roi.size == 0:
            return roi
        if hflip:
            roi = cv2.flip(roi, 1)
        if shrink and (shrink > 1e-6):
            h, w = roi.shape[:2]
            sw = int(max(0, min(w//2 - 1, round(w * shrink * 0.5))))
            sh = int(max(0, min(h//2 - 1, round(h * shrink * 0.5))))
            if sw > 0 or sh > 0:
                roi = roi[sh:h-sh, sw:w-sw] if (h - 2*sh > 1 and w - 2*sw > 1) else roi
        return roi

    @staticmethod
    def _embed_views_from_frame(emb, frame_bgr: np.ndarray,
                                rois3: List[Tuple[int,int,int,int,int]],
                                center_shrink: float, mirror_shrink: float) -> Optional[np.ndarray]:
        """
        단일 프레임에서 전달된 ROI 목록만 임베딩 후 concat.
        rois3: [(w,h,dx,dy,hflip), ...]  ← 길이 1도 허용
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        if not rois3:
            return None  # <<<< 기본 3개 강제 제거(버그 원인). 호출자가 원하는 ROI만 처리.

        out_vecs = []
        for i, (w,h,dx,dy,hf) in enumerate(rois3):
            shrink = center_shrink if i == 0 else mirror_shrink
            roi = Emb._crop_center_roi(frame_bgr, int(w), int(h), int(dx), int(dy), bool(hf), float(shrink))
            if roi is None or roi.size == 0:
                continue
            v = emb.embed_bgr(roi)
            if v is None:
                continue
            out_vecs.append(v.astype(np.float32))

        if not out_vecs:
            return None
        vcat = np.concatenate(out_vecs, axis=0).astype(np.float32)
        return l2_normalize(vcat)

    @staticmethod
    def embed_one_frame_shared_concat3(emb, S, dm_handle, lock,
                                       pregrab: int=12,
                                       mirror_period: int=3,          # 사용 안함(한 프레임에서 3ROI)
                                       mirror_shrink: float=0.08,
                                       center_shrink: float=0.0,
                                       rois3: Optional[List[Tuple[int,int,int,int,int]]]=None) -> Optional[np.ndarray]:
        """
        최신 프레임 1장을 읽어 [center, left-mirror, right-mirror] 3개 ROI를 모두 잘라
        각 ROI를 ONNX에 넣고 concat(3*128=384) → L2 normalize 해서 반환.
        """
        # 기본 rois3
        if (not rois3) or len(rois3) < 3:
            rois3 = [
                (540, 540, +103, -260, 0),  # center
                (240, 460, -380, -170, 1),  # left mirror
                (270, 440, +630, -170, 1),  # right mirror
            ]

        def _read_nonnull_frame_dm(dm_read, cap, tries=8, sleep_s=0.003):
            for _ in range(max(1, int(tries))):
                frm = dm_read(cap)
                if frm is not None:
                    return frm
                time.sleep(float(sleep_s))
            return None

        # DM 퍼시스턴트(공유) 경로
        if same_device(S.dm.camera, S.embedding.cam_dev) and (dm_handle is not None):
            try:
                from datamatrix import read_frame_nonblocking as dm_read
            except Exception as e:
                print("[embed.concat3] datamatrix import 실패:", e)
                return None
            try:
                with lock:
                    # flush
                    for _ in range(max(0, int(pregrab))):
                        _ = dm_read(dm_handle[1].cap)
                    # 최신 프레임 1장 확보
                    bgr = _read_nonnull_frame_dm(dm_read, dm_handle[1].cap, tries=8, sleep_s=0.003)
                if bgr is None:
                    return None

                vcat = Emb._embed_views_from_frame(
                    emb, bgr, rois3=rois3,
                    center_shrink=float(center_shrink),
                    mirror_shrink=float(mirror_shrink)
                )
                if vcat is not None:
                    # 진단 로그(최초 몇 번만 찍힘)
                    print(f"[embed.concat3] one-frame 3ROI → emb_len={vcat.shape[0]}")
                return vcat
            except Exception as e:
                print("[embed.concat3] shared path error:", e)
                return None

        # 별도 카메라 경로
        try:
            cap = Emb._open_camera(S.embedding.cam_dev, w=S.embedding.width, h=S.embedding.height,
                                   fps=S.embedding.fps, pixfmt=S.embedding.pixfmt or "YUYV")
        except Exception as e:
            print("[embed.concat3] separate device open error:", e)
            return None

        try:
            for _ in range(max(0, int(pregrab))):
                cap.grab()
            ok, bgr = cap.read()
            if not ok:
                return None

            vcat = Emb._embed_views_from_frame(
                emb, bgr, rois3=rois3,
                center_shrink=float(center_shrink),
                mirror_shrink=float(mirror_shrink)
            )
            if vcat is not None:
                print(f"[embed.concat3] one-frame 3ROI (separate) → emb_len={vcat.shape[0]}")
            return vcat
        except Exception as e:
            print("[embed.concat3] separate device error:", e)
            return None
        finally:
            try:
                cap.release()
            except Exception:
                pass

# ======== 모델/스토리지 ========
class Storage:
    @staticmethod
    def open_db(db_path: str):
        import sqlite3, time
        dbp = Path(db_path); dbp.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        conn = sqlite3.connect(str(dbp), isolation_level=None, timeout=5.0)
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

    @staticmethod
    def emb_to_blob(vec: np.ndarray) -> bytes:
        return np.asarray(vec, dtype=np.float32).tobytes(order="C")

    @staticmethod
    def on_sample_record(conn, feat15: dict, emb128: np.ndarray, product_id, has_label: int, origin: str):
        vals = (
            time.time(), product_id, int(has_label),
            float(feat15["d1"]),  float(feat15["d2"]),  float(feat15["d3"]),
            float(feat15["mad1"]), float(feat15["mad2"]), float(feat15["mad3"]),
            float(feat15["r1"]),  float(feat15["r2"]),  float(feat15["r3"]),
            float(feat15["sr1"]), float(feat15["sr2"]), float(feat15["sr3"]),
            float(feat15["logV"]), float(feat15["logsV"]), float(feat15["q"]),
            Storage.emb_to_blob(emb128), origin
        )
        conn.execute("""
        INSERT INTO sample_log(
          ts_unix, product_id, has_label,
          d1,d2,d3,mad1,mad2,mad3,r1,r2,r3,sr1,sr2,sr3,logV,logsV,q,
          emb, origin
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, vals)
        # 요약 출력
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

class Models:
    class ProbSmoother:
        def __init__(self, window=3, min_votes=2):
            from collections import deque
            self.window = int(window)
            self.min_votes = int(min_votes)
            self.buf = []  # [(label, prob)]
        def push(self, label, prob):
            self.buf.append((str(label), float(prob)))
            if len(self.buf) > self.window:
                self.buf.pop(0)
        def status(self):
            from collections import Counter
            cnt = Counter([lab for lab, _ in self.buf])
            if not cnt: return None, 0, 0.0
            top_lab, votes = cnt.most_common(1)[0]
            avg_p = float(np.mean([p for lab, p in self.buf if lab == top_lab]))
            return top_lab, votes, avg_p
        def maybe_decide(self, threshold=0.40):
            if len(self.buf) < self.window: return None
            lab, votes, avg_p = self.status()
            if votes >= self.min_votes and avg_p >= threshold:
                self.buf.clear()
                return (lab, avg_p)
            return None

    class InferenceEngine:
        def __init__(self, S):
            self.S = S
            self.cent_path = Path(S.paths.centroids)
            self.lgbm_path = Path(S.paths.lgbm)
            self.last_mtime_cent = 0.0
            self.last_mtime_lgbm = 0.0
            self.Cn, self.labels = (None, None)
            self.lgbm_model, self.lgbm_classes, self.lgbm_best_it = (None, None, None)
            self._load_initial()

        def _load_initial(self):
            if self.cent_path.exists():
                self.last_mtime_cent = self.cent_path.stat().st_mtime
                self.Cn, self.labels = self._load_centroid()
                if self.labels is not None:
                    print(f"[model] centroid 로드: {len(self.labels)} classes")
            if self.lgbm_path.exists():
                self.last_mtime_lgbm = self.lgbm_path.stat().st_mtime
                self.lgbm_model, self.lgbm_classes, self.lgbm_best_it = self._load_lgbm()
                if self.lgbm_model is not None:
                    print(f"[model] lgbm 로드: classes={len(self.lgbm_classes)} best_it={self.lgbm_best_it}")

        def reload_if_updated(self):
            if self.cent_path.exists():
                m = self.cent_path.stat().st_mtime
                if m > self.last_mtime_cent:
                    self.last_mtime_cent = m
                    self.Cn, self.labels = self._load_centroid()
                    if self.labels is not None:
                        print(f"[update] centroid 업데이트: {len(self.labels)} classes 로드")
            if self.lgbm_path.exists():
                m2 = self.lgbm_path.stat().st_mtime
                if m2 > self.last_mtime_lgbm:
                    self.last_mtime_lgbm = m2
                    self.lgbm_model, self.lgbm_classes, self.lgbm_best_it = self._load_lgbm()
                    if self.lgbm_model is not None:
                        print(f"[update] lgbm 업데이트: classes={len(self.lgbm_classes)} best_it={self.lgbm_best_it}")

        def _load_centroid(self):
            if not self.cent_path.exists(): return None, None
            z = np.load(str(self.cent_path), allow_pickle=True)
            C = z["C"].astype(np.float32)
            labels = z["labels"]
            Cn = C / (np.linalg.norm(C, axis=1, keepdims=True)+1e-8)
            return Cn, labels

        def _try_load_lgbm_npz(self, path: Path):
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

        def _try_load_lgbm_json(self, path: Path):
            try:
                import json
                obj = json.loads(path.read_text(encoding="utf-8"))
                booster_str = obj.get("booster_str", None)
                best_it = obj.get("best_iteration", None)
                classes = obj.get("classes_", None)
                return booster_str, classes, best_it
            except Exception:
                return None

        def _try_load_lgbm_joblib(self, path: Path):
            try:
                from joblib import load
                obj = load(str(path))
                if isinstance(obj, dict) and "booster_str" in obj:
                    return obj["booster_str"], obj.get("classes_", None), obj.get("best_iteration", None)
                clf = obj.get("model", None) if isinstance(obj, dict) else None
                classes = obj.get("classes_", getattr(clf, "classes_", None) if clf else None) if isinstance(obj, dict) else None
                best_it = getattr(clf, "best_iteration_", None) if clf else None
                return clf, classes, best_it
            except Exception:
                return None

        def _load_lgbm(self):
            p = self.lgbm_path
            if not p.exists(): return None, None, None
            # NPZ
            if p.suffix.lower() == ".npz":
                triple = self._try_load_lgbm_npz(p)
                if triple:
                    booster_str, classes, best_it = triple
                    import lightgbm as lgb
                    booster = lgb.Booster(model_str=booster_str)
                    return booster, np.array(classes), best_it
            # JSON
            if p.suffix.lower() == ".json":
                triple = self._try_load_lgbm_json(p)
                if triple:
                    booster_str, classes, best_it = triple
                    import lightgbm as lgb
                    booster = lgb.Booster(model_str=booster_str)
                    return booster, np.array(classes), best_it
            # Joblib
            jl = self._try_load_lgbm_joblib(p)
            if jl:
                if isinstance(jl[0], str):
                    import lightgbm as lgb
                    booster = lgb.Booster(model_str=jl[0])
                    return booster, np.array(jl[1]), jl[2]
                # sklearn wrapper 구 포맷
                return jl[0], np.array(jl[1]) if jl[1] is not None else None, jl[2]
            # text 폴백
            try:
                txt = p.read_text(encoding="utf-8")
                if txt.strip().startswith("tree"):
                    import lightgbm as lgb
                    booster = lgb.Booster(model_str=txt)
                    return booster, None, None
            except Exception:
                pass
            print("[model] lgbm load failed (unsupported format)")
            return None, None, None

        def _lgbm_expected_dim(self, model):
            for attr in ("n_features_", "n_features_in_"):
                if hasattr(model, attr):
                    try:
                        v = int(getattr(model, attr))
                        if v > 0: return v
                    except Exception:
                        pass
            if hasattr(model, "booster_"):
                try: return int(model.booster_.num_feature())
                except Exception: pass
            if hasattr(model, "num_feature"):
                try: return int(model.num_feature())
                except Exception: pass
            return None

        def _ensure_feat_dim(self, model, x: np.ndarray) -> np.ndarray:
            exp = self._lgbm_expected_dim(model)
            if exp is None: return x
            cur = x.shape[1]
            if cur == exp: return x
            if cur > exp:
                print(f"[infer] warn: feature_dim {cur} > expected {exp} → slice")
                return x[:, :exp]
            pad = np.zeros((x.shape[0], exp - cur), dtype=x.dtype)
            return np.hstack([x, pad])

        def _lgbm_predict_proba(self, model, classes, vec143: np.ndarray, best_iteration=None) -> np.ndarray:
            x = vec143.reshape(1, -1).astype(np.float32)
            x = self._ensure_feat_dim(model, x)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x)[0]
            else:
                num_it = int(best_iteration) if (best_iteration is not None and int(best_iteration) > 0) else None
                probs = model.predict(x, num_iteration=num_it)[0]
            return np.asarray(probs, dtype=np.float32)

        def _centroid_predict(self, xx: np.ndarray, topk: int):
            assert self.Cn is not None and self.labels is not None
            sims = self.Cn @ xx
            idx = np.argsort(-sims)[:topk]
            return [(str(self.labels[i]), float(sims[i])) for i in idx]

        def _centroid_conf_from_topk(self, preds, scale: float):
            if not preds:
                return "", 0.0, 0.0, 0.0
            arr = sorted(preds, key=lambda x: x[1], reverse=True)
            lab1, s1 = arr[0]
            s2 = arr[1][1] if len(arr) > 1 else -1.0
            margin = float(s1 - s2)
            # two-class view: conf = sigmoid(scale * margin)
            conf = float(1.0 / (1.0 + np.exp(-scale * margin))) if len(arr) > 1 else 1.0
            return str(lab1), conf, margin, s2

        def infer(self, full_vec_143: np.ndarray):
            """
            반환: (top_lab, top_p, gap, backend_str)  or backend=None
            """
            S = self.S
            ran = False
            top_lab, top_p, gap, backend = None, 0.0, 0.0, None

            if S.model.type == "lgbm":
                # LGBM 우선
                if self.lgbm_model is not None and self.lgbm_classes is not None:
                    try:
                        probs = self._lgbm_predict_proba(self.lgbm_model, self.lgbm_classes, full_vec_143, best_iteration=self.lgbm_best_it)
                        idx = np.argsort(-probs)
                        top_lab = str(self.lgbm_classes[idx[0]])
                        p1 = float(probs[idx[0]])
                        p2 = float(probs[idx[1]]) if len(idx) > 1 else 0.0
                        top_p = p1; gap = p1 - p2; backend = "LGBM"; ran = True
                    except Exception as e:
                        print("[infer] lgbm error:", e)
                # 폴백: centroid
                if not ran and self.Cn is not None and self.labels is not None:
                    try:
                        preds = self._centroid_predict(full_vec_143, topk=int(S.model.topk))
                        lab, conf, margin, s2 = self._centroid_conf_from_topk(preds, scale=float(S.model.centroid_margin_scale))
                        top_lab, top_p, gap, backend = lab, conf, 2.0*conf - 1.0, "centroid"; ran = True
                    except Exception as e:
                        print("[infer] centroid error:", e)
            else:
                # centroid 우선
                if self.Cn is not None and self.labels is not None:
                    try:
                        preds = self._centroid_predict(full_vec_143, topk=int(S.model.topk))
                        lab, conf, margin, s2 = self._centroid_conf_from_topk(preds, scale=float(S.model.centroid_margin_scale))
                        top_lab, top_p, gap, backend = lab, conf, 2.0*conf - 1.0, "centroid"; ran = True
                    except Exception as e:
                        print("[infer] centroid error:", e)
                # 폴백: LGBM
                if not ran and self.lgbm_model is not None and self.lgbm_classes is not None:
                    try:
                        probs = self._lgbm_predict_proba(self.lgbm_model, self.lgbm_classes, full_vec_143, best_iteration=self.lgbm_best_it)
                        idx = np.argsort(-probs)
                        top_lab = str(self.lgbm_classes[idx[0]])
                        p1 = float(probs[idx[0]])
                        p2 = float(probs[idx[1]]) if len(idx) > 1 else 0.0
                        top_p = p1; gap = p1 - p2; backend = "LGBM"; ran = True
                    except Exception as e:
                        print("[infer] lgbm error:", e)

            if not ran or top_lab is None:
                return None, 0.0, 0.0, None
            return top_lab, top_p, gap, backend

# 내부 helper
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
