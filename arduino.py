# main.py (아두이노 시리얼 신호 연동 버전)
# - 기존 기능 모두 동일
# - 아두이노에서 "1\n" 신호 수신 시 'D' 키 입력과 동일하게 동작

import sys
import time
import subprocess
import threading
import json
import os
import numpy as np
from pathlib import Path
from typing import Optional
import cv2

# ---- Serial (아두이노 연동용) ----
HAVE_SERIAL = True
try:
    import serial
except ImportError:
    HAVE_SERIAL = False

# ---- Flask (선택적) ----
HAVE_FLASK = True
try:
    from flask import Flask, jsonify, request, make_response, send_from_directory
except Exception:
    HAVE_FLASK = False

# ---- Waitress (선택적) ----
HAVE_WAITRESS = True
try:
    from waitress import serve as _waitress_serve
except Exception:
    HAVE_WAITRESS = False

# ---- 경로 세팅 ----
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

# ---- 로컬 모듈 ----
from depth import DepthEstimator
import depth as depthmod

from core.config import load_config, get_settings, apply_depth_overrides
from core.lite import DM, Emb, Models, Storage
from core.utils import (
    ts_wall, t_now, ms, stdin_readline_nonblock,
    maybe_run_jetson_perf, warmup_opencv_kernels,
    l2_normalize, same_device
)

np.set_printoptions(suppress=True, linewidth=200, threshold=50, precision=4)

# ---- 타입 매핑 ----
TYPE_MAP = {
    "000000": "도자기 컵",
    "000001": "플라스틱 컵",
    "000003": "투명 플라스틱 용기",
    "000004": "나무 통통통 사후르",
}
def type_name_from_id(tid: str) -> Optional[str]:
    if not tid:
        return None
    return TYPE_MAP.get(tid) or f"Unknown({tid})"

# ---- UI 공유 상태 ----
_UI_LOCK = threading.Lock()
_UI_STATE = {
    "type_id": None,
    "type_name": None,
    "W": None, "L": None, "H": None,      # mm
    "bubble_mm": None,
    "p": None,                             # top-1 prob
    "status": None,                        # "analyzing" | "done" | "paused" | "error" | None
    "updated_ts": None,
    "warn_msg": None,                      # p<0.40 경고
    "params": {
        "layers": 2,
        "overlap_mm": 120,
        "slack_ratio": 0.03,
        "round_to_mm": 10
    }
}

def _round_up(x, base):
    if base <= 0:
        return int(x)
    import math
    return int(math.ceil(float(x) / float(base)) * base)

def compute_bubble_length_mm(W, L, H,
                             layers=None, overlap_mm=None,
                             slack_ratio=None, round_to_mm=None):
    with _UI_LOCK:
        p = _UI_STATE["params"].copy()
    layers      = p["layers"]      if layers      is None else layers
    overlap_mm  = p["overlap_mm"]  if overlap_mm  is None else overlap_mm
    slack_ratio = p["slack_ratio"] if slack_ratio is None else slack_ratio
    round_to_mm = p["round_to_mm"] if round_to_mm is None else round_to_mm

    perimeter = 2.0 * (float(W) + float(L))
    base = perimeter * float(layers) + float(overlap_mm)
    total = base * (1.0 + float(slack_ratio))
    return _round_up(total, round_to_mm)

def _touch_ts():
    with _UI_LOCK:
        _UI_STATE["updated_ts"] = int(time.time())

def _set_ui_type(tid: Optional[str]):
    with _UI_LOCK:
        _UI_STATE["type_id"] = tid
        _UI_STATE["type_name"] = type_name_from_id(tid) if tid else None
        _UI_STATE["updated_ts"] = int(time.time())

def _set_ui_dims_and_bubble(w_mm, l_mm, h_mm):
    bubble_mm = compute_bubble_length_mm(w_mm, l_mm, h_mm)
    with _UI_LOCK:
        _UI_STATE["W"] = int(round(w_mm))
        _UI_STATE["L"] = int(round(l_mm))
        _UI_STATE["H"] = int(round(h_mm))
        _UI_STATE["bubble_mm"] = int(bubble_mm)
        _UI_STATE["updated_ts"] = int(time.time())

def _set_warn(msg=None):
    with _UI_LOCK:
        _UI_STATE["warn_msg"] = msg if msg else None
        _UI_STATE["updated_ts"] = int(time.time())

def _set_prob(pval, touch=True):
    with _UI_LOCK:
        try:
            _UI_STATE["p"] = None if pval is None else float(pval)
        except Exception:
            _UI_STATE["p"] = None
        if touch:
            _UI_STATE["updated_ts"] = int(time.time())

def _set_status(s: Optional[str], touch=True):
    with _UI_LOCK:
        _UI_STATE["status"] = s
        if touch:
            _UI_STATE["updated_ts"] = int(time.time())

# ---- 16자리 라벨 파싱 ----
def parse_dm_label_16(s):
    """
    [0:6]=type_id, [6:9]=W, [9:12]=L, [12:15]=H, [15]=material_id
    """
    if s is None:
        return None
    s = str(s).strip()
    if len(s) != 16 or (not s.isdigit()):
        return None
    try:
        return {
            "type_id":     s[0:6],
            "W":           int(s[6:9]),
            "L":           int(s[9:12]),
            "H":           int(s[12:15]),
            "material_id": int(s[15]),
            "raw":         s,
        }
    except Exception:
        return None

def _set_ui_from_label(s):
    info = parse_dm_label_16(s)
    if info is None:
        return False
    _set_ui_type(info["type_id"])
    _set_ui_dims_and_bubble(info["W"], info["L"], info["H"])
    with _UI_LOCK:
        bb = _UI_STATE['bubble_mm']
        tname = _UI_STATE['type_name']
    print(f"[ui] LABEL 사용: type_id={info['type_id']}({tname}) W={info['W']} L={info['L']} H={info['H']} mm → bubble={bb} mm")
    return True

# ---- Flask 앱 ----
_SilentWSGIRequestHandler = None
if HAVE_FLASK:
    import logging
    try:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
    except Exception:
        pass
    try:
        from werkzeug.serving import WSGIRequestHandler
        class _SilentWSGIRequestHandler(WSGIRequestHandler):
            def log(self, *args, **kwargs):
                pass
    except Exception:
        _SilentWSGIRequestHandler = None

    _APP = Flask(__name__)
    IMG_DIR = ROOT / "imgfile"

    @_APP.route("/")
    def home():
        return send_from_directory(str(ROOT), "index.html")

    @_APP.route("/ui.css")
    def serve_css():
        return send_from_directory(str(ROOT), "ui.css")

    @_APP.route("/ui.js")
    def serve_js():
        return send_from_directory(str(ROOT), "ui.js")

    @_APP.route("/api/state")
    def api_state():
        with _UI_LOCK:
            payload = json.dumps(_UI_STATE)
        resp = make_response(payload)
        resp.mimetype = "application/json"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    @_APP.route("/api/params", methods=["GET", "POST"])
    def api_params():
        data = {}
        if request.method == "POST":
            try:
                data = request.get_json(force=True) or {}
            except Exception:
                data = {}
        data.update(request.args.to_dict())

        def _to_float(x, default=None):
            try:
                if x is None: return default
                return float(x)
            except Exception:
                return default
        def _to_int(x, default=None):
            try:
                if x is None: return default
                return int(float(x))
            except Exception:
                return default

        with _UI_LOCK:
            p = _UI_STATE["params"]
            L = _to_int(data.get("layers"), p["layers"])
            O = _to_int(data.get("overlap_mm"), p["overlap_mm"])
            S_ = _to_float(data.get("slack_ratio"), p["slack_ratio"])
            R = _to_int(data.get("round_to_mm"), p["round_to_mm"])
            p.update({"layers":L, "overlap_mm":O, "slack_ratio":S_, "round_to_mm":R})
            res = {"ok": True, "params": p}
        _touch_ts()
        return jsonify(res)

    @_APP.route("/healthz")
    def healthz():
        return jsonify(ok=True, ts=int(time.time()))

    @_APP.route("/imgfile/<path:filename>")
    def serve_imgfile(filename):
        try:
            return send_from_directory(str(IMG_DIR), filename)
        except Exception:
            return ("", 404)

def _start_web_ui():
    if not HAVE_FLASK:
        print("[ui] Flask가 설치되어 있지 않아 UI를 비활성화합니다. (pip install Flask)")
        return
    host = os.getenv("UI_HOST", "0.0.0.0")
    port = int(os.getenv("UI_PORT", "8000"))
    threads = int(os.getenv("UI_THREADS", "2"))
    def _run():
        if HAVE_WAITRESS:
            print(f"[ui] Serving with Waitress (threads={threads}) on {host}:{port}")
            _waitress_serve(_APP, host=host, port=port, threads=threads, ident=None)
        else:
            print(f"[ui] Serving with Flask dev server on {host}:{port}")
            kwargs = dict(host=host, port=port, threaded=True, debug=False, use_reloader=False)
            try:
                if _SilentWSGIRequestHandler is not None:
                    kwargs["request_handler"] = _SilentWSGIRequestHandler
            except Exception:
                pass
            _APP.run(**kwargs)
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    print(f"[ui] 웹 UI 실행: http://<Jetson_IP>:{port}  (iPad/Safari 가능)")

def run_training_now(config_path: Optional[Path], force_type: Optional[str]):
    train_py = MODEL_DIR / "model" / "train.py" if (MODEL_DIR / "model" / "train.py").exists() else MODEL_DIR / "train.py"
    args = [sys.executable, str(train_py)]
    if config_path is not None:
        args += ["--config", str(config_path)]
    if force_type in ("lgbm", "centroid"):
        args += ["--type", force_type]
    print("[train] 시작:", " ".join(args))
    try:
        r = subprocess.run(args, check=False)
        print("[train] 종료 코드={}".format(r.returncode))
    except Exception as e:
        print("[train] 실행 실패:", e)

def main():
    t_all = time.time()
    print("[init] start")

    # 설정/경로
    CFG, CONFIG_PATH = load_config()
    S = get_settings(CFG)

    # 호환 셋업
    if not hasattr(S, 'dm') and hasattr(S, 'datamatrix'):
        S.dm = S.datamatrix

    class _NS: pass
    if not hasattr(S, 'paths'): S.paths = _NS()
    if not hasattr(S.paths, 'db'):
        S.paths.db = getattr(getattr(S, 'storage', _NS()), 'sqlite_path', 'pack.db')
    if not hasattr(S.paths, 'centroids'):
        S.paths.centroids = getattr(getattr(S, 'model', _NS()), 'centroids_path', 'centroids.npz')
    if not hasattr(S.paths, 'lgbm'):
        S.paths.lgbm = getattr(getattr(S, 'model', _NS()), 'lgbm_path', 'lgbm.pkl')

    if hasattr(S, 'model'):
        if not hasattr(S.model, 'min_margin'):      S.model.min_margin = 0.05
        if not hasattr(S.model, 'prob_threshold'):  S.model.prob_threshold = 0.40
        if not hasattr(S.model, 'smooth_window'):   S.model.smooth_window = 3
        if not hasattr(S.model, 'smooth_min'):      S.model.smooth_min = 2
        if not hasattr(S.model, 'topk'):            S.model.topk = 3
        if not hasattr(S.model, 'type'):            S.model.type = "lgbm"

    # UI 시작
    _start_web_ui()

    # ★★★★★ [수정 1] 아두이노 시리얼 포트 설정 ★★★★★
    ser = None
    if HAVE_SERIAL:
        # Jetson Nano에 연결된 아두이노의 포트 이름을 확인하고 맞게 수정하세요.
        # 터미널에서 `ls /dev/tty*` 명령어로 확인할 수 있습니다. (보통 /dev/ttyACM0)
        SERIAL_PORT = '/dev/ttyACM0'
        BAUD_RATE = 9600 # 아두이노 스케치에 설정된 값과 일치해야 합니다.
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01) # non-blocking read
            print(f"[serial] 아두이노 연결 성공: {SERIAL_PORT} (Baud: {BAUD_RATE})")
        except Exception as e:
            print(f"[serial] 아두이노 연결 실패: {e}")
            print("[serial] 키보드 'D' 입력은 계속 사용 가능합니다.")
    else:
        print("[serial] 'pyserial' 라이브러리가 없어 아두이노 연동을 건너뜁니다. (pip3 install pyserial)")


    # 경량 웜업
    maybe_run_jetson_perf()
    warmup_opencv_kernels()

    # depth 설정 적용
    apply_depth_overrides(depthmod, S)

    # DB
    conn = Storage.open_db(S.paths.db)

    # DepthEstimator
    try:
        roi_px = (int(S.depth.roi_px[0]), int(S.depth.roi_px[1]))
    except Exception:
        roi_px = (260, 260)
    try:
        roi_off = (int(S.depth.roi_offset[0]), int(S.depth.roi_offset[1]))
    except Exception:
        roi_off = (20, -100)

    depth = DepthEstimator(
        width=S.depth.width, height=S.depth.height, fps=S.depth.fps,
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
        _set_status("error")
        return

    # DM persistent
    dm_handle = DM.open_persistent(S.dm.camera, S.dm.prefer_res, S.dm.prefer_fps)

    # 호환 래퍼
    def _normalize_rois(rois):
        out = []
        if not rois:
            return out
        for r in rois:
            if isinstance(r, dict):
                name = r.get("name", "ROI")
                size = r.get("size", [0, 0]) or [0, 0]
                off  = r.get("offset", [0, 0]) or [0, 0]
                hflip = bool(r.get("hflip", False))
            else:
                name = getattr(r, "name", "ROI")
                size = getattr(r, "size", [0, 0]) or [0, 0]
                off  = getattr(r, "offset", [0, 0]) or [0, 0]
                hflip = bool(getattr(r, "hflip", False))
            try:
                size = [int(size[0]), int(size[1])]
                off  = [int(off[0]),  int(off[1])]
            except Exception:
                size = [0, 0]; off = [0, 0]
            out.append(dict(name=name, size=size, offset=off, hflip=hflip))
        return out

    def datamatrix_scan_persistent(timeout_s=None, debug=False, trace_id=None):
        to = float(getattr(S.dm, "scan_timeout_s", timeout_s if timeout_s is not None else 2.0))
        rois = _normalize_rois(getattr(S.dm, "rois", []))
        return DM.scan_fast4(dm_handle, rois, to, debug=bool(debug), trace_id=trace_id)

    # 임베더
    if same_device(S.dm.camera, S.embedding.cam_dev):
        print(f"[warn] DM_CAMERA({S.dm.camera})와 EMB_CAM_DEV({S.embedding.cam_dev}) 동일. shared persistent handle + lock 사용")
    emb = Emb.build_embedder(S)
    print(f"[img2emb.cfg] dev={S.embedding.cam_dev} {S.embedding.width}x{S.embedding.height}@{S.embedding.fps} pixfmt={S.embedding.pixfmt}")

    # e2e warmup
    Emb.warmup_shared(emb, S, dm_handle, DM.lock(), S.embedding.e2e_warmup_frames, S.embedding.e2e_pregrab)

    # 모델 & 스무더
    engine = Models.InferenceEngine(S)
    smoother = Models.ProbSmoother(
        window=int(getattr(S.model, 'smooth_window', 3)),
        min_votes=int(getattr(S.model, 'smooth_min', 2))
    )

    print("[ready] total init %.2fs" % (time.time()-t_all))
    print("[hint] D + Enter 또는 아두이노 신호 = 측정/스캔/추론")
    print("[hint] L + Enter = LGBM 학습")
    print("[hint] C + Enter = Centroid 학습")
    print("[hint] P = 일시정지 토글")
    if HAVE_FLASK:
        print("[hint] UI: http://localhost:8000")

    paused = False

    try:
        DM_TRACE_ID = 0
        while True:
            engine.reload_if_updated()

            # ★★★★★ [수정 2] 키보드와 아두이노 신호를 모두 감지 ★★★★★
            trigger_scan = False

            # 1. 키보드 입력 확인
            cmd = stdin_readline_nonblock(0.05)
            uc = ""
            if cmd:
                uc = cmd.strip().upper()
                if uc == "D":
                    trigger_scan = True

            # 2. 아두이노 시리얼 입력 확인
            if ser and ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line == '1':
                        print("[serial] 아두이노 신호 '1' 수신")
                        trigger_scan = True
                except Exception:
                    # 시리얼 읽기 오류는 무시하고 계속 진행
                    pass

            # 키보드 'L', 'C', 'P', 'T' 처리 (기존 로직)
            if uc:
                if uc == "L":
                    run_training_now(CONFIG_PATH, force_type="lgbm")
                    engine.reload_if_updated(); continue
                if uc == "C":
                    run_training_now(CONFIG_PATH, force_type="centroid")
                    engine.reload_if_updated(); continue
                if uc == "T":
                    print("[hint] 이제는 L/C로 모델별 학습이 가능합니다. (L=LGBM, C=Centroid)")
                    run_training_now(CONFIG_PATH, force_type=None)
                    engine.reload_if_updated(); continue
                if uc == "P":
                    paused = not paused
                    if paused:
                        _set_status("paused")
                        print("[state] 일시정지 ON: D 또는 아두이노 입력 무시")
                    else:
                        _set_status(None)
                        print("[state] 일시정지 OFF")
                    continue
            
            # ★★★★★ [수정 3] 스캔 트리거 조건 변경 ★★★★★
            if trigger_scan:
                if paused:
                    print("[state] 일시정지 상태. D/아두이노 신호 무시")
                    continue

                DM_TRACE_ID += 1
                tid = DM_TRACE_ID
                t_press = t_now()
                print(f"[D#{tid}][{ts_wall()}] key_down → scan_call")

                _set_status("analyzing", touch=False)
                _set_prob(None, touch=False)
                try:
                    # 1) DM 스캔
                    t_s0 = t_now()
                    payload = datamatrix_scan_persistent(None, debug=False, trace_id=tid)
                    t_s1 = t_now()
                    print(f"[D#{tid}][{ts_wall()}] scan_return elapsed={ms(t_s1 - t_s0)} "
                          f"Tpress→scan_return={ms(t_s1 - t_press)} payload={'YES' if payload else 'NO'}")
                    if payload:
                        print(f"[dm] payload={payload}")
                    else:
                        print("[dm] payload 없음")

                    # (A) DM 성공: 즉시 UI 반영 + 상태=done
                    if payload and _set_ui_from_label(payload):
                        _set_warn(None)
                        _set_prob(None)
                        _set_status("done")

                        # 기록용 depth/embedding 수집
                        t_depth0 = t_now()
                        feat = depth.measure_dimensions(duration_s=1.0, n_frames=10)
                        t_depth1 = t_now()
                        print(f"[D#{tid}][{ts_wall()}] depth_measure elapsed={ms(t_depth1 - t_depth0)}")
                        if feat is None:
                            print("[manual] 측정 실패")
                            continue
                        t_emb0 = t_now()
                        vec = Emb.embed_one_frame_shared(emb, S, dm_handle, DM.lock(), pregrab=3)
                        t_emb1 = t_now()
                        print(f"[D#{tid}][{ts_wall()}] embed_one_frame elapsed={ms(t_emb1 - t_emb0)}")
                        if vec is None:
                            print("[manual] 임베딩 실패")
                            continue
                        if feat["q"] < float(getattr(S.quality, 'q_warn', 0.30)):
                            print(f"[notify] 품질 경고: q={feat['q']:.2f} (임계 {float(getattr(S.quality, 'q_warn', 0.30)):.2f})")
                        Storage.on_sample_record(conn, feat, vec, product_id=payload, has_label=1, origin="manual_dm")
                        smoother.buf.clear()
                        print("[infer] skip: DM 라벨 확정 → 모델 추론 생략")
                        continue
                    else:
                        print("[ui] DM 라벨 없음/형식 불일치 → 모델 라벨에서 파싱 예정")

                    # 2) depth 측정
                    t_depth0 = t_now()
                    feat = depth.measure_dimensions(duration_s=1.0, n_frames=10)
                    t_depth1 = t_now()
                    print(f"[D#{tid}][{ts_wall()}] depth_measure elapsed={ms(t_depth1 - t_depth0)}")
                    if feat is None:
                        print("[manual] 측정 실패"); _set_status("error"); continue

                    # 3) 임베딩
                    t_emb0 = t_now()
                    vec = Emb.embed_one_frame_shared(emb, S, dm_handle, DM.lock(), pregrab=3)
                    t_emb1 = t_now()
                    print(f"[D#{tid}][{ts_wall()}] embed_one_frame elapsed={ms(t_emb1 - t_emb0)}")
                    if vec is None:
                        print("[manual] 임베딩 실패"); _set_status("error"); continue

                    # 품질 알림
                    if feat["q"] < float(getattr(S.quality, 'q_warn', 0.30)):
                        print(f"[notify] 품질 경고: q={feat['q']:.2f} (임계 {float(getattr(S.quality, 'q_warn', 0.30)):.2f})")

                    # 4) 저장
                    Storage.on_sample_record(conn, feat, vec, product_id=None, has_label=0, origin="manual_no_dm")

                    # === 벡터 구성(15+128) ===
                    meta = np.array([
                        feat["d1"], feat["d2"], feat["d3"],
                        feat["mad1"], feat["mad2"], feat["mad3"],
                        feat["r1"], feat["r2"], feat["r3"],
                        feat["sr1"], feat["sr2"], feat["sr3"],
                        feat["logV"], feat["logsV"], feat["q"]
                    ], np.float32)
                    full_vec = np.concatenate([meta, vec], axis=0)
                    full_vec = l2_normalize(full_vec)
                    print(f"[vector] dim={full_vec.shape[0]}")
                    print(f"[debug] ||full_vec||={np.linalg.norm(full_vec):.6f}")

                    # 5) 추론
                    top_lab, top_p, gap, backend = engine.infer(full_vec)
                    if backend is None:
                        print("[infer] 모델 없음(파일 미존재 또는 로드 실패)")
                        _set_prob(None); _set_status("error"); continue
                    print(f"[infer] {backend} top1: {top_lab} p={top_p:.3f} gap={gap:.4f}")
                    _set_prob(top_p)

                    # ---- UI 즉시 갱신 + p 임계 기반 상태/경고 결정 ----
                    prob_th = float(getattr(S.model, 'prob_threshold', 0.40))
                    low_prob = (top_p < prob_th)

                    updated = _set_ui_from_label(top_lab)
                    if low_prob:
                        _set_warn("Warning! Percent is very low")
                        _set_status("error")   # ★ p<th → 오류발생(빨간)
                    else:
                        _set_warn(None)
                        _set_status("done")    # ★ p>=th → 분석 완료(초록)

                    if not updated:
                        print("[ui] 모델 라벨 파싱 실패(16자리 규격 아님)")

                    # ---- 스무딩(로그용) ----
                    min_margin = float(getattr(S.model, 'min_margin', 0.05))
                    if gap < min_margin:
                        smoother.push(top_lab, top_p)
                        print(f"[smooth] hold: small_margin gap={gap:.4f} (<{min_margin:.3f}), len={len(smoother.buf)}/{S.model.smooth_window}, top={top_lab} p={top_p:.2f}")
                        decided = smoother.maybe_decide(threshold=prob_th)
                        if decided is not None:
                            lab, avgp = decided
                            print(f"[decision] smoothed: {lab} p={avgp:.2f}")
                    elif low_prob:
                        smoother.push(top_lab, top_p)
                        print(f"[smooth] hold: len={len(smoother.buf)}/{S.model.smooth_window}, top={top_lab} p={top_p:.2f} (<{prob_th:.2f})")
                        decided = smoother.maybe_decide(threshold=prob_th)
                        if decided is not None:
                            lab, avgp = decided
                            print(f"[decision] smoothed: {lab} p={avgp:.2f}")
                    else:
                        smoother.push(top_lab, top_p)
                        decided = smoother.maybe_decide(threshold=prob_th)
                        if decided is None:
                            lab, votes, avgp = smoother.status()
                            if lab is None:
                                print(f"[smooth] hold: len={len(smoother.buf)}/{S.model.smooth_window}")
                            else:
                                print(f"[smooth] hold: len={len(smoother.buf)}/{S.model.smooth_window}, lead={lab} votes={votes} avgp={avgp:.2f}")
                        else:
                            lab, avgp = decided
                            print(f"[decision] smoothed: {lab} p={avgp:.2f}")

                    # 주의: 여기서는 상태를 다시 덮어쓰지 않음(위에서 p 기준으로 이미 결정)
                    continue

                except Exception as e:
                    print("[error] D 처리 중 예외:", e)
                    _set_status("error")
                    _set_warn("오류가 발생했습니다.")
                    _set_prob(None)
                    continue

    except KeyboardInterrupt:
        print("[exit] keyboard interrupt")
    finally:
        # ★★★★★ [수정 4] 종료 시 시리얼 포트 닫기 ★★★★★
        if ser and ser.is_open:
            ser.close()
            print("[cleanup] serial port closed")
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA optimize")
        except Exception:
            pass
        try:
            depth.stop()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        try:
            DM.close_persistent()
        except Exception:
            pass
        print("[cleanup] stopped")

if __name__ == "__main__":
    main()