# main.py (아두이노 시리얼 + 저속-대기 펌프 + 버스트 플러시 + 방향 최적화)
# - 웹 자산: web/index.html, web/style.css, web/script.js만 사용 (ui.css/ui.js 완전 제거)
# - 아두이노 "1\n"=스캔, "2\n"=일시정지 ON, "3\n"=일시정지 OFF
# - 카메라 상시 켜두고 저속 펌프 → 트리거 시 버스트로 최신 프레임 확보
# - 라벨 16자리 마지막(s[15])=wrap_layers
# - 방향 최적화: 롤폭(기본 200mm)으로 덮는 축 선택, bands=ceil(덮는축/롤폭)
# - DM 없으면 모델 라벨(16자리)에서 타입/치수 파싱 → UI 반영
# - p < 0.40 → "error(빨간)" / p ≥ 0.40 → "done(초록)"
# - 상태 램프 순서는 index.html에서 제어(서버는 상태값만 제공)
# - 환경: Python 3.6 / OpenCV 4.1.1 / PyTorch 1.10 / torchvision 0.11 / CPU-only

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
import math

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

# ★ 웹 자산 디렉터리(고정)
WEB_DIR = ROOT / "web"   # web/index.html, web/style.css, web/script.js

# ---- 뽁뽁이 롤 폭(mm) ----
ROLL_WIDTH_MM = int(os.getenv("BUBBLE_ROLL_WIDTH_MM", "200"))

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
    return int(math.ceil(float(x) / float(base)) * base)

def _touch_ts():
    with _UI_LOCK:
        _UI_STATE["updated_ts"] = int(time.time())

def _set_ui_type(tid: Optional[str]):
    with _UI_LOCK:
        _UI_STATE["type_id"] = tid
        _UI_STATE["type_name"] = type_name_from_id(tid) if tid else None
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

# ---- 기본 계산식(라벨 없음) : 방향 최적화 ----
def compute_bubble_length_mm(W, L, H,
                             layers=None, overlap_mm=None,
                             slack_ratio=None, round_to_mm=None,
                             roll_width_mm: Optional[int] = None):
    """
    세 방향(W,L,H) 중 하나를 롤폭으로 덮는다고 가정하고,
    bands = ceil(덮는축 / 롤폭), 한 바퀴 둘레 = 2*(나머지 두 변의 합)
    총길이 = 둘레 * layers * bands + overlap
    이후 slack 적용, round_to_mm로 올림 → 최솟값 선택
    """
    with _UI_LOCK:
        p = _UI_STATE["params"].copy()

    layers      = p["layers"]      if layers      is None else layers
    overlap_mm  = p["overlap_mm"]  if overlap_mm  is None else overlap_mm
    slack_ratio = p["slack_ratio"] if slack_ratio is None else slack_ratio
    round_to_mm = p["round_to_mm"] if round_to_mm is None else round_to_mm
    roll_width  = ROLL_WIDTH_MM if roll_width_mm is None else int(roll_width_mm)

    dims = [float(W), float(L), float(H)]
    best = None

    for i in range(3):
        axis  = dims[i]
        other = [dims[j] for j in range(3) if j != i]
        bands = max(1, int(math.ceil(axis / float(roll_width))))
        perim = 2.0 * (other[0] + other[1])
        base  = perim * float(layers) * bands + float(overlap_mm)
        total = base * (1.0 + float(slack_ratio))
        total = _round_up(total, round_to_mm)
        if best is None or total < best:
            best = total

    return int(best)

def _set_ui_dims_and_bubble(w_mm, l_mm, h_mm):
    bubble_mm = compute_bubble_length_mm(w_mm, l_mm, h_mm)
    with _UI_LOCK:
        _UI_STATE["W"] = int(round(w_mm))
        _UI_STATE["L"] = int(round(l_mm))
        _UI_STATE["H"] = int(round(h_mm))
        _UI_STATE["bubble_mm"] = int(bubble_mm)
        _UI_STATE["updated_ts"] = int(time.time())

# ---- 16자리 라벨 파싱 ----
def parse_dm_label_16(s):
    """
    [0:6]=type_id, [6:9]=W, [9:12]=L, [12:15]=H, [15]=wrap_layers
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
            "wrap_layers": int(s[15]),
            "raw":         s,
        }
    except Exception:
        return None

# ---- 라벨용 계산(오버랩/슬랙 없음) : 방향 최적화 ----
def compute_bubble_length_perimeter_layers_oriented(W, L, H, layers, roll_width_mm: Optional[int] = None):
    roll_width = ROLL_WIDTH_MM if roll_width_mm is None else int(roll_width_mm)
    try:
        layers = int(layers)
    except Exception:
        layers = 1
    if layers < 1:
        layers = 1

    dims = [float(W), float(L), float(H)]
    best = None

    for i in range(3):
        axis  = dims[i]
        other = [dims[j] for j in range(3) if j != i]
        bands = max(1, int(math.ceil(axis / float(roll_width))))
        perim = 2.0 * (other[0] + other[1])
        total = perim * float(layers) * bands
        total = int(round(total))
        if best is None or total < best:
            best = total

    return int(best)

def _set_ui_from_label(s):
    info = parse_dm_label_16(s)
    if info is None:
        return False
    _set_ui_type(info["type_id"])

    bubble_mm = compute_bubble_length_perimeter_layers_oriented(
        info["W"], info["L"], info["H"], info.get("wrap_layers", 1)
    )
    with _UI_LOCK:
        _UI_STATE["W"] = int(info["W"])
        _UI_STATE["L"] = int(info["L"])
        _UI_STATE["H"] = int(info["H"])
        _UI_STATE["bubble_mm"] = int(bubble_mm)
        _UI_STATE["updated_ts"] = int(time.time())

        bb    = _UI_STATE['bubble_mm']
        tname = _UI_STATE['type_name']
        wl    = info.get("wrap_layers", 1)

    print("[ui] LABEL 사용: type_id={}({}) W={} L={} H={} mm wrap_layers={} → 방향최적화 bubble={} mm".format(
        info['type_id'], tname, info['W'], info['L'], info['H'], wl, bb
    ))
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

    # === 웹 자산: web/ 폴더만 ===
    @_APP.route("/")
    def home():
        return send_from_directory(str(WEB_DIR), "index.html")

    @_APP.route("/style.css")
    def style_css():
        return send_from_directory(str(WEB_DIR), "style.css")

    @_APP.route("/script.js")
    def script_js():
        return send_from_directory(str(WEB_DIR), "script.js")

    # web/ 내 기타 자산
    @_APP.route("/web/<path:filename>")
    def serve_web_assets(filename):
        return send_from_directory(str(WEB_DIR), filename)

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
            print("[ui] Serving with Waitress (threads={}) on {}:{}".format(threads, host, port))
            _waitress_serve(_APP, host=host, port=port, threads=threads, ident=None)
        else:
            print("[ui] Serving with Flask dev server on {}:{}".format(host, port))
            kwargs = dict(host=host, port=port, threaded=True, debug=False, use_reloader=False)
            try:
                if _SilentWSGIRequestHandler is not None:
                    kwargs["request_handler"] = _SilentWSGIRequestHandler
            except Exception:
                pass
            _APP.run(**kwargs)
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    print("[ui] 웹 UI 실행: http://<Jetson_IP>:{}  (iPad/Safari 가능)".format(port))

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

# =======================
# 프레임 펌프 / 버스트 플러시
# =======================

_RUN_PUMP = False
_PUMP_THREAD = None
_SCAN_BUSY = threading.Event()

def _start_frame_pump(dm_handle, rois, idle_fps: float = 1.0):
    global _RUN_PUMP, _PUMP_THREAD
    if dm_handle is None:
        print("[pump] dm_handle 없음 → 펌프 미사용")
        return
    if idle_fps <= 0:
        print("[pump] idle_fps<=0 → 펌프 비활성화")
        return

    _RUN_PUMP = True
    period = 1.0 / float(idle_fps)

    def _loop():
        print("[pump] start idle_fps={:.2f} (period={:.3f}s)".format(idle_fps, period))
        while _RUN_PUMP:
            if _SCAN_BUSY.is_set():
                time.sleep(period)
                continue
            try:
                DM.scan_fast4(dm_handle, rois, 0.005, debug=False, trace_id=None)
            except Exception:
                pass
            time.sleep(period)
        print("[pump] stopped")

    _PUMP_THREAD = threading.Thread(target=_loop, daemon=True)
    _PUMP_THREAD.start()

def _stop_frame_pump():
    global _RUN_PUMP, _PUMP_THREAD
    _RUN_PUMP = False
    if _PUMP_THREAD is not None:
        try:
            _PUMP_THREAD.join(timeout=1.0)
        except Exception:
            pass
        _PUMP_THREAD = None

def _burst_flush(dm_handle, rois, n=10, to_sec=0.001):
    try:
        for _ in range(int(n)):
            DM.scan_fast4(dm_handle, rois, float(to_sec), debug=False, trace_id=None)
    except Exception:
        pass

# =======================
# 임베딩 모드 래퍼 (3뷰 concat 지원; pregrab 최소화)
# =======================

def _inject_default_embedding_options(S):
    # 옵션 존재 보정(값은 Emb 구현이 처리)
    if not hasattr(S, 'embedding'):
        class _E: pass
        S.embedding = _E()
    if not hasattr(S.embedding, 'concat3'):
        S.embedding.concat3 = bool(int(os.getenv("EMB_CONCAT3", "0")))
    if not hasattr(S.embedding, 'mirror_period'):
        S.embedding.mirror_period = int(os.getenv("EMB_MIRROR_PERIOD", "3"))
    if not hasattr(S.embedding, 'rois3'):
        S.embedding.rois3 = None  # Emb 내부 기본 또는 dm.rois 활용

def _embed_one_any(emb, S, dm_handle, pregrab=0):
    """
    Emb에 concat3 경로가 있으면 사용, 없으면 단일뷰로 폴백.
    pregrab은 CPU-only에선 0 권장(프레임 대기 제거).
    """
    want_concat3 = bool(getattr(S.embedding, 'concat3', False))
    if want_concat3 and hasattr(Emb, "embed_one_frame_shared_concat3"):
        try:
            return Emb.embed_one_frame_shared_concat3(
                emb, S, dm_handle, DM.lock(),
                pregrab=int(pregrab),
                mirror_period=int(getattr(S.embedding, 'mirror_period', 3)),
                rois3=getattr(S.embedding, 'rois3', None)
            )
        except Exception as e:
            print("[embed] concat3 경로 예외 → 단일뷰 폴백:", e)
    return Emb.embed_one_frame_shared(emb, S, dm_handle, DM.lock(), pregrab=int(pregrab))

def main():
    t_all = time.time()
    print("[init] start]")

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

    # 3뷰 옵션 기본값 주입
    _inject_default_embedding_options(S)

    # UI 시작
    _start_web_ui()

    # 시리얼 설정
    ser = None
    if HAVE_SERIAL:
        SERIAL_PORT = os.getenv("ARDUINO_PORT", "/dev/ttyACM0")
        BAUD_RATE = int(os.getenv("ARDUINO_BAUD", "9600"))
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            print("[serial] 아두이노 연결 성공: {} (Baud: {})".format(SERIAL_PORT, BAUD_RATE))
        except Exception as e:
            print("[serial] 아두이노 연결 실패: {}".format(e))
            print("[serial] 키보드 'D' 입력은 계속 사용 가능합니다.")
    else:
        print("[serial] 'pyserial' 미설치 → 아두이노 연동 생략 (pip3 install pyserial)")

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
    print("[warmup] RealSense frames={}".format(frames))
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

    # ROI 정규화
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
        print("[warn] DM_CAMERA({})와 EMB_CAM_DEV({}) 동일. shared persistent handle + lock 사용".format(S.dm.camera, S.embedding.cam_dev))
    emb = Emb.build_embedder(S)
    print("[img2emb.cfg] dev={} {}x{}@{} pixfmt={}".format(
        S.embedding.cam_dev, S.embedding.width, S.embedding.height, S.embedding.fps, S.embedding.pixfmt
    ))

    # e2e warmup (임베딩 백엔드 워밍업)
    Emb.warmup_shared(emb, S, dm_handle, DM.lock(), S.embedding.e2e_warmup_frames, S.embedding.e2e_pregrab)

    # 프레임 펌프 시작
    IDLE_FPS = float(os.getenv("IDLE_PUMP_FPS", "1.0"))
    rois_for_pump = _normalize_rois(getattr(S.dm, "rois", []))
    _start_frame_pump(dm_handle, rois_for_pump, idle_fps=IDLE_FPS)

    # 모델 & 스무더
    engine = Models.InferenceEngine(S)
    smoother = Models.ProbSmoother(
        window=int(getattr(S.model, 'smooth_window', 3)),
        min_votes=int(getattr(S.model, 'smooth_min', 2))
    )

    # 임베딩 모드 안내
    if getattr(S.embedding, 'concat3', False):
        print("[embed] 3-view concat 모드 활성: mirror_period={}".format(
            int(getattr(S.embedding, 'mirror_period', 3))
        ))
    else:
        print("[embed] 단일뷰 모드")

    print("[ready] total init %.2fs" % (time.time()-t_all))
    print("[hint] D + Enter 또는 아두이노 신호 = 측정/스캔/추론")
    print("[hint] L + Enter = LGBM 학습")
    print("[hint] C + Enter = Centroid 학습")
    print("[hint] P = 일시정지 토글")
    if HAVE_FLASK:
        print("[hint] UI: http://localhost:8000")

    paused = False

    # 임베딩 트리거 시 pregrab(프레임 선확보) — CPU-only/6fps 환경에선 0 권장
    EMB_PREGRAB_ON_TRIGGER = int(os.getenv("EMB_PREGRAB_ON_TRIGGER", "0"))

    try:
        DM_TRACE_ID = 0
        while True:
            engine.reload_if_updated()

            # 트리거 감지
            trigger_scan = False

            # 1) 키보드
            cmd = stdin_readline_nonblock(0.05)
            uc = ""
            if cmd:
                uc = cmd.strip().upper()
                if uc == "D":
                    trigger_scan = True

            # 2) 아두이노
            if ser and ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line == '1':
                        print("[serial] 아두이노 '1' → 스캔 트리거(D)")
                        trigger_scan = True
                    elif line == '2':
                        if not paused:
                            paused = True
                            _set_status("paused")
                            print("[serial] 아두이노 '2' → 일시정지 ON")
                        else:
                            print("[serial] '2' 수신: 이미 일시정지 상태")
                    elif line == '3':
                        if paused:
                            paused = False
                            _set_status(None)
                            print("[serial] 아두이노 '3' → 일시정지 OFF")
                        else:
                            print("[serial] '3' 수신: 이미 동작 중")
                except Exception:
                    pass

            # L/C/T/P 핫키
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
                        print("[state] 일시정지 ON: D/아두이노 입력 무시")
                    else:
                        _set_status(None)
                        print("[state] 일시정지 OFF")
                    continue

            # 스캔 트리거
            if trigger_scan:
                if paused:
                    print("[state] 일시정지 상태. D/아두이노 신호 무시")
                    continue

                DM_TRACE_ID += 1
                tid = DM_TRACE_ID
                t_press = t_now()
                print("[D#{}][{}] key_down → scan_call".format(tid, ts_wall()))

                _set_status("analyzing")
                _set_prob(None)

                # 펌프 일시 정지 + 버스트 플러시
                _SCAN_BUSY.set()
                try:
                    BURST_N = int(os.getenv("BURST_FLUSH_N", "10"))
                    BURST_TO_MS = int(os.getenv("BURST_FLUSH_TO_MS", "1"))
                    n_flush = BURST_N if BURST_N > 0 else 10
                    t_flush = (BURST_TO_MS if BURST_TO_MS > 0 else 1) / 1000.0
                    _burst_flush(dm_handle, rois_for_pump, n=n_flush, to_sec=t_flush)
                except Exception:
                    pass

                try:
                    # 1) DM 스캔
                    t_s0 = t_now()
                    payload = datamatrix_scan_persistent(None, debug=False, trace_id=tid)
                    t_s1 = t_now()
                    print("[D#{}][{}] scan_return elapsed={} Tpress→scan_return={} payload={}".format(
                        tid, ts_wall(), ms(t_s1 - t_s0), ms(t_s1 - t_press), "YES" if payload else "NO"
                    ))
                    if payload:
                        print("[dm] payload={}".format(payload))
                    else:
                        print("[dm] payload 없음")

                    # (A) DM 성공 → UI 즉시 업데이트 후 보조 기록만
                    if payload and _set_ui_from_label(payload):
                        _set_warn(None)
                        _set_prob(None)
                        _set_status("done")

                        # 기록용 depth/embedding 수집
                        t_depth0 = t_now()
                        feat = depth.measure_dimensions(duration_s=1.0, n_frames=10)
                        t_depth1 = t_now()
                        print("[D#{}][{}] depth_measure elapsed={}".format(tid, ts_wall(), ms(t_depth1 - t_depth0)))
                        if feat is None:
                            print("[manual] 측정 실패")
                            _SCAN_BUSY.clear()
                            continue
                        t_emb0 = t_now()
                        vec = _embed_one_any(emb, S, dm_handle, pregrab=EMB_PREGRAB_ON_TRIGGER)  # ★ pregrab=0
                        t_emb1 = t_now()
                        print("[D#{}][{}] embed_one_frame elapsed={}".format(tid, ts_wall(), ms(t_emb1 - t_emb0)))
                        if vec is None:
                            print("[manual] 임베딩 실패")
                            _SCAN_BUSY.clear()
                            continue
                        if feat["q"] < float(getattr(S.quality, 'q_warn', 0.30)):
                            print("[notify] 품질 경고: q={:.2f} (임계 {:.2f})".format(
                                feat['q'], float(getattr(S.quality, 'q_warn', 0.30))
                            ))
                        Storage.on_sample_record(conn, feat, vec, product_id=payload, has_label=1, origin="manual_dm")
                        smoother.buf.clear()
                        print("[infer] skip: DM 라벨 확정 → 모델 추론 생략")
                        _SCAN_BUSY.clear()
                        continue
                    else:
                        print("[ui] DM 라벨 없음/형식 불일치 → 모델 라벨에서 파싱 예정")

                    # 2) depth 측정
                    t_depth0 = t_now()
                    feat = depth.measure_dimensions(duration_s=1.0, n_frames=10)
                    t_depth1 = t_now()
                    print("[D#{}][{}] depth_measure elapsed={}".format(tid, ts_wall(), ms(t_depth1 - t_depth0)))
                    if feat is None:
                        print("[manual] 측정 실패"); _set_status("error"); _SCAN_BUSY.clear(); continue

                    # 3) 임베딩 (단일/3뷰 자동)
                    t_emb0 = t_now()
                    vec = _embed_one_any(emb, S, dm_handle, pregrab=EMB_PREGRAB_ON_TRIGGER)  # ★ pregrab=0
                    t_emb1 = t_now()
                    print("[D#{}][{}] embed_one_frame elapsed={}".format(tid, ts_wall(), ms(t_emb1 - t_emb0)))
                    if vec is None:
                        print("[manual] 임베딩 실패"); _set_status("error"); _SCAN_BUSY.clear(); continue

                    # 품질 알림
                    if feat["q"] < float(getattr(S.quality, 'q_warn', 0.30)):
                        print("[notify] 품질 경고: q={:.2f} (임계 {:.2f})".format(
                            feat['q'], float(getattr(S.quality, 'q_warn', 0.30))
                        ))

                    # 4) 저장
                    Storage.on_sample_record(conn, feat, vec, product_id=None, has_label=0, origin="manual_no_dm")

                    # === 벡터 구성(15 + emb) ===
                    meta = np.array([
                        feat["d1"], feat["d2"], feat["d3"],
                        feat["mad1"], feat["mad2"], feat["mad3"],
                        feat["r1"], feat["r2"], feat["r3"],
                        feat["sr1"], feat["sr2"], feat["sr3"],
                        feat["logV"], feat["logsV"], feat["q"]
                    ], np.float32)
                    full_vec = np.concatenate([meta, vec], axis=0)
                    full_vec = l2_normalize(full_vec)
                    print("[vector] dim={} (meta 15 + emb {})".format(full_vec.shape[0], vec.shape[0]))
                    print("[debug] ||full_vec||={:.6f}".format(np.linalg.norm(full_vec)))

                    # 5) 추론
                    top_lab, top_p, gap, backend = engine.infer(full_vec)
                    if backend is None:
                        print("[infer] 모델 없음(파일 미존재 또는 로드 실패)")
                        _set_prob(None); _set_status("error"); _SCAN_BUSY.clear(); continue
                    print("[infer] {} top1: {} p={:.3f} gap={:.4f}".format(backend, top_lab, top_p, gap))
                    _set_prob(top_p)

                    # ---- UI 갱신 + p 임계 기반 상태/경고 ----
                    prob_th = float(getattr(S.model, 'prob_threshold', 0.40))
                    low_prob = (top_p < prob_th)

                    updated = _set_ui_from_label(top_lab)
                    if low_prob:
                        _set_warn("Warning! Percent is very low")
                        _set_status("error")
                    else:
                        _set_warn(None)
                        _set_status("done")

                    if not updated:
                        print("[ui] 모델 라벨 파싱 실패(16자리 규격 아님)")

                    # ---- 스무딩(로그용) ----
                    min_margin = float(getattr(S.model, 'min_margin', 0.05))
                    if gap < min_margin:
                        smoother.push(top_lab, top_p)
                        print("[smooth] hold: small_margin gap={:.4f} (<{:.3f}), len={}/{}, top={} p={:.2f}".format(
                            gap, min_margin, len(smoother.buf), S.model.smooth_window, top_lab, top_p
                        ))
                        decided = smoother.maybe_decide(threshold=prob_th)
                        if decided is not None:
                            lab, avgp = decided
                            print("[decision] smoothed: {} p={:.2f}".format(lab, avgp))
                    elif low_prob:
                        smoother.push(top_lab, top_p)
                        print("[smooth] hold: len={}/{} , top={} p={:.2f} (<{:.2f})".format(
                            len(smoother.buf), S.model.smooth_window, top_lab, top_p, prob_th
                        ))
                        decided = smoother.maybe_decide(threshold=prob_th)
                        if decided is not None:
                            lab, avgp = decided
                            print("[decision] smoothed: {} p={:.2f}".format(lab, avgp))
                    else:
                        smoother.push(top_lab, top_p)
                        decided = smoother.maybe_decide(threshold=prob_th)
                        if decided is None:
                            status = smoother.status()
                            if status[0] is None:
                                print("[smooth] hold: len={}/{}".format(len(smoother.buf), S.model.smooth_window))
                            else:
                                lab, votes, avgp = status
                                print("[smooth] hold: len={}/{} , lead={} votes={} avgp={:.2f}".format(
                                    len(smoother.buf), S.model.smooth_window, lab, votes, avgp
                                ))
                        else:
                            lab, avgp = decided
                            print("[decision] smoothed: {} p={:.2f}".format(lab, avgp))

                except Exception as e:
                    print("[error] D 처리 중 예외:", e)
                    _set_status("error")
                    _set_warn("오류가 발생했습니다.")
                    _set_prob(None)
                finally:
                    _SCAN_BUSY.clear()

    except KeyboardInterrupt:
        print("[exit] keyboard interrupt")
    finally:
        _stop_frame_pump()
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
