# main.py  핵심 동작 요약
# 입력은 RealSense D435와 아두이노 신호, 출력은 웹 UI와 커팅기 제어
# 흐름은 초기화 후 웹 UI 기동, 프레임 펌프, 이벤트 루프, 안전 종료

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

# Serial / 아두이노 연동용
HAVE_SERIAL = True
try:
    import serial
except ImportError:
    HAVE_SERIAL = False

# Flask 가용성 체크
HAVE_FLASK = True
try:
    from flask import Flask, jsonify, request, make_response, send_from_directory
except Exception:
    HAVE_FLASK = False

# Waitress 가용성 체크
HAVE_WAITRESS = True
try:
    from waitress import serve as _waitress_serve
except Exception:
    HAVE_WAITRESS = False

# 경로 세팅
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

# 웹 디렉터리
WEB_DIR = ROOT / "web"  # web/index.html, web/style.css, web/script.js

# 뽁뽁이 롤 폭(200mm)
ROLL_WIDTH_MM = int(os.getenv("BUBBLE_ROLL_WIDTH_MM", "200"))

# 로컬 모듈
from depth import DepthEstimator
import depth as depthmod
from core.config import load_config, get_settings, apply_depth_overrides
from core.lite import DM, Emb, Models, Storage
from core.utils import (
    ts_wall,
    t_now,
    ms,
    stdin_readline_nonblock,
    maybe_run_jetson_perf,
    warmup_opencv_kernels,
    l2_normalize,
    same_device,
)

# DM 프레임 읽기
try:
    from datamatrix import read_frame_nonblocking as dm_read_frame
except Exception:
    dm_read_frame = None

np.set_printoptions(suppress=True, linewidth=200, threshold=50, precision=4)

# 타입 매핑
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

# UI 공유 상태
_UI_LOCK = threading.Lock()
_UI_STATE = {
    "type_id": None,
    "type_name": None,
    "W": None,
    "L": None,
    "H": None,  # mm
    "bubble_mm": None,
    "p": None,  # top-1 prob
    "status": None,  # "analyzing" | "done" | "paused" | "error" | None
    "updated_ts": None,
    "warn_msg": None,
    "params": {
        "layers": 2,
        "overlap_mm": 120,
        "slack_ratio": 0.03,
        "round_to_mm": 10,
    },
    "cap_image": None,
    "event_id": None,
}

# 최신 캡처된 JPEG 메모리에서 응답
_CAP_JPEG: Optional[bytes] = None

# 최신 캡처 JPEG를 메모리에 저장하고 UI 상태에 반영
def _set_cap_jpeg(jpeg_bytes: Optional[bytes]):
    global _CAP_JPEG
    with _UI_LOCK:
        _CAP_JPEG = jpeg_bytes
        _UI_STATE["cap_image"] = "__mem__" if jpeg_bytes else None
        _UI_STATE["updated_ts"] = int(time.time())

# 기준 배수로 올림한 정수 반환
def _round_up(x, base):
    if base <= 0:
        return int(x)
    return int(math.ceil(float(x) / float(base)) * base)

# UI 타임스탬프 갱신
def _touch_ts():
    with _UI_LOCK:
        _UI_STATE["updated_ts"] = int(time.time())

# UI에 타입 ID와 이름 기록
def _set_ui_type(tid: Optional[str]):
    with _UI_LOCK:
        _UI_STATE["type_id"] = tid
        _UI_STATE["type_name"] = type_name_from_id(tid) if tid else None
        _UI_STATE["updated_ts"] = int(time.time())

# UI에 경고 메시지 설정 또는 해제
def _set_warn(msg=None):
    with _UI_LOCK:
        _UI_STATE["warn_msg"] = msg if msg else None
        _UI_STATE["updated_ts"] = int(time.time())

# UI에 확률값 설정
def _set_prob(pval, touch=True):
    with _UI_LOCK:
        try:
            _UI_STATE["p"] = None if pval is None else float(pval)
        except Exception:
            _UI_STATE["p"] = None
        if touch:
            _UI_STATE["updated_ts"] = int(time.time())

# UI 상태 문자열 설정
def _set_status(s: Optional[str], touch=True):
    with _UI_LOCK:
        _UI_STATE["status"] = s
        if touch:
            _UI_STATE["updated_ts"] = int(time.time())

# UI에 이벤트 고유 ID 설정
def _set_event_id(eid: int):
    with _UI_LOCK:
        _UI_STATE["event_id"] = int(eid)
        _UI_STATE["updated_ts"] = int(time.time())

# UI에 표시할 캡처 이미지 소스 설정
def _set_cap_image(fname: Optional[str]):
    with _UI_LOCK:
        _UI_STATE["cap_image"] = fname if fname else None
        _UI_STATE["updated_ts"] = int(time.time())

# W L H를 받아 방향 최적화로 최소 포장 길이 계산
def compute_bubble_length_mm(
    W, L, H, layers=None, overlap_mm=None, slack_ratio=None, round_to_mm=None, roll_width_mm: Optional[int] = None,
):
    with _UI_LOCK:
        p = _UI_STATE["params"].copy()
        layers = p["layers"] if layers is None else layers
        overlap_mm = p["overlap_mm"] if overlap_mm is None else overlap_mm
        slack_ratio = p["slack_ratio"] if slack_ratio is None else slack_ratio
        round_to_mm = p["round_to_mm"] if round_to_mm is None else round_to_mm
    roll_width = ROLL_WIDTH_MM if roll_width_mm is None else int(roll_width_mm)

    dims = [float(W), float(L), float(H)]
    best = None
    for i in range(3):
        axis = dims[i]
        other = [dims[j] for j in range(3) if j != i]
        bands = max(1, int(math.ceil(axis / float(roll_width))))
        perim = 2.0 * (other[0] + other[1])
        base = perim * float(layers) * bands + float(overlap_mm)
        total = base * (1.0 + float(slack_ratio))
        total = _round_up(total, round_to_mm)
        if best is None or total < best:
            best = total
    return int(best)

# 측정된 W L H를 UI에 반영하고 포장 길이도 함께 기록
def _set_ui_dims_and_bubble(w_mm, l_mm, h_mm):
    bubble_mm = compute_bubble_length_mm(w_mm, l_mm, h_mm)
    with _UI_LOCK:
        _UI_STATE["W"] = int(round(w_mm))
        _UI_STATE["L"] = int(round(l_mm))
        _UI_STATE["H"] = int(round(h_mm))
        _UI_STATE["bubble_mm"] = int(bubble_mm)
        _UI_STATE["updated_ts"] = int(time.time())

# 16자리 Datamatrix 라벨을 파싱해 규격 필드로 분해
def parse_dm_label_16(s):
    if s is None:
        return None
    s = str(s).strip()
    if len(s) != 16 or (not s.isdigit()):
        return None
    try:
        return {
            "type_id": s[0:6],
            "W": int(s[6:9]),
            "L": int(s[9:12]),
            "H": int(s[12:15]),
            "wrap_layers": int(s[15]),
            "raw": s,
        }
    except Exception:
        return None

# 고정 레이어 수로 각 방향을 평가해 최소 포장 길이 계산
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
        axis = dims[i]
        other = [dims[j] for j in range(3) if j != i]
        bands = max(1, int(math.ceil(axis / float(roll_width))))
        perim = 2.0 * (other[0] + other[1])
        total = perim * float(layers) * bands
        total = int(round(total))
        if best is None or total < best:
            best = total
    return int(best)

# 라벨 문자열을 파싱해 UI에 치수와 포장 길이를 반영
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
    bb = _UI_STATE["bubble_mm"]
    tname = _UI_STATE["type_name"]
    wl = info.get("wrap_layers", 1)
    print(f"[ui] LABEL 사용: type_id={info['type_id']}({tname}) W={info['W']} L={info['L']} H={info['H']} mm wrap_layers={wl} → 방향최적화 bubble={bb} mm")
    return True

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
            def log(self, *args, **kwargs): pass
    except Exception:
        _SilentWSGIRequestHandler = None

    _APP = Flask(__name__)
    IMG_DIR = ROOT / "imgfile"
    try:
        IMG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # 루트 페이지 제공
    @_APP.route("/")
    def home(): return send_from_directory(str(WEB_DIR), "index.html")

    # CSS 제공
    @_APP.route("/style.css")
    def style_css(): return send_from_directory(str(WEB_DIR), "style.css")

    # JS 제공
    @_APP.route("/script.js")
    def script_js(): return send_from_directory(str(WEB_DIR), "script.js")

    # web 폴더 정적 자원 제공
    @_APP.route("/web/<path:filename>")
    def serve_web_assets(filename): return send_from_directory(str(WEB_DIR), filename)

    # UI 상태 JSON 제공
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

    # 포장 파라미터 조회 및 설정
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
            p.update({"layers": L, "overlap_mm": O, "slack_ratio": S_, "round_to_mm": R})
            res = {"ok": True, "params": p}
        _touch_ts()
        return jsonify(res)

    # 헬스체크 응답
    @_APP.route("/healthz")
    def healthz(): return jsonify(ok=True, ts=int(time.time()))

    # 이미지 파일 제공
    @_APP.route("/imgfile/<path:filename>")
    def serve_imgfile(filename):
        try:
            return send_from_directory(str(IMG_DIR), filename)
        except Exception:
            return ("", 404)

    # 최신 메모리 캡처 JPEG 제공
    @_APP.route("/api/capture.jpg")
    def api_capture_jpeg():
        with _UI_LOCK:
            data = _CAP_JPEG
        if not data:
            return ("", 404)
        resp = make_response(data)
        resp.mimetype = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

# Flask 웹 UI 서버를 백그라운드로 시작
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
    print(f"[ui] 웹 UI 실행: http://<Jetson_IP>:{port}")

# 학습 스크립트를 외부 프로세스로 실행
def run_training_now(config_path: Optional[Path], force_type: Optional[str]):
    train_py = (MODEL_DIR / "model" / "train.py" if (MODEL_DIR / "model" / "train.py").exists() else MODEL_DIR / "train.py")
    args = [sys.executable, str(train_py)]
    if config_path is not None: args += ["--config", str(config_path)]
    if force_type in ("lgbm", "centroid"): args += ["--type", force_type]
    print("[train] 시작:", " ".join(args))
    try:
        r = subprocess.run(args, check=False)
        print("[train] 종료 코드={}".format(r.returncode))
    except Exception as e:
        print("[train] 실행 실패:", e)

_RUN_PUMP = False
_PUMP_THREAD = None
_SCAN_BUSY = threading.Event()

# Datamatrix 스캐너 프레임 펌프를 주기적으로 호출
def _start_frame_pump(dm_handle, rois, idle_fps: float = 1.0):
    global _RUN_PUMP, _PUMP_THREAD
    if dm_handle is None:
        print("[pump] dm_handle 없음 → 펌프 미사용"); return
    if idle_fps <= 0:
        print("[pump] idle_fps<=0 → 펌프 비활성화"); return
    _RUN_PUMP = True
    period = 1.0 / float(idle_fps)
    def _loop():
        print(f"[pump] start idle_fps={idle_fps:.2f} (period={period:.3f}s)")
        while _RUN_PUMP:
            if _SCAN_BUSY.is_set():
                time.sleep(period); continue
            try:
                DM.scan_fast4(dm_handle, rois, 0.005, debug=False, trace_id=None)
            except Exception:
                pass
            time.sleep(period)
        print("[pump] stopped")
    _PUMP_THREAD = threading.Thread(target=_loop, daemon=True)
    _PUMP_THREAD.start()

# 프레임 펌프를 중지
def _stop_frame_pump():
    global _RUN_PUMP, _PUMP_THREAD
    _RUN_PUMP = False
    if _PUMP_THREAD is not None:
        try: _PUMP_THREAD.join(timeout=1.0)
        except Exception: pass
        _PUMP_THREAD = None

# 스캔 직전 버스트 호출로 장치 버퍼를 비움
def _burst_flush(dm_handle, rois, n=10, to_sec=0.001):
    try:
        for _ in range(int(n)):
            DM.scan_fast4(dm_handle, rois, float(to_sec), debug=False, trace_id=None)
    except Exception:
        pass

# 설정 객체에 임베딩 기본 옵션을 주입
def _inject_default_embedding_options(S):
    if not hasattr(S, "embedding"):
        class _E: pass
        S.embedding = _E()
    if not hasattr(S.embedding, "concat3"):
        S.embedding.concat3 = bool(int(os.getenv("EMB_CONCAT3", "0")))
    if not hasattr(S.embedding, "mirror_period"):
        S.embedding.mirror_period = int(os.getenv("EMB_MIRROR_PERIOD", "3"))
    if not hasattr(S.embedding, "rois3"):
        S.embedding.rois3 = None

# 단일뷰 또는 3뷰 임베딩을 한 번 수행
def _embed_one_any(emb, S, dm_handle, pregrab=0):
    want_concat3 = bool(getattr(S.embedding, "concat3", False))
    if want_concat3 and hasattr(Emb, "embed_one_frame_shared_concat3"):
        try:
            return Emb.embed_one_frame_shared_concat3(
                emb, S, dm_handle, DM.lock(), pregrab=int(pregrab),
                mirror_period=int(getattr(S.embedding, "mirror_period", 3)),
                rois3=getattr(S.embedding, "rois3", None),
            )
        except Exception as e:
            print("[embed] concat3 경로 예외 → 단일뷰 폴백:", e)
    return Emb.embed_one_frame_shared(emb, S, dm_handle, DM.lock(), pregrab=int(pregrab))

# DM 퍼시스턴트 핸들에서 OpenCV VideoCapture를 추출
def _extract_vcap_from_dm_handle(dm_handle):
    if isinstance(dm_handle, (list, tuple)):
        for item in dm_handle:
            for attr in ("cap", "vcap"):
                if hasattr(item, attr):
                    vcap = getattr(item, attr)
                    if vcap is not None:
                        return vcap
            if hasattr(item, "read") or hasattr(item, "grab"):
                return item
            for holder in ("camera", "cam"):
                if hasattr(item, holder):
                    inner = getattr(item, holder)
                    if inner is None:
                        continue
                    for attr in ("cap", "vcap"):
                        if hasattr(inner, attr):
                            vcap = getattr(inner, attr)
                            if vcap is not None:
                                return vcap
                    if hasattr(inner, "read") or hasattr(inner, "grab"):
                        return inner
    for attr in ("cap", "vcap"):
        if hasattr(dm_handle, attr):
            vcap = getattr(dm_handle, attr)
            if vcap is not None:
                return vcap
    for holder in ("camera", "cam"):
        if hasattr(dm_handle, holder):
            inner = getattr(dm_handle, holder)
            if inner is None:
                continue
            for attr in ("cap", "vcap"):
                if hasattr(inner, attr):
                    vcap = getattr(inner, attr)
                    if vcap is not None:
                        return vcap
            if hasattr(inner, "read") or hasattr(inner, "grab"):
                return inner
    return None

# DM 핸들에서 스냅샷을 뽑아 ROI 크롭 후 메모리에 JPEG로 보관
def _capture_rgb_snapshot_quick(
    S, dm_handle, save_dir: Path, event_id: int, rois_for_capture=None) -> Optional[str]:
    CAPTURE_NAME = os.getenv("CAPTURE_NAME", "capture.jpg")
    CAPTURE_TO_DISK = int(os.getenv("CAPTURE_TO_DISK", "0"))
    JPEG_Q = int(os.getenv("CAPTURE_JPEG_QUALITY", "85"))
    frame = None
    vcap = _extract_vcap_from_dm_handle(dm_handle)
    if vcap is not None:
        try:
            with DM.lock():
                if dm_read_frame is not None:
                    out = dm_read_frame(vcap)
                    if isinstance(out, tuple):
                        ok = bool(out[0])
                        frame = out[1] if ok and len(out) > 1 else None
                    else:
                        frame = out
                else:
                    ok, f = vcap.read()
                    if ok and f is not None:
                        frame = f
        except Exception as e:
            print("[cap] DM vcap 캡처 실패:", e)
            frame = None
    else:
        print("[cap] dm_handle에서 cam/cap을 찾지 못함")
    if frame is None:
        try:
            if not same_device(getattr(S.dm, "camera", 0), getattr(S.embedding, "cam_dev", 0)):
                cap = cv2.VideoCapture(getattr(S.embedding, "cam_dev", 0))
                try:
                    w = int(getattr(S.embedding, "width", 0) or 1280)
                    h = int(getattr(S.embedding, "height", 0) or 720)
                    fps = int(getattr(S.embedding, "fps", 6) or 6)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    cap.set(cv2.CAP_PROP_FPS, fps)
                except Exception:
                    pass
                ok, f = cap.read()
                cap.release()
                if ok and f is not None:
                    frame = f
        except Exception as e:
            print("[cap] embedding 폴백 캡처 실패:", e)
            frame = None
    if frame is None:
        _set_cap_jpeg(None)
        return None
    try:
        H, W = frame.shape[:2]
        rw, rh = 520, 550
        ox, oy = 90, -160
        cx, cy = W // 2, H // 2
        x0 = int(round(cx + ox - rw / 2.0))
        y0 = int(round(cy + oy - rh / 2.0))
        x1 = x0 + rw
        y1 = y0 + rh
        x0 = max(0, min(W, x0)); y0 = max(0, min(H, y0))
        x1 = max(0, min(W, x1)); y1 = max(0, min(H, y1))
        if x1 > x0 and y1 > y0:
            frame = frame[y0:y1, x0:x1]
        else:
            print("[cap] 고정 ROI가 유효 영역을 벗어남 → 원본 저장")
    except Exception as e:
        print("[cap] 고정 ROI 크롭 실패(원본 사용):", e)
    try:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
        if not ok:
            print("[cap] imencode 실패")
            _set_cap_jpeg(None)
            return None
        jpeg_bytes = buf.tobytes()
        _set_cap_jpeg(jpeg_bytes)
        if CAPTURE_TO_DISK:
            fpath = save_dir / CAPTURE_NAME
            try:
                with open(str(fpath), "wb") as f:
                    f.write(jpeg_bytes)
                print("[cap] saved(backup):", fpath)
            except Exception as e:
                print("[cap] backup save 실패:", e)
        return "__mem__"
    except Exception as e:
        print("[cap] encode/save 예외:", e)
        _set_cap_jpeg(None)
        return None

# 엔드투엔드 실행 진입점
def main():
    # 시작 로그와 설정 로드
    t_all = time.time()
    print("[init] start]")
    CFG, CONFIG_PATH = load_config()
    S = get_settings(CFG)
    if not hasattr(S, "dm") and hasattr(S, "datamatrix"):
        S.dm = S.datamatrix

    # 설정 객체 정리와 기본값 보정
    class _NS: pass
    if not hasattr(S, "paths"): S.paths = _NS()
    if not hasattr(S.paths, "db"): S.paths.db = getattr(getattr(S, "storage", _NS()), "sqlite_path", "pack.db")
    if not hasattr(S.paths, "centroids"): S.paths.centroids = getattr(getattr(S, "model", _NS()), "centroids_path", "centroids.npz")
    if not hasattr(S.paths, "lgbm"): S.paths.lgbm = getattr(getattr(S, "model", _NS()), "lgbm_path", "lgbm.pkl")
    if hasattr(S, "model"):
        if not hasattr(S.model, "min_margin"): S.model.min_margin = 0.05
        if not hasattr(S.model, "prob_threshold"): S.model.prob_threshold = 0.55
        if not hasattr(S.model, "smooth_window"): S.model.smooth_window = 3
        if not hasattr(S.model, "smooth_min"): S.model.smooth_min = 2
        if not hasattr(S.model, "topk"): S.model.topk = 3
        if not hasattr(S.model, "type"): S.model.type = "lgbm"

    # 웹 UI 시작과 임베딩 옵션 주입
    _inject_default_embedding_options(S)
    _start_web_ui()

    # 시리얼 포트 연결 시도  스캐너와 커팅기 분리
    ser = None
    ser_cutter = None
    if HAVE_SERIAL:
        SERIAL_PORT = os.getenv("ARDUINO_PORT", "/dev/ttyACM0")
        BAUD_RATE = int(os.getenv("ARDUINO_BAUD", "9600"))
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            print(f"[serial] 스캐너 아두이노 연결 성공: {SERIAL_PORT} (Baud: {BAUD_RATE})")
        except Exception as e:
            print(f"[serial] 스캐너 아두이노 연결 실패: {e}")
            print("[serial] 키보드 'D' 입력은 계속 사용 가능합니다.")
        CUTTER_SERIAL_PORT = os.getenv("CUTTER_ARDUINO_PORT", "/dev/ttyACM1")
        CUTTER_BAUD_RATE = int(os.getenv("CUTTER_ARDUINO_BAUD", "9600"))
        try:
            ser_cutter = serial.Serial(CUTTER_SERIAL_PORT, CUTTER_BAUD_RATE, timeout=1.0)
            print(f"[serial] 커팅기 아두이노 연결 성공: {CUTTER_SERIAL_PORT} (Baud: {CUTTER_BAUD_RATE})")
        except Exception as e:
            print(f"[serial] 커팅기 아두이노 연결 실패: {e}")
    else:
        print("[serial] 'pyserial' 미설치 → 아두이노 연동 생략 (pip3 install pyserial)")

    # 젯슨 성능 모드 준비  OpenCV 워밍업  DB 연결
    maybe_run_jetson_perf()
    warmup_opencv_kernels()
    apply_depth_overrides(depthmod, S)
    conn = Storage.open_db(S.paths.db)

    # RealSense 초기화와 캘리브레이션
    try: roi_px = (int(S.depth.roi_px[0]), int(S.depth.roi_px[1]))
    except Exception: roi_px = (260, 260)
    try: roi_off = (int(S.depth.roi_offset[0]), int(S.depth.roi_offset[1]))
    except Exception: roi_off = (20, -100)
    depth = DepthEstimator(width=S.depth.width, height=S.depth.height, fps=S.depth.fps, roi_px=roi_px, roi_offset=roi_off)
    depth.start()
    frames = depth.warmup(seconds=1.5)
    print("[warmup] RealSense frames={}".format(frames))
    ok_calib = depth.calibrate(max_seconds=3.0)
    if not ok_calib:
        print("[fatal] depth calib 실패. 바닥만 보이게 하고 재실행하세요.")
        try: depth.stop()
        except Exception: pass
        try: conn.close()
        except Exception: pass
        _set_status("error")
        return

    # Datamatrix 퍼시스턴트 핸들 생성
    dm_handle = DM.open_persistent(S.dm.camera, S.dm.prefer_res, S.dm.prefer_fps)

    # 임베더 준비와 워밍업  프레임 펌프 시작
    def _normalize_rois(rois):
        out = []
        if not rois: return out
        for r in rois:
            if isinstance(r, dict):
                name = r.get("name", "ROI"); size = r.get("size", [0, 0]) or [0, 0]; off = r.get("offset", [0, 0]) or [0, 0]; hflip = bool(r.get("hflip", False))
            else:
                name = getattr(r, "name", "ROI"); size = getattr(r, "size", [0, 0]) or [0, 0]; off = getattr(r, "offset", [0, 0]) or [0, 0]; hflip = bool(getattr(r, "hflip", False))
            try: size = [int(size[0]), int(size[1])]; off = [int(off[0]), int(off[1])]
            except Exception: size = [0, 0]; off = [0, 0]
            out.append(dict(name=name, size=size, offset=off, hflip=hflip))
        return out

    def datamatrix_scan_persistent(timeout_s=None, debug=False, trace_id=None):
        to = float(getattr(S.dm, "scan_timeout_s", timeout_s if timeout_s is not None else 2.0))
        rois = _normalize_rois(getattr(S.dm, "rois", []))
        return DM.scan_fast4(dm_handle, rois, to, debug=bool(debug), trace_id=trace_id)

    if same_device(S.dm.camera, S.embedding.cam_dev):
        print("[warn] DM_CAMERA({})와 EMB_CAM_DEV({}) 동일".format(S.dm.camera, S.embedding.cam_dev))
    emb = Emb.build_embedder(S)
    print("[img2emb.cfg] dev={} {}x{}@{} pixfmt={}".format(
        S.embedding.cam_dev, S.embedding.width, S.embedding.height, S.embedding.fps, S.embedding.pixfmt
    ))

    WARMUP_FRAMES = int(getattr(S.embedding, "e2e_warmup_frames", 0) or 0)
    if same_device(S.dm.camera, S.embedding.cam_dev):
        print("[warmup] skip shared e2e warmup (same DM/EMB device)")
        WARMUP_FRAMES = 0

    try:
        if WARMUP_FRAMES > 0:
            Emb.warmup_shared(
                emb, S, dm_handle, DM.lock(), WARMUP_FRAMES, S.embedding.e2e_pregrab
            )
        else:
            print("[warmup] none (frames=0)")
    except Exception as e:
        print("[warmup] skipped due to error:", e)

    IDLE_FPS = float(os.getenv("IDLE_PUMP_FPS", "1.0"))
    rois_for_pump = _normalize_rois(getattr(S.dm, "rois", []))
    _start_frame_pump(dm_handle, rois_for_pump, idle_fps=IDLE_FPS)

    # 추론 엔진 로드와 확률 스무더 구성
    engine = Models.InferenceEngine(S)
    smoother = Models.ProbSmoother(window=int(getattr(S.model, "smooth_window", 3)), min_votes=int(getattr(S.model, "smooth_min", 2)))

    if getattr(S.embedding, "concat3", False):
        print(f"[embed] 3-view concat 모드 활성: mirror_period={int(getattr(S.embedding, 'mirror_period', 3))}")
    else:
        print("[embed] 단일뷰 모드")

    print("[ready] total init %.2fs" % (time.time() - t_all))
    print("[hint] D + Enter 또는 아두이노 신호 = 측정/스캔/추론")
    print("[hint] L + Enter = LGBM 학습")
    print("[hint] C + Enter = Centroid 학습")
    print("[hint] P = 일시정지 토글")
    if HAVE_FLASK: print("[hint] UI: http://localhost:8000")

    # 메인 이벤트 루프 시작  키보드와 아두이노 입력 처리
    paused = False
    EMB_PREGRAB_ON_TRIGGER = int(os.getenv("EMB_PREGRAB_ON_TRIGGER", "0"))

    def send_to_cutter(bubble_length_mm):
        if ser_cutter and ser_cutter.is_open and bubble_length_mm and bubble_length_mm > 0:
            try:
                message = f"B{int(bubble_length_mm)}\n"
                ser_cutter.write(message.encode("utf-8"))
                print(f"[serial] 커팅기 전송: '{message.strip()}' ({bubble_length_mm} mm)")
                return True
            except Exception as e:
                print("[serial] 커팅기 전송 실패:", e)
                return False
        return False

    try:
        DM_TRACE_ID = 0
        while True:
            engine.reload_if_updated()
            trigger_scan = False
            cmd = stdin_readline_nonblock(0.05)
            uc = ""
            if cmd:
                uc = cmd.strip().upper()
                if uc == "D": trigger_scan = True
            if ser and ser.in_waiting > 0:
                try:
                    line = ser.readline().decode("utf-8").strip()
                    if line == "1":
                        print("[serial] 아두이노 '1' → 스캔 트리거(D)"); trigger_scan = True
                    elif line == "2":
                        if not paused: paused = True; _set_status("paused"); print("[serial] 아두이노 '2' → 일시정지 ON")
                        else: print("[serial] '2' 수신: 이미 일시정지 상태")
                    elif line == "3":
                        if paused: paused = False; _set_status(None); print("[serial] 아두이노 '3' → 일시정지 OFF")
                        else: print("[serial] '3' 수신: 이미 동작 중")
                except Exception:
                    pass

            if uc:
                if uc == "L": run_training_now(CONFIG_PATH, force_type="lgbm"); engine.reload_if_updated(); continue
                if uc == "C": run_training_now(CONFIG_PATH, force_type="centroid"); engine.reload_if_updated(); continue
                if uc == "T": print("[hint] 이제는 L/C로 모델별 학습이 가능합니다. (L=LGBM, C=Centroid)"); run_training_now(CONFIG_PATH, force_type=None); engine.reload_if_updated(); continue
                if uc == "P":
                    paused = not paused
                    if paused: _set_status("paused"); print("[state] 일시정지 ON: D/아두이노 입력 무시")
                    else: _set_status(None); print("[state] 일시정지 OFF")
                    continue

            # 스캔 트리거 시 스냅샷  버스트 플러시  DM 스캔
            if trigger_scan:
                if paused:
                    print("[state] 일시정지 상태. D/아두이노 신호 무시"); continue
                DM_TRACE_ID += 1
                tid = DM_TRACE_ID
                t_press = t_now()
                print(f"[D#{tid}][{ts_wall()}] key_down → scan_call")
                _set_status("analyzing"); _set_prob(None); _set_event_id(tid)

                _SCAN_BUSY.set()
                try:
                    cap_name = _capture_rgb_snapshot_quick(S, dm_handle, IMG_DIR, tid)
                    _set_cap_image(cap_name)
                except Exception as e:
                    print("[cap] 예외:", e); _set_cap_image(None)

                try:
                    BURST_N = int(os.getenv("BURST_FLUSH_N", "10"))
                    BURST_TO_MS = int(os.getenv("BURST_FLUSH_TO_MS", "1"))
                    n_flush = BURST_N if BURST_N > 0 else 10
                    t_flush = (BURST_TO_MS if BURST_TO_MS > 0 else 1) / 1000.0
                    _burst_flush(dm_handle, rois_for_pump, n=n_flush, to_sec=t_flush)
                except Exception:
                    pass

                try:
                    t_s0 = t_now()
                    payload = datamatrix_scan_persistent(None, debug=False, trace_id=tid)
                    t_s1 = t_now()
                    print(f"[D#{tid}][{ts_wall()}] scan_return elapsed={ms(t_s1 - t_s0)} ms Tpress→scan_return={ms(t_s1 - t_press)} ms payload={'YES' if payload else 'NO'}")
                    if payload: print(f"[dm] payload={payload}")
                    else: print("[dm] payload 없음")

                    # 라벨 성공 시 치수 반영과 커팅 전송  로그 저장
                    if payload and _set_ui_from_label(payload):
                        _set_warn(None); _set_prob(None); _set_event_id(tid); _set_status("done")
                        with _UI_LOCK: final_bubble_mm = _UI_STATE.get("bubble_mm")
                        send_to_cutter(final_bubble_mm)
                        t_depth0 = t_now(); feat = depth.measure_dimensions(duration_s=1.0, n_frames=10); t_depth1 = t_now()
                        print(f"[D#{tid}][{ts_wall()}] depth_measure elapsed={ms(t_depth1 - t_depth0)}")
                        if feat is None:
                            print("[manual] 측정 실패"); _SCAN_BUSY.clear(); continue
                        t_emb0 = t_now(); vec = _embed_one_any(emb, S, dm_handle, pregrab=EMB_PREGRAB_ON_TRIGGER); t_emb1 = t_now()
                        print(f"[D#{tid}][{ts_wall()}] embed_one_frame elapsed={ms(t_emb1 - t_emb0)}")
                        if vec is None:
                            print("[manual] 임베딩 실패"); _SCAN_BUSY.clear(); continue
                        if feat["q"] < float(getattr(S.quality, "q_warn", 0.30)):
                            print(f"[notify] 품질 경고: q={feat['q']:.2f} (임계 {float(getattr(S.quality, 'q_warn', 0.30)):.2f})")
                        Storage.on_sample_record(conn, feat, vec, product_id=payload, has_label=1, origin="manual_dm")
                        smoother.buf.clear()
                        print("[infer] skip: DM 라벨 확정 → 모델 추론 생략")
                        _SCAN_BUSY.clear()
                        continue
                    else:
                        print("[ui] DM 라벨 없음/형식 불일치 → 모델 라벨에서 파싱 예정")

                    # 라벨 실패 시 depth 측정과 임베딩  모델 추론  스무딩
                    t_depth0 = t_now(); feat = depth.measure_dimensions(duration_s=1.0, n_frames=10); t_depth1 = t_now()
                    print(f"[D#{tid}][{ts_wall()}] depth_measure elapsed={ms(t_depth1 - t_depth0)}")
                    if feat is None:
                        print("[manual] 측정 실패"); _set_status("error"); _SCAN_BUSY.clear(); continue
                    t_emb0 = t_now(); vec = _embed_one_any(emb, S, dm_handle, pregrab=EMB_PREGRAB_ON_TRIGGER); t_emb1 = t_now()
                    print(f"[D#{tid}][{ts_wall()}] embed_one_frame elapsed={ms(t_emb1 - t_emb0)}")
                    if vec is None:
                        print("[manual] 임베딩 실패"); _set_status("error"); _SCAN_BUSY.clear(); continue
                    if feat["q"] < float(getattr(S.quality, "q_warn", 0.30)):
                        print(f"[notify] 품질 경고: q={feat['q']:.2f} (임계 {float(getattr(S.quality, 'q_warn', 0.30)):.2f})")
                    Storage.on_sample_record(conn, feat, vec, product_id=None, has_label=0, origin="manual_no_dm")
                    meta = np.array(
                        [feat["d1"], feat["d2"], feat["d3"], feat["mad1"], feat["mad2"], feat["mad3"],
                         feat["r1"], feat["r2"], feat["r3"], feat["sr1"], feat["sr2"], feat["sr3"],
                         feat["logV"], feat["logsV"], feat["q"]], np.float32)
                    full_vec = np.concatenate([meta, vec], axis=0)
                    full_vec = l2_normalize(full_vec)
                    print(f"[vector] dim={full_vec.shape[0]} (meta 15 + emb {vec.shape[0]})")
                    print(f"[debug] ||full_vec||={np.linalg.norm(full_vec):.6f}")
                    top_lab, top_p, gap, backend = engine.infer(full_vec)
                    if backend is None:
                        print("[infer] 모델 없음(파일 미존재 또는 로드 실패)"); _set_prob(None); _set_status("error"); _SCAN_BUSY.clear(); continue
                    print(f"[infer] {backend} top1: {top_lab} p={top_p:.3f} gap={gap:.4f}")
                    _set_prob(top_p)
                    prob_th = float(getattr(S.model, "prob_threshold", 0.55))
                    low_prob = top_p < prob_th
                    updated = _set_ui_from_label(top_lab)
                    _set_event_id(tid)
                    if low_prob: _set_warn("Warning! Percent is very low"); _set_status("error")
                    else: _set_warn(None); _set_status("done")
                    with _UI_LOCK: final_bubble_mm = _UI_STATE.get("bubble_mm")
                    send_to_cutter(final_bubble_mm)
                    if not updated:
                        print("[ui] 모델 라벨 파싱 실패(16자리 규격 아님)")
                    min_margin = float(getattr(S.model, "min_margin", 0.05))
                    if gap < min_margin:
                        smoother.push(top_lab, top_p)
                        print(f"[smooth] hold: small_margin gap={gap:.4f} (<{min_margin:.3f}), len={len(smoother.buf)}/{S.model.smooth_window}, top={top_lab} p={top_p:.2f}")
                        decided = smoother.maybe_decide(threshold=prob_th)
                        if decided is not None:
                            lab, avgp = decided; print(f"[decision] smoothed: {lab} p={avgp:.2f}")
                    elif low_prob:
                        smoother.push(top_lab, top_p)
                        print(f"[smooth] hold: len={len(smoother.buf)}/{S.model.smooth_window} , top={top_lab} p={top_p:.2f} (<{prob_th:.2f})")
                        decided = smoother.maybe_decide(threshold=prob_th)
                        if decided is not None:
                            lab, avgp = decided; print(f"[decision] smoothed: {lab} p={avgp:.2f}")
                    else:
                        smoother.push(top_lab, top_p)
                        decided = smoother.maybe_decide(threshold=prob_th)
                        if decided is None:
                            status = smoother.status()
                            if status[0] is None:
                                print(f"[smooth] hold: len={len(smoother.buf)}/{S.model.smooth_window}")
                            else:
                                lab, votes, avgp = status
                                print(f"[smooth] hold: len={len(smoother.buf)}/{S.model.smooth_window} , lead={lab} votes={votes} avgp={avgp:.2f}")
                        else:
                            lab, avgp = decided; print(f"[decision] smoothed: {lab} p={avgp:.2f}")
                except Exception as e:
                    print("[error] D 처리 중 예외:", e)
                    _set_status("error"); _set_warn("오류가 발생했습니다."); _set_prob(None)
                finally:
                    _SCAN_BUSY.clear()

    except KeyboardInterrupt:
        print("[exit] keyboard interrupt")
    finally:
        # 종료 및 정리  장치 리소스 해제와 DB 최적화
        _stop_frame_pump()
        if ser and ser.is_open:
            ser.close(); print("[cleanup] 스캐너 시리얼 포트가 닫혔습니다.")
        if ser_cutter and ser_cutter.is_open:
            ser_cutter.close(); print("[cleanup] 커팅기 시리얼 포트가 닫혔습니다.")
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)"); conn.execute("PRAGMA optimize")
        except Exception:
            pass
        try: depth.stop()
        except Exception: pass
        try: conn.close()
        except Exception: pass
        try: DM.close_persistent()
        except Exception: pass
        print("[cleanup] stopped")

if __name__ == "__main__":
    main()