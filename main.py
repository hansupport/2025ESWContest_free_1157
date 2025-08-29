# main.py (refactored with core.lite, CUDA warmup removed)
# - YAML/JSON 설정 로딩
# - depth/img2emb/datamatrix에 설정값 주입
# - SQLite 로깅(모든 샘플 기록)
# - D+Enter: DataMatrix 스캔(빠른 4방향만) + 1초 치수 측정 + 임베딩
# - L+Enter: LGBM만 학습 트리거 (--type lgbm)
# - C+Enter: Centroid만 학습 트리거 (--type centroid)
# - DM 카메라 persistent 공유 + 락
# - 모델: centroid + LGBM 동시 지원(파일 변경 자동 리로드, config.model.type로 추론 우선순위 선택)
# - 추론: 학습과 동일한 L2정규화, top-1 확신도 임계치/마진 + 3~5프레임 스무딩
# - Flask 웹 UI(1024x600)로 W/L/H와 뽁뽁이 길이 표시 (http://localhost:8000)
# - Flask 액세스 로그 무음 + Waitress 사용(설치 시)
# - DM이 없으면 모델 라벨(16자리)에서 W/L/H 파싱해 즉시 표시
# - p<0.40이면 UI에 빨간 경고 표시 ("Warning! Percent is very low")
# - 경고만 바뀌어도 updated_ts 증가 + /api/state 캐시 금지
import sys
import time
import subprocess
import threading
import json
import numpy as np
from pathlib import Path
from typing import Optional  # Python 3.6 호환
import cv2

# ---- 웹 UI 준비: Flask는 선택적 의존성 ----
HAVE_FLASK = True
try:
    from flask import Flask, jsonify, render_template_string, request, make_response
except Exception:
    HAVE_FLASK = False

# ---- Waitress 사용 가능 여부 ----
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

# 로컬 모듈
from depth import DepthEstimator
import depth as depthmod

# core 패키지 (집약 래퍼)
from core.config import load_config, get_settings, apply_depth_overrides
from core.lite import DM, Emb, Models, Storage
from core.utils import (
    ts_wall, t_now, ms, stdin_readline_nonblock,
    maybe_run_jetson_perf, warmup_opencv_kernels,
    l2_normalize, same_device
)

# 터미널 I/O 부담 완화: 과도한 배열 출력 금지
np.set_printoptions(suppress=True, linewidth=200, threshold=50, precision=4)

# ---- UI 공유 상태 ----
_UI_LOCK = threading.Lock()
_UI_STATE = {
    "W": None,    # mm
    "L": None,    # mm
    "H": None,    # mm
    "bubble_mm": None,
    "updated_ts": None,
    "warn_msg": None,   # p<0.40일 때 경고 문구
    "params": {
        "layers": 2,         # 감쌀 층수
        "overlap_mm": 120,   # 겹침/여유
        "slack_ratio": 0.03, # 여유 3%
        "round_to_mm": 10    # 10mm 단위 올림
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
    layers = p["layers"] if layers is None else layers
    overlap_mm = p["overlap_mm"] if layers is None else p["overlap_mm"] if overlap_mm is None else overlap_mm
    overlap_mm = p["overlap_mm"] if overlap_mm is None else overlap_mm
    slack_ratio = p["slack_ratio"] if slack_ratio is None else slack_ratio
    round_to_mm = p["round_to_mm"] if round_to_mm is None else round_to_mm

    perimeter = 2.0 * (float(W) + float(L))
    base = perimeter * float(layers) + float(overlap_mm)
    total = base * (1.0 + float(slack_ratio))
    return _round_up(total, round_to_mm)

def _touch_ts():
    with _UI_LOCK:
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
        _UI_STATE["updated_ts"] = int(time.time())  # 경고만 바뀌어도 TS 갱신

# ---- DM/라벨 파싱 ----
def parse_dm_dimensions(s):
    """
    16자리 숫자 문자열을 W,L,H(mm)로 파싱.
    형식: [0:6]=종류(무시), [6:9]=W, [9:12]=L, [12:15]=H, [15]=재질(무시)
    """
    if s is None:
        return None
    s = str(s).strip()
    if len(s) == 16 and s.isdigit():
        try:
            w = int(s[6:9]); l = int(s[9:12]); h = int(s[12:15])
            return (w, l, h)
        except Exception:
            return None
    return None

def _set_ui_dims_from_label(lab):
    dims = parse_dm_dimensions(lab)
    if dims is None:
        return False
    w, l, h = dims
    _set_ui_dims_and_bubble(w, l, h)
    with _UI_LOCK:
        bb = _UI_STATE['bubble_mm']
    print(f"[ui] MODEL dims 사용: W={w} L={l} H={h} mm → bubble={bb} mm")
    return True

# ---- Flask 앱 & 무음 로그 설정 ----
_SilentWSGIRequestHandler = None  # 기본값

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

    _HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>포장 계산기</title>
  <meta name="viewport" content="width=1024, height=600, initial-scale=1, user-scalable=no">
  <style>
    :root { --bg:#0b0b0f; --fg:#e8f1ff; --accent:#60a5fa; --muted:#9aa4b2; }
    html, body { margin:0; height:100%; background:var(--bg); color:var(--fg); font-family:system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
    .wrap { width:1024px; height:600px; margin:0 auto; display:flex; flex-direction:column; }
    header { padding:14px 20px; font-weight:700; letter-spacing:0.3px; color:#cbd5e1; border-bottom:1px solid #111827; }
    main { flex:1; display:flex; gap:24px; padding:24px; box-sizing:border-box; }
    .card { flex:1; background:#0f172a; border:1px solid #111827; border-radius:16px; padding:20px; display:flex; flex-direction:column; justify-content:center; align-items:center; box-shadow:0 0 0 1px rgba(255,255,255,0.02) inset; }
    .label { font-size:18px; color:var(--muted); margin-bottom:12px; }
    .value { font-size:56px; font-weight:800; letter-spacing:1px; }
    .warn { 
      display:none; 
      margin-top:12px; 
      font-size:28px; 
      font-weight:900; 
      color:#ef4444; 
      background:rgba(239,68,68,0.16); 
      border:2px solid #ef4444; 
      border-radius:12px; 
      padding:10px 14px; 
      text-align:center;
      text-shadow:0 0 6px rgba(239,68,68,0.5);
      animation:pulse 1.6s infinite ease-in-out;
    }
    @keyframes pulse {
      0% { box-shadow:0 0 0 0 rgba(239,68,68,0.45); }
      70%{ box-shadow:0 0 0 12px rgba(239,68,68,0); }
      100%{ box-shadow:0 0 0 0 rgba(239,68,68,0); }
    }
    .sub { margin-top:10px; font-size:16px; color:#a1a1aa; }
    .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:24px; width:100%; }
    footer { padding:10px 20px; font-size:14px; color:#94a3b8; display:flex; justify-content:space-between; border-top:1px solid #111827; }
    .hint { color:#64748b; }
  </style>
</head>
<body>
  <div class="wrap">
    <header>포장 계산기 · 1024×600 · <span class="hint">D 키로 측정 갱신</span></header>
    <main>
      <div class="card" style="flex:1.6">
        <div class="label">필요 뽁뽁이 길이</div>
        <div class="value" id="bubble">— mm</div>
        <div class="sub" id="bubble_cm">— cm</div>
        <div class="warn" id="warn"></div>
        <div class="sub" id="params">계산식: 2×(W+L)×층수 + overlap + slack</div>
      </div>
      <div class="card" style="flex:2.4">
        <div class="grid">
          <div class="card">
            <div class="label">가로 (W)</div>
            <div class="value" id="w">— mm</div>
            <div class="sub" id="w_cm">— cm</div>
          </div>
          <div class="card">
            <div class="label">세로 (L)</div>
            <div class="value" id="l">— mm</div>
            <div class="sub" id="l_cm">— cm</div>
          </div>
          <div class="card">
            <div class="label">높이 (H)</div>
            <div class="value" id="h">— mm</div>
            <div class="sub" id="h_cm">— cm</div>
          </div>
        </div>
      </div>
    </main>
    <footer>
      <div>업데이트: <span id="ts">—</span></div>
      <div id="note" class="hint">층수/여유 변경: <code>/api/params?layers=2&overlap_mm=120</code></div>
    </footer>
  </div>
  <script>
    function mmToCm(x){ if(x==null) return "—"; return (x/10).toFixed(1); }
    function tsFmt(t){ if(!t) return "—"; const d = new Date(t*1000); return d.toLocaleString(); }
    function setText(id, txt){ const el=document.getElementById(id); if(el) el.textContent = txt; }
    function setWarn(msg){
      const el = document.getElementById('warn');
      if(!el) return;
      if(msg && msg.length){
        el.textContent = msg;
        el.style.display = 'block';
      }else{
        el.textContent = '';
        el.style.display = 'none';
      }
    }

    async function tick(){
      try{
        const r = await fetch('/api/state', {cache:'no-store'});
        const s = await r.json();
        setText('w', s.W!=null ? s.W+' mm' : '— mm');
        setText('l', s.L!=null ? s.L+' mm' : '— mm');
        setText('h', s.H!=null ? s.H+' mm' : '— mm');
        setText('w_cm', s.W!=null ? mmToCm(s.W)+' cm' : '— cm');
        setText('l_cm', s.L!=null ? mmToCm(s.L)+' cm' : '— cm');
        setText('h_cm', s.H!=null ? mmToCm(s.H)+' cm' : '— cm');
        setText('bubble', s.bubble_mm!=null ? s.bubble_mm+' mm' : '— mm');
        setText('bubble_cm', s.bubble_mm!=null ? mmToCm(s.bubble_mm)+' cm' : '— cm');
        setText('ts', tsFmt(s.updated_ts));
        const p = s.params || {};
        document.getElementById('params').textContent =
          `계산식: 2×(W+L)×${p.layers||'?'}층 + overlap(${p.overlap_mm||'?'}mm) + slack(${Math.round((p.slack_ratio||0)*100)}%)`;
        setWarn(s.warn_msg);
      }catch(e){ /* noop */ }
    }
    setInterval(tick, 500);
    tick();
  </script>
</body>
</html>
    """

    @_APP.route("/")
    def home():
        return render_template_string(_HTML)

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

def _start_web_ui():
    if not HAVE_FLASK:
        print("[ui] Flask가 설치되어 있지 않아 UI를 비활성화합니다. (pip install Flask)")
        return
    def _run():
        if HAVE_WAITRESS:
            print("[ui] Serving with Waitress (threads=2)")
            _waitress_serve(_APP, host="0.0.0.0", port=8000, threads=2, ident=None)
        else:
            print("[ui] Serving with Flask dev server (silent handler)")
            kwargs = dict(host="0.0.0.0", port=8000, threaded=True, debug=False, use_reloader=False)
            try:
                if _SilentWSGIRequestHandler is not None:
                    kwargs["request_handler"] = _SilentWSGIRequestHandler
            except Exception:
                pass
            _APP.run(**kwargs)
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    print("[ui] 웹 UI 실행: http://localhost:8000  (Chromium 1024x600 권장)")

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

    # 설정
    CFG, CONFIG_PATH = load_config()
    S = get_settings(CFG)

    # ---- 호환 셋업: datamatrix -> dm alias / paths 채우기 / 모델 디폴트 ----
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

    # 웹 UI 시작
    _start_web_ui()

    # 경량 웜업 (CUDA 불필요)
    maybe_run_jetson_perf()
    warmup_opencv_kernels()

    # depth 파라미터 오버라이드
    apply_depth_overrides(depthmod, S)

    # DB
    conn = Storage.open_db(S.paths.db)

    # DepthEstimator 구성값 주입(roi_px+roi_offset)
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
        return

    # DataMatrix persistent open
    dm_handle = DM.open_persistent(S.dm.camera, S.dm.prefer_res, S.dm.prefer_fps)

    # ---- 예전 API 호환 래퍼 + ROI 정규화 + 기본 타임아웃 ----
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

    # 모델 엔진 + 스무더
    engine = Models.InferenceEngine(S)
    smoother = Models.ProbSmoother(
        window=int(getattr(S.model, 'smooth_window', 3)),
        min_votes=int(getattr(S.model, 'smooth_min', 2))
    )

    print("[ready] total init %.2fs" % (time.time()-t_all))
    print("[hint] D + Enter = 수동 측정 1초 & DataMatrix 스캔")
    print("[hint] L + Enter = LGBM 학습(model/train.py --type lgbm)")
    print("[hint] C + Enter = Centroid 학습(model/train.py --type centroid)")
    if HAVE_FLASK:
        print("[hint] UI: http://localhost:8000  (Chromium 1024x600)")

    try:
        DM_TRACE_ID = 0
        while True:
            # 핫리로드
            engine.reload_if_updated()

            # 입력
            cmd = stdin_readline_nonblock(0.05)
            if not cmd:
                continue

            uc = cmd.strip().upper()
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

            if uc == "D":
                DM_TRACE_ID += 1
                tid = DM_TRACE_ID
                t_press = t_now()
                print(f"[D#{tid}][{ts_wall()}] key_down → scan_call")

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

                # (A) DM에서 치수 먼저 시도 → 성공 시 즉시 UI 업데이트 + 경고 해제
                dm_dims = parse_dm_dimensions(payload) if payload else None
                if dm_dims is not None:
                    W_mm, L_mm, H_mm = dm_dims
                    _set_ui_dims_and_bubble(W_mm, L_mm, H_mm)
                    _set_warn(None)  # DM은 신뢰하므로 경고 제거
                    with _UI_LOCK:
                        bb = _UI_STATE['bubble_mm']
                    print(f"[ui] DM dims 사용: W={W_mm} L={L_mm} H={H_mm} mm → bubble={bb} mm")
                else:
                    print("[ui] DM 치수 없음/형식 불일치 → 모델 라벨에서 치수 파싱 예정")

                # 2) depth 측정 (표시는 하지 않음; 메타/학습용)
                t_depth0 = t_now()
                feat = depth.measure_dimensions(duration_s=1.0, n_frames=10)
                t_depth1 = t_now()
                print(f"[D#{tid}][{ts_wall()}] depth_measure elapsed={ms(t_depth1 - t_depth0)}")
                if feat is None:
                    print("[manual] 측정 실패"); continue

                # 3) 임베딩 1프레임
                t_emb0 = t_now()
                vec = Emb.embed_one_frame_shared(emb, S, dm_handle, DM.lock(), pregrab=3)
                t_emb1 = t_now()
                print(f"[D#{tid}][{ts_wall()}] embed_one_frame elapsed={ms(t_emb1 - t_emb0)}")
                if vec is None:
                    print("[manual] 임베딩 실패"); continue

                # 4) 품질 경고 (UI와 무관)
                if feat["q"] < float(getattr(S.quality, 'q_warn', 0.30)):
                    print(f"[notify] 품질 경고: q={feat['q']:.2f} (임계 {float(getattr(S.quality, 'q_warn', 0.30)):.2f})")

                # 5) 저장
                if payload:
                    Storage.on_sample_record(conn, feat, vec, product_id=payload, has_label=1, origin="manual_dm")
                    smoother.buf.clear()
                    print("[infer] skip: DataMatrix 라벨 확정 → 모델 추론 생략")
                    continue
                else:
                    Storage.on_sample_record(conn, feat, vec, product_id=None, has_label=0, origin="manual_no_dm")

                # === full_vec(15+128) 구성 ===
                meta = np.array([
                    feat["d1"], feat["d2"], feat["d3"],
                    feat["mad1"], feat["mad2"], feat["mad3"],
                    feat["r1"], feat["r2"], feat["r3"],
                    feat["sr1"], feat["sr2"], feat["sr3"],
                    feat["logV"], feat["logsV"], feat["q"]
                ], np.float32)
                full_vec = np.concatenate([meta, vec], axis=0)
                full_vec = l2_normalize(full_vec)
                print(f"[debug] ||full_vec||={np.linalg.norm(full_vec):.6f}")

                # ---- 추론 ----
                top_lab, top_p, gap, backend = engine.infer(full_vec)
                if backend is None:
                    print("[infer] 모델 없음(파일 미존재 또는 로드 실패)"); continue
                print(f"[infer] {backend} top1: {top_lab} p={top_p:.3f} gap={gap:.4f}")

                # ★ DM 실패 시: 임계치/스무딩과 무관하게 '즉시' 라벨로 UI 갱신 + 경고 처리
                if dm_dims is None:
                    updated = _set_ui_dims_from_label(top_lab)
                    prob_th = float(getattr(S.model, 'prob_threshold', 0.40))
                    if top_p < prob_th:
                        _set_warn("Warning! Percent is very low")
                    else:
                        _set_warn(None)
                    if not updated:
                        print("[ui] 모델 라벨에서 치수 파싱 실패(16자리 규격 아님)")

                # ---- 임계/마진/스무딩 (로깅/의사결정용, UI는 이미 갱신됨) ----
                min_margin = float(getattr(S.model, 'min_margin', 0.05))
                prob_th    = float(getattr(S.model, 'prob_threshold', 0.40))

                if gap < min_margin:
                    smoother.push(top_lab, top_p)
                    print(f"[smooth] hold: small_margin gap={gap:.4f} (<{min_margin:.3f}), "
                          f"len={len(smoother.buf)}/{S.model.smooth_window}, top={top_lab} p={top_p:.2f}")
                    decided = smoother.maybe_decide(threshold=prob_th)
                    if decided is not None and dm_dims is None:
                        lab, avgp = decided
                        print(f"[decision] smoothed: {lab} p={avgp:.2f}")
                    continue

                if top_p < prob_th:
                    smoother.push(top_lab, top_p)
                    print(f"[smooth] hold: len={len(smoother.buf)}/{S.model.smooth_window}, "
                          f"top={top_lab} p={top_p:.2f} (<{prob_th:.2f})")
                    decided = smoother.maybe_decide(threshold=prob_th)
                    if decided is not None and dm_dims is None:
                        lab, avgp = decided
                        print(f"[decision] smoothed: {lab} p={avgp:.2f}")
                    continue

                smoother.push(top_lab, top_p)
                decided = smoother.maybe_decide(threshold=prob_th)
                if decided is None:
                    lab, votes, avgp = smoother.status()
                    if lab is None:
                        print(f"[smooth] hold: len={len(smoother.buf)}/{S.model.smooth_window}")
                    else:
                        print(f"[smooth] hold: len={len(smoother.buf)}/{S.model.smooth_window}, "
                              f"lead={lab} votes={votes} avgp={avgp:.2f}")
                else:
                    lab, avgp = decided
                    print(f"[decision] smoothed: {lab} p={avgp:.2f}")
                continue

    except KeyboardInterrupt:
        print("[exit] keyboard interrupt")
    finally:
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
