# main.py (refactored with core.lite)
# - YAML/JSON 설정 로딩
# - depth/img2emb/datamatrix에 설정값 주입
# - SQLite 로깅(모든 샘플 기록)
# - D+Enter: DataMatrix 스캔(빠른 4방향만) + 1초 치수 측정 + 임베딩
# - L+Enter: LGBM만 학습 트리거 (--type lgbm)
# - C+Enter: Centroid만 학습 트리거 (--type centroid)
# - DM 카메라 persistent 공유 + 락
# - 모델: centroid + LGBM 동시 지원(파일 변경 자동 리로드, config.model.type로 추론 우선순위 선택)
# - 추론: 학습과 동일한 L2정규화, top-1 확신도 임계치/마진 + 3~5프레임 스무딩

import sys
import time
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional  # Python 3.6 호환
import cv2

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
    maybe_run_jetson_perf, warmup_opencv_kernels, warmup_torch_cuda,
    l2_normalize, same_device
)

# 터미널 I/O 부담 완화: 과도한 배열 출력 금지
np.set_printoptions(suppress=True, linewidth=200, threshold=50, precision=4)

def run_training_now(config_path: Optional[Path], force_type: Optional[str]):
    train_py = MODEL_DIR / "train.py"
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

    maybe_run_jetson_perf()
    warmup_opencv_kernels()
    warmup_torch_cuda()

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

    # 임베더
    if same_device(S.dm.camera, S.embedding.cam_dev):
        print(f"[warn] DM_CAMERA({S.dm.camera})와 EMB_CAM_DEV({S.embedding.cam_dev}) 동일. shared persistent handle + lock 사용")
    emb = Emb.build_embedder(S)
    print(f"[img2emb.cfg] dev={S.embedding.cam_dev} {S.embedding.width}x{S.embedding.height}@{S.embedding.fps} pixfmt={S.embedding.pixfmt}")

    # e2e warmup
    Emb.warmup_shared(emb, S, dm_handle, DM.lock(), S.embedding.e2e_warmup_frames, S.embedding.e2e_pregrab)

    # 모델 엔진 + 스무더
    engine = Models.InferenceEngine(S)
    smoother = Models.ProbSmoother(window=S.model.smooth_window, min_votes=S.model.smooth_min)

    print("[ready] total init %.2fs" % (time.time()-t_all))
    print("[hint] D + Enter = 수동 측정 1초 & DataMatrix 스캔")
    print("[hint] L + Enter = LGBM 학습(model/train.py --type lgbm)")
    print("[hint] C + Enter = Centroid 학습(model/train.py --type centroid)")

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
                engine.reload_if_updated()
                continue
            if uc == "C":
                run_training_now(CONFIG_PATH, force_type="centroid")
                engine.reload_if_updated()
                continue
            if uc == "T":
                print("[hint] 이제는 L/C로 모델별 학습이 가능합니다. (L=LGBM, C=Centroid)")
                run_training_now(CONFIG_PATH, force_type=None)
                engine.reload_if_updated()
                continue

            if uc == "D":
                DM_TRACE_ID += 1
                tid = DM_TRACE_ID
                t_press = t_now()
                print(f"[D#{tid}][{ts_wall()}] key_down → scan_call")

                # 1) DM 스캔 (디버그 로그는 강제 OFF로 입력 지연 최소화)
                t_s0 = t_now()
                payload = DM.scan_fast4(
                    dm_handle, S.dm.rois, float(S.dm.scan_timeout_s),
                    debug=False,  # 콘솔 스팸 방지
                    trace_id=tid
                )
                t_s1 = t_now()
                print(f"[D#{tid}][{ts_wall()}] scan_return elapsed={ms(t_s1 - t_s0)} "
                      f"Tpress→scan_return={ms(t_s1 - t_press)} payload={'YES' if payload else 'NO'}")
                if payload:
                    print(f"[dm] payload={payload}")
                else:
                    print("[dm] payload 없음")

                # 2) depth 측정
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

                # 4) 품질 경고
                if feat["q"] < float(S.quality.q_warn):
                    print(f"[notify] 품질 경고: q={feat['q']:.2f} (임계 {float(S.quality.q_warn):.2f})")

                # 5) 저장
                if payload:
                    Storage.on_sample_record(conn, feat, vec, product_id=payload, has_label=1, origin="manual_dm")
                    smoother.buf.clear()  # 라벨 확정 → 스무딩 초기화
                else:
                    Storage.on_sample_record(conn, feat, vec, product_id=None, has_label=0, origin="manual_no_dm")

                # === 학습과 동일한 L2 정규화로 full_vec(15+128) 구성 ===
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
                    print("[infer] 모델 없음(파일 미존재 또는 로드 실패)")
                    continue
                print(f"[infer] {backend} top1: {top_lab} p={top_p:.3f} gap={gap:.4f}")

                # ---- 임계/마진/스무딩 ----
                if gap < float(S.model.min_margin):
                    smoother.push(top_lab, top_p)
                    print(f"[smooth] hold: small_margin gap={gap:.4f} (<{float(S.model.min_margin):.3f}), "
                          f"len={len(smoother.buf)}/{S.model.smooth_window}, top={top_lab} p={top_p:.2f}")
                    continue

                if top_p < float(S.model.prob_threshold):
                    smoother.push(top_lab, top_p)
                    print(f"[smooth] hold: len={len(smoother.buf)}/{S.model.smooth_window}, "
                          f"top={top_lab} p={top_p:.2f} (<{float(S.model.prob_threshold):.2f})")
                    decided = smoother.maybe_decide(threshold=float(S.model.prob_threshold))
                    if decided is not None:
                        lab, avgp = decided
                        print(f"[decision] smoothed: {lab} p={avgp:.2f}")
                    continue

                # 충분히 높으면 즉시 or 스무더 통과
                smoother.push(top_lab, top_p)
                decided = smoother.maybe_decide(threshold=float(S.model.prob_threshold))
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
