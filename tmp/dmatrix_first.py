# dmatrix_tick_print_every_0_2s_optimized.py
# 0.2초마다 O/X 출력 + "없는 경우" 가속(부재 필터/다운스케일/timeout) + 동적 백오프 + 타이밍 로그

import time, sys, numpy as np, cv2 as cv

TICK = 0.20                 # O/X 출력 주기
LOG_EVERY_DECODE = 1        # 디코딩 시도 로그 주기(N번마다 1회). 로그 줄이고 싶으면 5~10으로
MAX_BACKOFF = 1.0           # 디코딩 백오프 최대 간격(초)
ROI_W, ROI_H = 530, 420     # 고정 ROI 크기
DX, DY = 120, -100          # ROI 오프셋(화면 중앙 기준)

def log(msg: str):
    """stderr로 타임스탬프 포함 로깅(프로파일/디버그용)"""
    ts = time.strftime("%H:%M:%S", time.localtime())
    sys.stderr.write(f"[{ts}] {msg}\n")
    sys.stderr.flush()

# ---- pylibdmtx 안전 임포트 ----
def load_dmtx_decode():
    t0 = time.perf_counter()
    try:
        from pylibdmtx.pylibdmtx import decode
        log(f"pylibdmtx import(ok) in {(time.perf_counter()-t0)*1000:.2f} ms")
        return decode
    except Exception:
        pass
    try:
        t1 = time.perf_counter()
        import pylibdmtx.pylibdmtx as dmtx
        log(f"pylibdmtx fallback import(ok) in {(time.perf_counter()-t1)*1000:.2f} ms")
        return dmtx.decode
    except Exception as e:
        log(f"pylibdmtx import FAIL in {(time.perf_counter()-t0)*1000:.2f} ms")
        raise ImportError(
            "pylibdmtx decode 로드 실패: " + str(e) + "\n(참고: sudo apt install libdmtx0a libdmtx-dev 후 "
            "pip install --no-binary :all: pylibdmtx==0.1.10)"
        )

dm_decode = load_dmtx_decode()

# OpenCV 최적화 (Jetson 권장)
try:
    cv.setUseOptimized(True); cv.setNumThreads(1)
    log("OpenCV optimized=True, threads=1")
except Exception:
    pass

# ---- 카메라 열기 (RealSense 우선) ----
def open_camera():
    t_all0 = time.perf_counter()
    try:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        tried = []
        for w, h, fps in [(1920,1080,6), (1920,1080,15)]:
            t0 = time.perf_counter()
            try:
                config.disable_all_streams()
                config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
                profile = pipeline.start(config)
                dev = profile.get_device()
                color_sensor = dev.first_color_sensor()
                try:
                    color_sensor.set_option(rs.option.frames_queue_size, 2)
                except Exception:
                    pass
                dt_ms = (time.perf_counter() - t0) * 1000
                log(f"RealSense start {w}x{h}@{fps} in {dt_ms:.2f} ms")
                log(f"Camera total open time {(time.perf_counter()-t_all0)*1000:.2f} ms")
                return ("realsense", (pipeline, rs))
            except Exception as e:
                dt_ms = (time.perf_counter() - t0) * 1000
                tried.append(f"{w}x{h}@{fps} 실패({dt_ms:.1f} ms): {e}")
                try:
                    pipeline.stop()
                except:
                    pass
                pipeline = rs.pipeline(); config = rs.config()
        log("[WARN] RealSense 설정 실패: " + " | ".join(tried))
        raise RuntimeError("RealSense 실패")
    except Exception as e:
        log(f"[WARN] RealSense 실패, 웹캠 폴백: {e}")
        t0 = time.perf_counter()
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT,1080)
        cap.set(cv.CAP_PROP_FPS, 6)
        dt_ms = (time.perf_counter() - t0) * 1000
        if cap.isOpened():
            log(f"Webcam open 1920x1080@6fps in {dt_ms:.2f} ms")
            log(f"Camera total open time {(time.perf_counter()-t_all0)*1000:.2f} ms")
            return ("webcam", cap)
        log("[ERR] 카메라 열기 실패")
        sys.exit(1)

def read_frame_nonblocking(cam):
    """프레임이 없으면 None을 즉시 반환(블로킹 최소화)."""
    if cam[0] == "realsense":
        pipeline, rs = cam[1]
        frames = pipeline.poll_for_frames()
        if not frames:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=50)  # <= 여기서 블록될 수 있음
            except Exception:
                return None
        c = frames.get_color_frame()
        if not c:
            return None
        return np.asanyarray(c.get_data())
    else:
        cap = cam[1]
        ret, frame = cap.read()
        if not ret:
            return None
        return frame

# ---- 빠른 “부재(없음) 필터” ----
def likely_has_code(gray):
    # 1) Laplacian variance: 텍스처 수준
    lap_var = cv.Laplacian(gray, cv.CV_64F).var()
    if lap_var < 15.0:  # 필요 시 10~30 사이에서 조정
        return False
    # 2) Canny 에지 비율
    edges = cv.Canny(gray, 50, 150)
    edge_ratio = (edges > 0).mean()
    if edge_ratio < 0.01:  # 1% 미만이면 거의 무늬 없음으로 가정
        return False
    return True

# ---- 요청한 고정 ROI + 다운스케일 1차 스캔 + timeout ----
def try_decode_datamatrix(bgr):
    h, w = bgr.shape[:2]

    # 화면 중앙 기준 오프셋 + 클리핑
    x0 = (w - ROI_W) // 2 + DX
    y0 = (h - ROI_H) // 2 + DY
    x0 = max(0, min(w - ROI_W, x0))
    y0 = max(0, min(h - ROI_H, y0))

    roi = bgr[y0:y0+ROI_H, x0:x0+ROI_W]
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    # 0) 빠른 “없음” 필터: 실패 시 바로 False
    if not likely_has_code(gray):
        return False

    # 1) 다운스케일로 빠른 1차 스캔
    small = cv.resize(gray, (ROI_W//2, ROI_H//2), interpolation=cv.INTER_AREA)
    try:
        res_small = dm_decode(small, max_count=1, timeout=50)  # timeout 지원 시 수십 ms 컷
    except TypeError:
        res_small = dm_decode(small, max_count=1)
    if res_small:
        return True

    # 2) 후보일 수 있으니 원본으로 한 번 더 (조금 더 긴 timeout)
    try:
        res = dm_decode(gray, max_count=1, timeout=80)
    except TypeError:
        res = dm_decode(gray, max_count=1)
    return bool(res)

def main():
    cam = open_camera()
    last_print = 0.0
    last_decode = 0.0
    detected = False
    decode_iter = 0

    # 디코딩 간격 동적 조절(백오프)
    decode_interval = TICK * 0.9  # 기본 약 0.18s
    miss_streak = 0

    log("=== Start loop: O/X to STDOUT every 0.2s; timing logs to STDERR ===")
    print("[INFO] 0.2초마다 O/X 출력. Ctrl+C 종료")
    try:
        while True:
            now = time.monotonic()

            # 디코딩: 동적 간격
            if now - last_decode >= decode_interval:
                t_cycle0 = time.perf_counter()

                # 프레임 읽기
                t0 = time.perf_counter()
                frame = read_frame_nonblocking(cam)
                t1 = time.perf_counter()

                # ROI/그레이 & 디코딩
                t_roi0 = t1
                if frame is not None:
                    new_detected = try_decode_datamatrix(frame)
                else:
                    new_detected = False
                t2 = time.perf_counter()

                # 시간 기록
                dt_read_ms = (t1 - t0) * 1000.0
                dt_roi_dec_ms = (t2 - t_roi0) * 1000.0
                dt_total_ms = (t2 - t_cycle0) * 1000.0

                detected = new_detected
                decode_iter += 1
                if decode_iter % LOG_EVERY_DECODE == 0:
                    log(
                        f"decode#{decode_iter} "
                        f"read={dt_read_ms:.2f} ms | roi+gray+decode={dt_roi_dec_ms:.2f} ms | total={dt_total_ms:.2f} ms | "
                        f"result={'O' if detected else 'X'} | interval={decode_interval:.3f}s"
                    )

                # 백오프 갱신
                if detected:
                    miss_streak = 0
                    decode_interval = TICK * 0.9               # 즉시 원래 속도로
                else:
                    miss_streak += 1
                    if miss_streak >= 3:
                        decode_interval = min(MAX_BACKOFF, decode_interval * 1.5)
                    else:
                        decode_interval = TICK * 0.9

                last_decode = now

            # 출력은 무조건 0.2초마다 (STDOUT)
            if now - last_print >= TICK:
                print("O" if detected else "X", flush=True)
                last_print = now

            # CPU 과점유 방지
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] 종료")
        log("KeyboardInterrupt received. Exiting...")
    finally:
        if cam[0] == "realsense":
            pipeline, rs = cam[1]; pipeline.stop()
        else:
            cam[1].release()
        log("Camera released.")

if __name__ == "__main__":
    main()
