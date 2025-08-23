# dmatrix_realsense_check_1080p6fps.py
import time
import sys
import numpy as np
import cv2 as cv

# ===== SETTINGS =====
ROI_RATIO = 0.60             # 중앙 정사각형 ROI (min(w,h) * 비율)
DECODE_INTERVAL = 0.20       # 디코딩 간격(초) = 0.2s
# ====================

# OpenCV 최적화
try:
    cv.setUseOptimized(True)
    cv.setNumThreads(1)
except Exception:
    pass

# ---- pylibdmtx 안전 임포트 가드 ----
def load_dmtx_decode():
    try:
        from pylibdmtx.pylibdmtx import decode
        return decode
    except Exception:
        pass
    try:
        import pylibdmtx.pylibdmtx as dmtx
        return dmtx.decode
    except Exception as e:
        raise ImportError(
            "pylibdmtx decode 로드 실패: " + str(e) +
            "\n(참고: sudo apt install libdmtx0a libdmtx-dev 후 "
            "pip install --no-binary :all: pylibdmtx==0.1.10)"
        )

dm_decode = load_dmtx_decode()
# -----------------------------------

def open_camera():
    """1920x1080@6fps 우선 시도, 불가 시 1920x1080@15fps 폴백."""
    try:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()

        for w, h, fps in [(1920, 1080, 6), (1920, 1080, 15)]:
            try:
                config.disable_all_streams()
                config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
                profile = pipeline.start(config)
                print(f"[INFO] RealSense color 시작: {w}x{h} @{fps}fps")
                return ("realsense", (pipeline, rs))
            except Exception:
                try:
                    pipeline.stop()
                except Exception:
                    pass
                pipeline = rs.pipeline()
                config = rs.config()

        raise RuntimeError("RealSense color 스트림 시작 실패")
    except Exception as e:
        print("[WARN] RealSense 실패, 웹캠 폴백:", e)
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv.CAP_PROP_FPS, 6)
        if cap.isOpened():
            print("[INFO] 웹캠 1920x1080 @6fps 시도")
            return ("webcam", cap)
        print("[ERR] 카메라 열기 실패")
        sys.exit(1)

def read_frame(cam):
    if cam[0] == "realsense":
        pipeline, rs = cam[1]
        frames = pipeline.wait_for_frames()
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

def release_camera(cam):
    if cam[0] == "realsense":
        pipeline, rs = cam[1]
        pipeline.stop()
    else:
        cam[1].release()

def main():
    cam = open_camera()
    last_decode_ts = 0.0

    print("[INFO] 0.2초마다 인식 결과 출력 (O=성공, X=실패). Ctrl+C로 종료")
    try:
        while True:
            frame = read_frame(cam)
            if frame is None:
                continue

            now = time.time()
            if now - last_decode_ts >= DECODE_INTERVAL:
                last_decode_ts = now

                h, w = frame.shape[:2]
                side = int(min(w, h) * ROI_RATIO)
                x0 = (w - side) // 2
                y0 = (h - side) // 2
                roi = frame[y0:y0+side, x0:x0+side]

                gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

                try:
                    res = dm_decode(gray, max_count=1)
                except TypeError:
                    res = dm_decode(gray)

                if res:
                    print("O")
                else:
                    print("X")

    except KeyboardInterrupt:
        print("\n[INFO] 종료합니다.")
    finally:
        release_camera(cam)

if __name__ == "__main__":
    main()
