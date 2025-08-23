# main.py
import subprocess
import time
from dmatrix_final import DMatrixWatcher
import sys
import threading
import queue
from typing import Optional, Tuple, List
import numpy as np
import cv2 as cv

# depth, img2emb를 실행하는 커맨드 예시
# (환경에 맞게 파일 경로/파라미터 수정)
DEPTH_CMD   = ["python3", "depth.py"]
IMG2EMB_CMD = ["python3", "img2emb.py"]

def run_depth_and_embeddings(payloads):
    """
    dmatrix 인식 결과(payloads)를 받아
    depth.py, img2emb.py 등을 순차 실행.
    필요하면 payloads를 인자로 넘겨도 됨.
    """
    # 예: 첫 페이로드만 인자로 넘기고 싶다면:
    # target = payloads[0] if payloads else ""
    # cmd = DEPTH_CMD + ["--code", target]
    # subprocess.run(cmd, check=True)

    print("[MAIN] depth 시작…")
    subprocess.run(DEPTH_CMD, check=True)       # 카메라/리소스 필요 시 활용
    print("[MAIN] depth 완료")

    print("[MAIN] img2emb 시작…")
    subprocess.run(IMG2EMB_CMD, check=True)
    print("[MAIN] img2emb 완료")

def main():
    watcher = DMatrixWatcher(prefer_res=(1920,1080), prefer_fps=6)
    watcher.start()
    print("[MAIN] 상시 추정 시작. 인식되면 자동 일시정지 → 후속 작업 → 재개")

    try:
        while True:
            # 인식 이벤트 기다림 (None이면 타임아웃)
            event = watcher.get_detection(timeout=0.5)
            if event is None:
                continue

            ts, payloads = event
            print(f"[MAIN] 인식 시각 {ts:.3f}, payloads={payloads}")

            # 여기서 depth/img2emb 실행 (카메라 필요한 경우 안전)
            # 워커는 이미 '일시정지 + 카메라 해제' 상태
            try:
                run_depth_and_embeddings(payloads)
            except subprocess.CalledProcessError as e:
                print("[MAIN][ERR] 후속 작업 실패:", e)

            # 후속 작업 끝났으면 다시 추정 재개
            print("[MAIN] 재개합니다.")
            watcher.resume()

    except KeyboardInterrupt:
        print("\n[MAIN] 종료 요청")
    finally:
        watcher.stop()
        print("[MAIN] 종료 완료")

if __name__ == "__main__":
    main()
