import pyrealsense2 as rs
import numpy as np
import cv2
import time

# --- 사용자 설정 부분 ---
CHESSBOARD_CORNERS = (8, 6)
SQUARE_SIZE = 0.032
# -------------------------

# [1] 카메라 초기화
serials = []
try:
    for dev in rs.context().query_devices():
        serials.append(dev.get_info(rs.camera_info.serial_number))
    if len(serials) < 2:
        raise RuntimeError("2대 이상의 RealSense 카메라가 필요합니다.")
except Exception as e:
    print(e)
    exit()

print("연결된 카메라:", serials)
pipelines = []
for serial in serials:
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(config)
    pipelines.append(pipe)

time.sleep(1)

# [2] 캘리브레이션 데이터 수집
obj_points = []
img_points1 = []
img_points2 = []

objp = np.zeros((CHESSBOARD_CORNERS[0] * CHESSBOARD_CORNERS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS[0], 0:CHESSBOARD_CORNERS[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE
capture_count = 0

print("\n--- 캘리브레이션 시작 ---")
print("'c' 키: 이미지 캡처 | 'q' 키: 캘리브레이션 시작")

try:
    while True:
        frames1 = pipelines[0].wait_for_frames()
        frames2 = pipelines[1].wait_for_frames()
        
        color_frame1 = frames1.get_color_frame()
        color_frame2 = frames2.get_color_frame()

        if not color_frame1 or not color_frame2:
            continue

        img1 = np.asanyarray(color_frame1.get_data())
        img2 = np.asanyarray(color_frame2.get_data())

        gpu_img1 = cv2.cuda_GpuMat()
        gpu_img2 = cv2.cuda_GpuMat()
        gpu_img1.upload(img1)
        gpu_img2.upload(img2)

        gpu_gray1 = cv2.cuda.cvtColor(gpu_img1, cv2.COLOR_BGR2GRAY)
        gpu_gray2 = cv2.cuda.cvtColor(gpu_img2, cv2.COLOR_BGR2GRAY)

        gray1 = gpu_gray1.download()
        gray2 = gpu_gray2.download()

        ret1, corners1 = cv2.findChessboardCorners(gray1, CHESSBOARD_CORNERS, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, CHESSBOARD_CORNERS, None)

        if ret1 and ret2:
            cv2.drawChessboardCorners(img1, CHESSBOARD_CORNERS, corners1, ret1)
            cv2.drawChessboardCorners(img2, CHESSBOARD_CORNERS, corners2, ret2)
        
        display_img = np.hstack((img1, img2))
        cv2.putText(display_img, f"Captured: {capture_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Stereo Calibration', display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if ret1 and ret2:
                corners1_sub = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                corners2_sub = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                obj_points.append(objp)
                img_points1.append(corners1_sub)
                img_points2.append(corners2_sub)
                capture_count += 1
                print(f"이미지 캡처 완료 ({capture_count} / 20)")
            else:
                print("오류: 두 카메라에서 동시에 체커보드를 찾지 못했습니다.")
        elif key == ord('q'):
            if capture_count < 10:
                print("오류: 캘리브레이션을 위해 최소 10장 이상의 이미지가 필요합니다.")
            else:
                break
finally:
    cv2.destroyAllWindows()
    # 파이프라인 중지는 캘리브레이션이 끝난 후로 이동

# [3] 스테레오 캘리브레이션 수행
if capture_count > 9:
    print("\n캘리브레이션을 수행 중입니다. 잠시 기다려주세요...")

    # --- 추가된 부분: 각 카메라의 재보정 오류(Reprojection Error) 출력 ---
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(obj_points, img_points1, gray1.shape[::-1], None, None)
    ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(obj_points, img_points2, gray2.shape[::-1], None, None)

    print(f"\n--- 캘리브레이션 정확도 ---")
    print(f"카메라 1 재보정 오류 (Reprojection Error): {ret1:.4f} pixels")
    print(f"카메라 2 재보정 오류 (Reprojection Error): {ret2:.4f} pixels")

    if ret1 > 1.0 or ret2 > 1.0:
        print("[경고] 개별 카메라의 재보정 오류가 1.0 픽셀을 초과했습니다. 이미지 품질을 확인하고 다시 캡처하는 것을 권장합니다.")
    # --------------------------------------------------------------------

    flags = cv2.CALIB_FIX_INTRINSIC
    
    # --- 추가된 부분: 스테레오 재보정 오류 출력 ---
    ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points1, img_points2, mtx1, dist1, mtx2, dist2, gray1.shape[::-1], flags=flags
    )
    print(f"스테레오 재보정 오류 (Stereo Reprojection Error): {ret_stereo:.4f} pixels")
    print("--------------------------")
    # ---------------------------------------------
    
    # --- 수정된 부분: 성공 여부를 오류 값 기준으로 판단 ---
    if ret_stereo < 1.5: # 일반적으로 1.0 미만을 목표로 하지만, 1.5 정도로 기준을 완화
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = T.ravel()
        np.save('transform_matrix.npy', transform_matrix)
        
        print("\n캘리브레이션 성공!")
        print("`transform_matrix.npy` 파일이 저장되었습니다.")
        print("변환 행렬:\n", transform_matrix)
    else:
        print("\n캘리브레이션 실패. 재보정 오류가 너무 높습니다 (1.5 초과). 더 나은 품질의 이미지를 사용해 다시 시도하세요.")

# [4] 파이프라인 최종 종료
for pipe in pipelines:
    pipe.stop()
