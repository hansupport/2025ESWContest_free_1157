import pyrealsense2 as rs
import numpy as np
import cv2
import time

# --- 사용자 설정 부분 ---
CAMERA_INDEX = 0            # 사용할 카메라 인덱스 (0 = 첫 번째 카메라, 1 = 두 번째 카메라, ...)
CHESSBOARD_CORNERS = (8, 6) # 체커보드 내부 코너 개수 (가로, 세로)
SQUARE_SIZE = 0.032         # 체커보드 한 칸의 실제 크기 (미터 단위, 32mm)
CAPTURE_TARGET = 15         # 캡처할 이미지 목표 개수
# -------------------------

# --- 1. 연결된 카메라 목록 확인 및 변수를 이용한 선택 ---
ctx = rs.context()
devices = ctx.query_devices()
serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]

if len(serials) == 0:
    print("오류: RealSense 카메라가 연결되지 않았습니다.")
    exit()

print("연결된 모든 카메라 시리얼 번호:")
for i, serial in enumerate(serials):
    print(f"  카메라 {i}: {serial}")

if CAMERA_INDEX >= len(serials):
    print(f"\n오류: 설정된 카메라 인덱스({CAMERA_INDEX})에 해당하는 카메라를 찾을 수 없습니다.")
    print(f"사용 가능한 인덱스 범위: 0 ~ {len(serials) - 1}")
    exit()

selected_serial = serials[CAMERA_INDEX]
print(f"\n선택된 카메라: [인덱스 {CAMERA_INDEX}] {selected_serial}")
# --------------------------------------------------------

# [2] 선택된 카메라 초기화
pipeline = rs.pipeline()
config = rs.config()

# 선택된 시리얼 번호로 장치 활성화
config.enable_device(selected_serial)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 파이프라인 시작
try:
    profile = pipeline.start(config)
    print(f"\n카메라 {selected_serial}가 성공적으로 초기화되었습니다.")
    
    # 참고용: RealSense SDK에서 제공하는 공장 캘리브레이션 값(Intrinsics) 출력
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print("\n참고: RealSense SDK 제공 내부 파라미터 (공장 캘리브레이션 값)")
    print(f"  - fx: {intrinsics.fx:.4f}, fy: {intrinsics.fy:.4f}")
    print(f"  - cx: {intrinsics.ppx:.4f}, cy: {intrinsics.ppy:.4f}")
    print(f"  - Distortion Model: {intrinsics.model}")
    print("이 값들은 체커보드 캘리브레이션 결과와 비교해볼 수 있습니다.")

except Exception as e:
    print(f"오류: 카메라를 시작할 수 없습니다. {e}")
    exit()

time.sleep(1) # 카메라 안정화 대기

# [3] 캘리브레이션 데이터 수집
obj_points = []  # 3D 실제 좌표
img_points = []  # 2D 이미지 좌표

# 체커보드의 3D 좌표 생성
objp = np.zeros((CHESSBOARD_CORNERS[0] * CHESSBOARD_CORNERS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS[0], 0:CHESSBOARD_CORNERS[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

capture_count = 0
print("\n--- 캘리브레이션 데이터 수집 시작 ---")
print("카메라에 체커보드를 여러 각도와 거리에서 비춰주세요.")
print(f"'c' 키: 이미지 캡처 | 'q' 키: 캘리브레이션 실행 (목표: {CAPTURE_TARGET}장)")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_CORNERS, None)
        if ret:
            cv2.drawChessboardCorners(img, CHESSBOARD_CORNERS, corners, ret)
        
        # 선택된 카메라의 영상만 시각화
        cv2.putText(img, f"Captured: {capture_count}/{CAPTURE_TARGET}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f'Camera Calibration - S/N: {selected_serial}', img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if ret:
                corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                obj_points.append(objp)
                img_points.append(corners_sub)
                capture_count += 1
                print(f"이미지 캡처 완료 ({capture_count} / {CAPTURE_TARGET})")
            else:
                print("체커보드를 찾지 못해 캡처할 수 없습니다.")

        elif key == ord('q'):
            if capture_count < 10:
                print(f"오류: 캘리브레이션을 위해 최소 10장 이상의 이미지가 필요합니다. (현재 {capture_count}장)")
            else:
                break
finally:
    cv2.destroyAllWindows()

# [4] 카메라 캘리브레이션 수행
if capture_count > 9:
    print("\n캘리브레이션을 수행 중입니다. 잠시 기다려주세요...")
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    if ret:
        print("\n--- 캘리브레이션 결과 ---")
        print(f"재보정 오류 (Reprojection Error): {ret:.4f} pixels")
        if ret > 1.0:
            print("[경고] 재보정 오류가 1.0 픽셀을 초과했습니다. 이미지 품질이 낮을 수 있습니다.")

        print("\n카메라 행렬 (Intrinsic Matrix):")
        print(mtx)
        print("\n왜곡 계수 (Distortion Coefficients):")
        print(dist)

        # 결과 파일 이름에 시리얼 번호 포함하여 저장
        output_filename = f'calib_data_{selected_serial}.npz'
        np.savez(output_filename, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        print(f"\n캘리브레이션 성공! '{output_filename}' 파일이 저장되었습니다.")

    else:
        print("\n캘리브레이션 실패.")

# [5] 파이프라인 최종 종료
pipeline.stop()
print("\n카메라 파이프라인이 중지되었습니다.")

