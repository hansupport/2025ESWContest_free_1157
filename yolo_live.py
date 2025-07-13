from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("yolov8n.pt")  # yolov8n.pt, yolov8s.pt 등 사용 가능

# 웹캠 열기 (RealSense는 보통 /dev/video2일 수 있음)
cap = cv2.VideoCapture(2)

# MJPG 포맷으로 설정 (고해상도 호환을 위해)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# 해상도 설정 시도
target_width = 1920
target_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

# 실제 적용된 해상도 확인
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

if actual_width != target_width or actual_height != target_height:
    print(f"❌ 원하는 해상도 {target_width}x{target_height} 설정 실패.")
    print(f"➡️ 현재 해상도는 {int(actual_width)}x{int(actual_height)} 입니다.")
else:
    print(f"✅ 해상도 설정 완료: {int(actual_width)}x{int(actual_height)}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 영상을 읽지 못했습니다.")
        break

    # YOLO 예측
    results = model.predict(source=frame, show=True, conf=0.4)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
