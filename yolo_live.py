# yolo_live.py

from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("yolov8n.pt")  # yolov8n.pt, yolov8s.pt 등 사용 가능

# 웹캠 열기 (RealSense는 보통 /dev/video2일 수 있음)
cap = cv2.VideoCapture(2)

# 해상도 설정 (원하는 경우)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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