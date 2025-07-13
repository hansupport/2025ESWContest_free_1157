from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("yolov8n.pt")  # yolov8n.pt, yolov8s.pt 등 사용 가능

# 웹캠 열기 (RealSense는 보통 /dev/video2일 수 있음)
cap = cv2.VideoCapture(2)

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("▶ 터미널에서 '1'을 입력하면 다음 프레임을 예측합니다.")
print("▶ 'q'를 입력하면 종료합니다.")

while True:
    key = input("입력 대기 중 (1: 다음 프레임, q: 종료) > ").strip()
    if key == 'q':
        break
    elif key != '1':
        continue  # 1 외에는 무시하고 다시 대기

    ret, frame = cap.read()
    if not ret:
        print("카메라에서 영상을 읽지 못했습니다.")
        break

    # YOLO 예측
    results = model.predict(source=frame, show=True, conf=0.4)

cap.release()
cv2.destroyAllWindows()
