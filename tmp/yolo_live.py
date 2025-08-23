from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("yolov8n.pt")

# 웹캠 열기
cap = cv2.VideoCapture(2)
# 해상도 설정 (필요 시)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 윈도우 생성 (최초 한 번)
cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 영상을 읽지 못했습니다.")
        break

    # YOLO 예측 (show=False)
    results = model.predict(source=frame, conf=0.4, show=False)

    # 플롯된 이미지를 가져와 BGR로 변환
    result_img = results[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

    # 화면에 띄우고 3000ms(3초) 대기
    cv2.imshow("YOLOv8 Detection", result_img)
    key = cv2.waitKey(3000) & 0xFF

    # 'q' 또는 ESC(27) 누르면 종료
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
