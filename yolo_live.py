from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("yolov8n.pt")

# 웹캠 열기
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

print("▶ 창에서 '1' 키를 누르면 다음 프레임, 'q' 또는 ESC 키를 누르면 종료합니다.")

show_new_frame = True
result_img = None

while True:
    if show_new_frame:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 읽지 못했습니다.")
            break

        # YOLO 예측
        results = model.predict(source=frame, conf=0.4, show=False)

        # 시각화 및 색상 보정
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

        show_new_frame = False  # 다음 입력 전까진 같은 이미지 유지

    # 이미지 창 표시
    cv2.imshow("YOLOv8 Detection", result_img)

    # 키 입력 대기 (100ms 간격)
    key = cv2.waitKey(100) & 0xFF

    if key == ord('1'):
        show_new_frame = True  # 다음 프레임 보여주기
    elif key == ord('q') or key == 27:  # 'q' or ESC
        break

cap.release()
cv2.destroyAllWindows()
