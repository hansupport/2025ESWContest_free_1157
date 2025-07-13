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
        continue

    ret, frame = cap.read()
    if not ret:
        print("카메라에서 영상을 읽지 못했습니다.")
        break

    # YOLO 예측 (show=False로 설정)
    results = model.predict(source=frame, conf=0.4, show=False)

    # 시각화된 결과 이미지 얻기
    result_img = results[0].plot()

    # 결과 이미지 표시
    cv2.imshow("YOLOv8 Detection", result_img)

    # 터미널 입력이 들어올 때까지 이미지 창 유지
    while True:
        # ESC 키 누르면 종료
        if cv2.waitKey(100) & 0xFF == 27:  # ESC = 27
            print("ESC 입력으로 종료합니다.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # 터미널 입력이 있으면 다음 루프
        if cv2.getWindowProperty("YOLOv8 Detection", cv2.WND_PROP_VISIBLE) < 1:
            # 창이 닫힌 경우 종료
            break

        # 여기서 다시 터미널 입력 받음
        if input("▶ 다음 프레임: 1 / 종료: q > ").strip() == '1':
            break
        else:
            print("❗ '1'을 입력해야 다음 프레임으로 넘어갑니다.")

cv2.destroyAllWindows()
cap.release()
