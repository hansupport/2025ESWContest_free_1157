from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("yolov8n.pt")

# 웹캠 열기
cap = cv2.VideoCapture(2)

# 해상도 & 밝기 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # 밝기 수동 조절 (범위는 카메라마다 다름)

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

    # YOLO 예측
    results = model.predict(source=frame, conf=0.4, show=False)

    # 시각화 결과 얻기
    result_img = results[0].plot()

    # 창 띄우고, 주기적으로 다시 보여주기 (안 그러면 창이 얼거나 어두워질 수 있음)
    while True:
        cv2.imshow("YOLOv8 Detection", result_img)

        # ESC 키로 종료
        if cv2.waitKey(100) & 0xFF == 27:
            print("ESC 입력으로 종료합니다.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # 창이 닫히면 break
        if cv2.getWindowProperty("YOLOv8 Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

        # 터미널 입력이 들어오면 다음 프레임
        next_key = input("▶ 다음 프레임: 1 / 종료: q > ").strip()
        if next_key == '1':
            break
        elif next_key == 'q':
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
