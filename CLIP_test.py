import cv2
import torch
import clip
from PIL import Image
import numpy as np
import time

# 1. 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. CLIP 모델 로드
model, preprocess = clip.load("ViT-B/32", device=device)

# 3. 웹캠에서 한 장 캡처
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("카메라에서 프레임을 읽지 못했습니다. (device index 2)")

# 디버그: 각 단계 시간 측정 시작
t0 = time.perf_counter()

# 4. OpenCV BGR → PIL RGB
image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
t1 = time.perf_counter()

# 5. CLIP 전처리
image_input = preprocess(image).unsqueeze(0).to(device)
t2 = time.perf_counter()

# 6. 클래스 토크나이징
classes = [
    "person", "car", "bicycle", "dog", "cat",
    "cup", "cell phone", "bottle", "chair", "tree"
]
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
t3 = time.perf_counter()

# 7. 특징 추출 & 유사도 계산
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features  = model.encode_text(text_inputs)
    logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    probs  = logits.cpu().numpy()[0]
t4 = time.perf_counter()

# 8. 상위 k개 출력
topk = 5
inds = np.argsort(probs)[::-1][:topk]
print("=== CLIP zero-shot 분류 결과 ===")
for i in inds:
    print(f"{classes[i]:<12s}: {probs[i]*100:5.2f}%")
t5 = time.perf_counter()

# 9. 캡처한 이미지 보여주기
cv2.imshow("Captured Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
t6 = time.perf_counter()

# 디버그: 단계별 소요 시간 출력
print("\n--- 단계별 소요 시간 ---")
print(f"4. BGR→RGB 변환          : {t1 - t0:.3f}초")
print(f"5. CLIP 전처리           : {t2 - t1:.3f}초")
print(f"6. 텍스트 토크나이징    : {t3 - t2:.3f}초")
print(f"7. 특징 추출 & 유사도   : {t4 - t3:.3f}초")
print(f"8. 결과 출력            : {t5 - t4:.3f}초")
print(f"9. 이미지 표시          : {t6 - t5:.3f}초")
print(f"총합                    : {t6 - t0:.3f}초")
