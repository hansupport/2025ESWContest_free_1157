import cv2
import torch
import clip
from PIL import Image
import numpy as np

# 1. 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. CLIP 모델 로드
model, preprocess = clip.load("ViT-B/32", device=device)

# 3. 웹캠에서 한 장 캡처
cap = cv2.VideoCapture(0)   # /dev/video0 인 경우
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("카메라에서 프레임을 읽지 못했습니다.")

# 4. OpenCV BGR → PIL RGB
image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# 5. CLIP 전처리
image_input = preprocess(image).unsqueeze(0).to(device)

# 6. 분류하고 싶은 클래스 리스트 (예시)
classes = [
    "person", "car", "bicycle", "dog", "cat",
    "cup", "cell phone", "bottle", "chair", "tree"
]
# "a photo of a {class}" 형태로 텍스트 토크나이징
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)

# 7. 특징 추출 & 유사도 계산
with torch.no_grad():
    image_features = model.encode_image(image_input)      # (1, 512)
    text_features  = model.encode_text(text_inputs)       # (N, 512)
    # 내적 → softmax
    logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)  # (1, N)
    probs  = logits.cpu().numpy()[0]  # (N,)

# 8. 상위 k개 출력
topk = 5
inds = np.argsort(probs)[::-1][:topk]
print("=== CLIP zero-shot 분류 결과 ===")
for i in inds:
    print(f"{classes[i]:<12s}: {probs[i]*100:5.2f}%")

# 9. (선택) 캡처한 이미지도 함께 보여주기
cv2.imshow("Captured Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
