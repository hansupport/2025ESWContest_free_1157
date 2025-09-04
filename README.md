# 뽁스박스: AI 기반 자동 포장재 커팅 시스템

AI로 물체를 이해하고, DataMatrix 인식 여부에 따라 유연하게 분기하며, 최종적으로 포장재를 자동 재단·출력하는 하이브리드 포장 자동화 프로젝트입니다. 3D 뎁스와 이미지 임베딩을 결합해 크기(W×L×H)와 시각적 특징을 동시에 다루며, 불확실한 경우에는 안전 게이트를 통해 커팅을 보류합니다.

- 시연 영상: https://youtu.be/RoNCsCjtFeY
- 저장소: https://github.com/hansupport/2025ESWContest_free_1157

---

## 개발 개요

수동 포장은 자재 낭비와 편차, 처리 속도 저하 문제를 야기합니다. 뽁스박스는 크기만 재는 기존 자동화에서 더 나아가, 물체의 종류·특성까지 파악하여 적정 완충재 길이를 정하고 즉시 재단합니다. 그리고, DataMatrix가 인식되지 않는 경우에도 모델 추론 경로를 통해 라벨을 추정하여 포장을 진행할 수 있습니다. 목표는 포장 품질 표준화, 자재 절감, 처리량 향상입니다.

---

## 핵심 아이디어

1) DataMatrix가 보이면 즉시 결정
2) 인식 실패 시 백업 경로로 전환  
   3D 뎁스 + 3-view 이미지 임베딩 → ML 분류 → 최적 길이 산출 → 커팅

불확실할 때는 품질지수 q, Top1–Top2 마진 기준으로 Unknown으로 처리하여 오동작 컷을 예방합니다.

---

## 시스템 구성

### HW
- 제어부: NVIDIA Jetson Nano, Arduino R3(2대)
- 인식부: Intel RealSense D435f
- 구동부: NEMA 17 스테퍼, 서보
- 전원부: SMPS

### SW
- 언어/OS: Python 3.6, C++, Ubuntu 18.04 LTS
- 주요 라이브러리: OpenCV, Numpy, LightGBM, ONNX Runtime, Flask, PySerial, pyrealsense2, Waitress

---

## 동작 흐름

1. 스캔 트리거 수신(컨베이어/IR) → 분석 시작
2. RGBD 캡처 → UI에 최신 프레임 전송
3. 제한 시간 내 DataMatrix 시도
4. 분기
   - 성공: 라벨·치수·포장 길이 결정 → 커팅기로 전송 → DB 로깅
   - 실패: Depth 특징 + 3-view 임베딩 → LGBM 분류 → 길이 결정 → 커팅기로 전송 → DB 로깅
5. 상태 완료로 전환

---

## 디렉터리 구조
```text
.
├── core
│ ├── config.py # 구성 및 파라미터
│ ├── lite.py # 경량 추론 모듈
│ └── utils.py # 공통 유틸
├── model
│ ├── datamatrix.py # DataMatrix 인식
│ ├── depth.py # W×L×H 추정
│ ├── img2emb.py # 이미지 → 임베딩(ONNX)
│ ├── train.py # LightGBM 학습 파이프라인
│ └── pretrain
│ ├── capture.py
│ └── pretrain_weight.py
├── web
│ ├── index.html
│ ├── style.css
│ └── script.js
├── arduino
│ ├── conveyer.ino # 컨베이어 제어
│ └── cutting.ino # 커팅 제어
└── main.py # 실행 진입점
```

---

## 알고리즘 개요

### Depth 기반 치수 산출
- 바닥 평면 추정(RANSAC) 후 높이맵 계산
- 임계 기준으로 물체 영역 분리·노이즈 제거
- 바닥 투영으로 길이/너비, 높이 최대값 산출
- 다프레임 요약으로 강건화(중앙값, 변동도, 품질지수)

### 3-view 이미지 임베딩
- 백본: MobileNetV2의 GAP 출력 사용(FC/Softmax 제거)
- ONNX 변환·ONNX Runtime 추론
- ROI: 센터+좌우 거울 3-view, 각 128D → 총 임베딩 차원  \( 384 = 128 \*  3\)
- 출력 벡터는 L2 정규화로 안정화

### ML 분류 파이프라인
- 특징: depth 스칼라 15개 + 3-view 임베딩 384D (+메타)
- 전처리: NaN/Inf 안전 치환, L2 정규화
- 분류기: LightGBM, Grid Search + Stratified K-Fold(5)
- 추론: 클래스 확률, Top-1 및 신뢰도 산출
- 차원 메모(데이터 자산화):  \( 399 = 15 + 384 + \text{meta} \)

---

## 커팅 제어 시퀀스

1. 디스펜서 구동
2. 양쪽 홀더 고정
3. 커터 작동
4. 홀더 해제

프로토콜 예: Jetson → Arduino  
'B1500' (밀리미터 단위 길이)

---

## 현장 운용

- Unknown 게이트: q, Top1–Top2 마진 기준으로 불확실 시 컷 보류
- 듀얼 아두이노로 컨베이어/커팅 분리
- 모든 샘플의 임베딩·치수 로그를 DB에 축적(원인 분석·재학습)
- 구성 가능한 파라미터: ROI, 임계값, 타임아웃
- 모델 파일 변경 자동 감지 → 핫 리로드

---

## 기대 효과와 적용 분야

- 자재 절감, 처리량 향상, 포장 품질 표준화
- 전자상거래 물류센터, 제조 포장 라인, 택배/물류, 소규모 유통 등

---

## 팀

- 한지원: hansupport@naver.com
- 최민기: mc86757@gmail.com
- 이형민: good7895day@gmail.com
