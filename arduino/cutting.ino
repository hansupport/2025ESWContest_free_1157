#include <Servo.h>

//서보모터 정의
Servo servo1;
Servo servo2;
Servo servo3;

// 핀 정의
const int pulPin1 = 6;
const int dirPin1 = 7;
const int enaPin1 = 8;  //커터 모터 핀 정의

const int pulPin2 = 3;
const int dirPin2 = 4;
const int enaPin2 = 5;  // 디스펜서 모터 핀 정의

const int switchPin = 2; // 정지 스위치

// 모터 제어 및 계산용 변수
const int motorStepsPerRev = 200 * 16; // 모터 1회전 당 스텝 수 (마이크로스텝핑 1/16)
const float pulleyRatio = 30.0 / 60.0; // 풀리 기어비
const float shaftDiameter = 16.0;      // 공급축 지름 (mm)
const float shaftCircumference = 3.1416 * shaftDiameter / 10.0; // 공급축 둘레 (cm)

// 스위치 클릭 감지용 변수
int buttonState;
int lastButtonState = HIGH;
long lastDebounceTime = 0;
long debounceDelay = 50;
int clickCount = 0;
long lastClickTime = 0;
long multiClickTimeout = 500; // 다중 클릭 대기 시간

// 포장 공정 전체를 수행하는 함수
void wrappingProcess(float cuttingSizeCm) {
  // 1. 디스펜서를 이용해 설정된 길이만큼 뽁뽁이 공급
  digitalWrite(dirPin2, LOW); // 정방향 설정

  float shaftRevs = cuttingSizeCm / shaftCircumference; 
  float motorRevs = shaftRevs / pulleyRatio;
  long totalPulses = (long)(motorRevs * motorStepsPerRev);
  totalPulses = totalPulses * 4/3; // 오차 보정

  for (long i = 0; i < totalPulses; i++) {
    digitalWrite(pulPin2, HIGH);
    delayMicroseconds(60);
    digitalWrite(pulPin2, LOW);
    delayMicroseconds(60);
  }

  // 2. 고정용 서보모터(1, 2번) 동작
  servo1.write(60);
  servo2.write(0);
  delay(2000);

  // 3. 뽁뽁이 고정을 위해 디스펜서 모터 살짝 역회전
  digitalWrite(dirPin2, HIGH); // 역방향 설정
  for (int i = 0; i < 5000; i++) {
    digitalWrite(pulPin2, HIGH);
    delayMicroseconds(25);
    digitalWrite(pulPin2, LOW);
    delayMicroseconds(25);
  }
  digitalWrite(dirPin2, LOW); // 다음 동작을 위해 정방향으로 원복
  delay(500);

  // 4. 커팅을 위한 모터 동작
  // 정방향 이동
  digitalWrite(dirPin1, HIGH);
  for (long i = 0; i < 36000; i++) {
    digitalWrite(pulPin1, HIGH);
    delayMicroseconds(20);
    digitalWrite(pulPin1, LOW);
    delayMicroseconds(20);
  }
  // 역방향 이동 및 원위치
  digitalWrite(dirPin1, LOW);
  for (long i = 0; i < 36000; i++) {
    digitalWrite(pulPin1, HIGH);
    delayMicroseconds(20);
    digitalWrite(pulPin1, LOW);
    delayMicroseconds(20);
  }

  // 5. 고정용 서보모터(1, 2번) 원위치
  servo1.write(0);
  servo2.write(60);
  
  // 6. 정리용 서보모터(3번) 동작으로 올바른 일정한 커팅 품질 유지
  servo3.write(0);   // 0도로 내려서 다듬기
  delay(500);
  servo3.write(90);  // 90도로 원위치
  
  delay(4500); // 다음 공정을 위한 대기

  // 7. 마지막에 디스펜서에서 10cm 추가 공급 (다음 과정에 뽑을 것을 미리 좀 뽑아놓아 밑걸림 방지)
  float finalDispenseSize = 10.0;
  digitalWrite(dirPin2, LOW); // 정방향 설정

  shaftRevs = finalDispenseSize / shaftCircumference;
  motorRevs = shaftRevs / pulleyRatio;
  totalPulses = (long)(motorRevs * motorStepsPerRev);
  totalPulses = totalPulses * 4/3; // 오차 보정

  for (long i = 0; i < totalPulses + 5000; i++) {
    digitalWrite(pulPin2, HIGH);
    delayMicroseconds(60);
    digitalWrite(pulPin2, LOW);
    delayMicroseconds(60);
  }
}

// 여기서부턴 커팅기 수동 조작 (기계 메인터넌스용)
// 택트 스위치 1회 클릭 시 동작 (뽁뽁이 8cm 공급)
void singleClickAction() {
  float singleClickSize = 8.0;

  digitalWrite(dirPin2, LOW); // 정방향 설정

  float shaftRevs = singleClickSize / shaftCircumference;
  float motorRevs = shaftRevs / pulleyRatio;
  long totalPulses = (long)(motorRevs * motorStepsPerRev);
  totalPulses = totalPulses * 4/3; // 오차 보정

  for (long i = 0; i < totalPulses; i++) {
    digitalWrite(pulPin2, HIGH);
    delayMicroseconds(60);
    digitalWrite(pulPin2, LOW);
    delayMicroseconds(60);
  }
}

// 택트 스위치 2회 클릭 시 동작 (10cm 포장 공정)
void doubleClickAction() {
  wrappingProcess(10.0);
}

// 택트 스위치 3회 클릭 시 동작 (30cm 포장 공정)
void tripleClickAction() {
  wrappingProcess(30.0);
}

// 셋업
void setup() {
  Serial.begin(9600);

  // 서보모터 핀 설정 및 초기화
  servo1.attach(9);
  servo2.attach(10);
  servo3.attach(11); 

  // 서보모터 초기 위치 설정
  servo1.write(0);
  servo2.write(60);
  servo3.write(90); 

  // 스텝모터 및 스위치 핀 모드 설정
  pinMode(pulPin1, OUTPUT);
  pinMode(dirPin1, OUTPUT);
  pinMode(enaPin1, OUTPUT);
  pinMode(pulPin2, OUTPUT);
  pinMode(dirPin2, OUTPUT);
  pinMode(enaPin2, OUTPUT);
  pinMode(switchPin, INPUT_PULLUP);

  // 스텝모터 드라이버 활성화
  digitalWrite(enaPin1, LOW);
  digitalWrite(enaPin2, LOW);
}

void loop() {
  // 젯슨 나노로부터 시리얼 데이터 수신 처리
  if (Serial.available() > 0) {
    String message = Serial.readStringUntil('\n');
    
    // 메시지가 B로 시작하면 포장 공정 실행
    if (message.startsWith("B")) {
      String numberPart = message.substring(1);
      long bubbleLengthMm = numberPart.toInt();
      
      if (bubbleLengthMm > 0) {
        float bubbleLengthCm = (float)bubbleLengthMm / 10.0;
        float targetLengthCm = bubbleLengthCm;

        if (targetLengthCm > 0) {
            wrappingProcess(targetLengthCm);
        } 
      }
    }
  }

  // 수동조작 스위치 정의
  int reading = digitalRead(switchPin);

  // 버튼 상태가 변경되면 디바운스 타이머 리셋 (더블, 트리플 클릭 인식을 위한 설정)
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  // 디바운싱 시간이 지나면 버튼 상태 업데이트
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != buttonState) {
      buttonState = reading;
      // 버튼이 눌렸을 때(LOW) 클릭 카운트 증가
      if (buttonState == LOW) {
        clickCount++;
        lastClickTime = millis();
      }
    }
  }
  lastButtonState = reading;

  // 마지막 클릭 후 일정 시간이 지나면 클릭 횟수에 따라 함수 실행
  if (clickCount > 0 && (millis() - lastClickTime) > multiClickTimeout) {
    if (clickCount == 1) {
      singleClickAction();  // 1번 클릭
    } else if (clickCount == 2) {
      doubleClickAction(); // 2번 클릭
    } else if (clickCount >= 3) {
      tripleClickAction(); // 3번 이상 클릭
    }
    clickCount = 0; // 클릭 카운트 초기화
  }
}

