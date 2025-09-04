// 핀 번호 설정
#define ENA_PIN 7
#define DIR_PIN 6
#define PUL_PIN 5
#define TRIG_PIN 9
#define ECHO_PIN 2
#define RELAY_PIN 8
#define BUTTON_PIN 3
#define IR_PIN 4

// 모터 및 센서 제어 변수
int pulseDelay = 80;
unsigned long previoㅌusMicros = 0;
unsigned long previousSensorMillis = 0;
const long sensorInterval = 100;

// 초음파 센서 변수
volatile unsigned long echoStartTime = 0;
volatile unsigned long echoEndTime = 0;
volatile bool echoReceived = false;

// 물체 통과 감지 및 재시작 로직 관련 변수
const long stopDelay = 700;      // 물체 통과 후 정지까지의 시간
const long restartDelay = 5000;  // 정지 후 재시작까지의 시간

// 시스템 상태 정의
enum SystemState {
  IDLE_RUNNING,       // 기본 동작 상태 (물체 없음)
  OBJECT_PASSING,     // 물체 통과가 확인된 상태
  WAITING_TO_STOP,    // IR 센서에서 물체가 사라져 정지를 기다리는 상태
  STOPPED             // 모터가 정지한 상태
};
SystemState currentState = IDLE_RUNNING;

unsigned long stateChangeTime = 0;
bool isMotorRunning = true;

// 비상정지 버튼 관련 변수
bool emergencyStop = false;
bool lastButtonState = HIGH;

// stopDelay 보정 관련 변수 (stopDelay 상태 시 비상정지를 위한 보조 변수)
long remainingStopDelay = 0;
bool pausedInWaiting = false;

// 초기 셋팅
void setup() {
  Serial.begin(9600);

  //센서 및 모터 핀모드 설정
  pinMode(ENA_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(PUL_PIN, OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(IR_PIN, INPUT);

  //모터 및 릴레이 활성화
  digitalWrite(ENA_PIN, LOW);
  digitalWrite(DIR_PIN, HIGH);
  digitalWrite(RELAY_PIN, LOW);

  //초음파 센서 타이밍 간섭을 피하기 위한 함수
  attachInterrupt(digitalPinToInterrupt(ECHO_PIN), echo_interrupt, CHANGE);
}

void loop() {
  unsigned long currentMillis = millis();

  // 버튼 입력 처리
  bool buttonState = digitalRead(BUTTON_PIN);
  if (lastButtonState == HIGH && buttonState == LOW) {
    emergencyStop = !emergencyStop;
    if (emergencyStop) {
      isMotorRunning = false;
      Serial.println(2); // 긴급정지

      if (currentState == WAITING_TO_STOP) {
        long elapsed = currentMillis - stateChangeTime;
        remainingStopDelay = max(0, stopDelay - elapsed);
        pausedInWaiting = true;
      }
    } else {
      isMotorRunning = true;
      Serial.println(3); // 재개

      if (pausedInWaiting && currentState == WAITING_TO_STOP) {
        stateChangeTime = currentMillis;
        pausedInWaiting = false;
      }
      if (currentState == STOPPED) {
        stateChangeTime = currentMillis;
      }
    }
    delay(200);
  }
  lastButtonState = buttonState;

  // 초음파 트리거
  if (!emergencyStop && (currentMillis - previousSensorMillis >= sensorInterval)) {
    previousSensorMillis = currentMillis;
    digitalWrite(TRIG_PIN, LOW); delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
  }

  // 센서 값 읽기
  bool irObjectDetected = (digitalRead(IR_PIN) == LOW);
  long distance = -1;

  if (echoReceived) {
    long duration = echoEndTime - echoStartTime;
    distance = (duration * 0.034) / 2;
    echoReceived = false;
  }

  // 상태 머신
  if (!emergencyStop) {
    if (currentState == IDLE_RUNNING) {
      isMotorRunning = true;
      // 빈 상태가 아닐 때를 물체 감지로 판단
      bool isObjectConfirmedByUS = (distance > 0 && (distance < 45 || distance > 47));

      // IR과 초음파가 동시에 물체를 감지하면 상태 변경
      if (irObjectDetected && isObjectConfirmedByUS) {
        digitalWrite(RELAY_PIN, HIGH);
        currentState = OBJECT_PASSING;
      }
    }
    else if (currentState == OBJECT_PASSING) {
      isMotorRunning = true;
      // IR 센서에서 물체가 사라지면 상태 변경
      if (!irObjectDetected) {
        currentState = WAITING_TO_STOP;
        stateChangeTime = currentMillis;
      }
    }
    else if (currentState == WAITING_TO_STOP) {
      isMotorRunning = true; // stopDelay 동안 모터는 계속 동작
      // 설정된 stopDelay 시간이 지나면 모터 정지
      if (currentMillis - stateChangeTime >= (pausedInWaiting ? remainingStopDelay : stopDelay)) {
        isMotorRunning = false;
        Serial.println(1); // 정상 정지 신호
        currentState = STOPPED;
        stateChangeTime = currentMillis;
        remainingStopDelay = 0;
      }
    }
    else if (currentState == STOPPED) {
      isMotorRunning = false;
      // 설정된 restartDelay 시간이 지나면 다시 시작
      if (currentMillis - stateChangeTime >= restartDelay) {
        digitalWrite(RELAY_PIN, LOW);
        currentState = IDLE_RUNNING;
      }
    }
  }

  // 모터 구동
  if (isMotorRunning && !emergencyStop) {
    unsigned long currentMicros = micros();
    if (currentMicros - previousMicros >= pulseDelay) {
      previousMicros = currentMicros;
      digitalWrite(PUL_PIN, HIGH);
      delayMicroseconds(5);
      digitalWrite(PUL_PIN, LOW);
    }
  }
}

// 초음파 센서 거리 측정 보조 함수
void echo_interrupt() {
  if (digitalRead(ECHO_PIN) == HIGH) {
    echoStartTime = micros();
  } else {
    echoEndTime = micros();
    echoReceived = true;
  }
}
