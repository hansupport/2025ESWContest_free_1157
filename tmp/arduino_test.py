# 젯슨 나노 (수신 측) 코드 - '1'을 수신하는지 확인

import serial
import time

#!!!!!!!!!! 본인의 환경에 맞게 포트 이름을 수정해주세요. !!!!!!!!!!
serial_port = '/dev/ttyACM0' 
baud_rate = 9600

try:
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    print(f"Waiting for '1' from Arduino on {serial_port}...")
    time.sleep(2)

    while True:
        if ser.in_waiting > 0:
            # 아두이노로부터 한 줄을 읽어옵니다.
            line = ser.readline()
            
            # 수신된 바이트 데이터를 문자열로 변환하고 공백을 제거합니다.
            received_data = line.decode('utf-8').strip()

            # 수신된 데이터가 '1'인지 확인합니다.
            if received_data == '1':
                print("OK: Arduino sent '1'.")
            else:
                # 예상치 못한 다른 데이터가 수신된 경우 출력합니다.
                if received_data: # 빈 데이터는 출력하지 않음
                    print(f"Warning: Received something else -> {received_data}")

except serial.SerialException as e:
    print(f"Error: Could not open port {serial_port}. {e}")
except KeyboardInterrupt:
    print("\nProgram stopped.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")