import serial
import time

arduino = serial.Serial(port='/dev/cu.usbserial-130', baudrate=9600, timeout=1)
time.sleep(2)  # allow Arduino to reset

print("Connected to Arduino...")

while True:
    line = arduino.readline().decode('utf-8').strip()
    if line:
        print("MPU6050:", line)
        