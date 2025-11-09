import serial
import time
import requests

# === CONFIGURATION ===
PORT = '/dev/cu.usbserial-130'   # your Arduino port
BAUD = 9600
FLASK_URL = 'http://127.0.0.1:5000/predict'

# === CONNECT TO ARDUINO ===
arduino = serial.Serial(port=PORT, baudrate=BAUD, timeout=1)
time.sleep(2)
print("Connected to Arduino...")

# === MAIN LOOP ===
while True:
    try:
        line = arduino.readline().decode('utf-8').strip()
        if not line:
            continue

        # Expecting: "accX,accY,accZ"
        parts = line.split(',')
        if len(parts) != 3:
            print("Invalid data:", line)
            continue

        acc_x, acc_y, acc_z = map(float, parts)

        # Send to Flask app
        response = requests.post(FLASK_URL, json={
            "acc_x": acc_x,
            "acc_y": acc_y,
            "acc_z": acc_z
        })

        if response.status_code == 200:
            result = response.json()
            print(f"📊 X:{acc_x:.2f} Y:{acc_y:.2f} Z:{acc_z:.2f} → {result['label'].upper()} (Pred={result['prediction']}, Prob={result.get('prob_fall')})")
        else:
            print("❌ Flask Error:", response.text)

        time.sleep(5)  # check every 5 seconds

    except Exception as e:
        print("Error:", e)
        time.sleep(2)