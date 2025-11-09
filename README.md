# 🧠 Parkinson's Fall Detection Using MPU6050

A machine learning–based fall detection system designed to assist Parkinson’s patients by continuously monitoring body motion using the **MPU6050 accelerometer and gyroscope sensor**.  
The system detects abnormal motion patterns and predicts whether a **fall has occurred or is likely to occur** using a trained **Random Forest Classifier**.

---

## 🚀 Project Overview

This project uses real-time sensor data from an **MPU6050** connected to an **Arduino Uno**, which streams acceleration and gyroscope readings to a **Flask web application**.  
The Flask app hosts a trained ML model that predicts whether the patient is _stable_ or _falling_ and displays live updates on a web dashboard.  
The goal is to create a **low-cost, real-time fall detection system** for Parkinson’s patients that can help trigger early alerts and assist caregivers.

---

## ⚙️ System Architecture

The system consists of **three main components**:

### 1️⃣ Data Acquisition Layer

- **Hardware:** Arduino Uno + MPU6050 sensor
- **Function:** Continuously measures body motion through 3-axis acceleration and gyroscope data.
- **Communication:** Sends serial data to the connected PC via USB.

### 2️⃣ Data Processing Layer

- **Software:** Python script (`serial_to_flask.py`)
- **Function:**
  - Reads serial data from the Arduino port
  - Formats it into JSON
  - Sends it to the Flask server endpoint (`/predict`) every few seconds

### 3️⃣ Machine Learning & Web Interface Layer

- **Software:** Flask web app (`app.py`)
- **Function:**
  - Receives sensor readings through a POST request
  - Uses a trained **Random Forest model** to classify the motion as “Fall” or “Stable”
  - Displays live prediction results on the dashboard
  - Logs readings for further analysis

---

## 💡 Features

✅ Real-time motion sensing  
✅ Machine learning–based fall prediction  
✅ Web-based live visualization  
✅ Low-cost and easy-to-build prototype  
✅ Scalable to include cloud data storage or IoT alerts (SMS/email)

---

## 🧠 Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **Training Data:** Accelerometer readings representing stable and fall conditions
- **Input Features:**
  - Acceleration X, Y, Z
  - Derived features such as magnitude and angular velocity (optional)
- **Output:**
  - `0 → Stable`
  - `1 → Fall`

---

## 🧰 Hardware & Software Requirements

### 🔧 Hardware

- Arduino Uno
- MPU6050 Sensor
- USB Cable
- Jumper Wires

### 💻 Software

- Python 3.8+
- Flask
- scikit-learn
- pySerial
- requests
- joblib

---

## ⚡ How to Run the Project

1️⃣ Clone the repository

```bash
git clone https://github.com/vaibhav-kiran/parkinsons-fall-detection-using-mpu6050.git
cd parkinsons-fall-detection-using-mpu6050
```
