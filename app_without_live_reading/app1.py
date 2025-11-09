import os
import math
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'fall_detection_rf_model.pkl')

try:
    model = joblib.load(MMODEL_PATH if (MMODEL_PATH := MODEL_PATH) else MODEL_PATH)  # simple guard for static linters
except Exception as e:
    model = None
    print(f"Failed to load model at {MODEL_PATH}: {e}")


def build_features(acc_x: float, acc_y: float, acc_z: float):
    resultant = math.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    abs_x = abs(acc_x)
    abs_y = abs(acc_y)
    abs_z = abs(acc_z)
    return np.array([[acc_x, acc_y, acc_z, resultant, abs_x, abs_y, abs_z]])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    data = request.get_json(silent=True) or {}

    # Expect accelerometer inputs; gyro values accepted but unused by current model
    try:
        acc_x = float(data.get('acc_x', 0.0))
        acc_y = float(data.get('acc_y', 0.0))
        acc_z = float(data.get('acc_z', 0.0))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input. Provide numeric acc_x, acc_y, acc_z."}), 400

    X = build_features(acc_x, acc_y, acc_z)
    pred = int(model.predict(X)[0])
    proba = None
    try:
        proba = float(model.predict_proba(X)[0][1])
    except Exception:
        pass

    return jsonify({
        "prediction": pred,
        "label": "fall" if pred == 1 else "stable",
        "prob_fall": proba
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


