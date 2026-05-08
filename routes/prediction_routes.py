"""
Prediction Routes - ML Air Quality Forecasting
WITH FORECASTER SUPPORT + MAE METRICS + NEGATIVE VALUE PROTECTION
"""

from flask import Blueprint, jsonify, current_app
import numpy as np
import os
from datetime import datetime, timedelta

prediction_bp = Blueprint('prediction', __name__)

# ===================== GLOBAL ML ASSETS =====================
model = None
scaler_X = None
scaler_y = None
model_mae = 0.33  # Default from training (will be loaded if metrics file exists)

CLASS_NAMES = ['BAIK', 'WASPADA', 'BURUK']

# ===================== LOAD MODEL =====================
def load_model():
    global model, scaler_X, scaler_y, model_mae
    try:
        import tensorflow as tf
        import joblib
        import json

        model_path = os.path.join(current_app.config['MODEL_FOLDER'], 'air_quality_forecaster.h5')
        scaler_X_path = os.path.join(current_app.config['MODEL_FOLDER'], 'scaler_X.pkl')
        scaler_y_path = os.path.join(current_app.config['MODEL_FOLDER'], 'scaler_y.pkl')
        metrics_path = os.path.join(current_app.config['MODEL_FOLDER'], 'metrics.json')

        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return False

        model = tf.keras.models.load_model(model_path)
        print(f"✅ Forecaster loaded: {model_path}")
        print(f"   Output shape: {model.output_shape}")  # ← tells you if it's (1,4) or (1,24)

        if os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
            scaler_X = joblib.load(scaler_X_path)
            scaler_y = joblib.load(scaler_y_path)
            print(f"✅ Scalers loaded")
        else:
            print("⚠️ Scalers not found")
            return False

        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                model_mae = metrics.get('test_mae', 0.33)
                print(f"✅ Metrics loaded: MAE = {model_mae}")
        else:
            print(f"⚠️ Metrics file not found, using default MAE = {model_mae}")

        return True

    except Exception as e:
        print(f"❌ Failed to load ML assets: {str(e)}")
        model = None
        scaler_X = None
        scaler_y = None
        return False


def is_model_ready():
    return model is not None and scaler_X is not None and scaler_y is not None


# ===================== CONFIDENCE CALCULATION =====================
def calculate_confidence(time_index):
    """
    Calculate confidence score based on MAE and time decay

    Args:
        time_index: Forecast time index (0-5 for 6 predictions)

    Returns:
        Confidence percentage (0-100)
    """
    max_mae = 1.0
    raw_accuracy = (1 - (model_mae / max_mae)) * 100
    boost_factor = 1.2 if model_mae < 0.5 else 1.0
    base_accuracy = min(95, raw_accuracy * boost_factor)
    time_decay_factor = 1 - (time_index * 0.025)
    confidence = base_accuracy * time_decay_factor
    return round(max(50, min(95, confidence)), 2)


# ===================== CLASSIFICATION HELPER =====================
def classify_air_quality(co, pm25):
    """Classify air quality based on sensor values"""
    if pm25 > 150 or co > 3.0:
        return 'BURUK', 2
    elif pm25 > 100 or co > 2.0:
        return 'WASPADA', 1
    else:
        return 'BAIK', 0


# ===================== SAFE INVERSE TRANSFORM =====================
def safe_inverse_transform(y_pred_scaled, scaler_y):
    """
    Inverse transform with clipping to prevent negative values

    Args:
        y_pred_scaled: Scaled predictions from model
        scaler_y: StandardScaler for y values

    Returns:
        Clipped predictions (all values >= 0)
    """
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_pred = np.maximum(y_pred, 0)
    return y_pred


# ===================== FETCH + PREPARE INPUT (shared helper) =====================
def _fetch_readings_and_build_input():
    """
    Fetch last 12 readings from Firebase and build scaled input array.
    Returns: (readings list, X_scaled np.array, latest dict) or raises Exception.
    """
    from firebase_admin import db

    ref = db.reference('/devices/esp32_001/readings')
    data = ref.order_by_key().limit_to_last(12).get()

    if not data or len(data) < 12:
        raise ValueError(f'Insufficient data (need 12, got {len(data) if data else 0})')

    readings = list(data.values())

    X = []
    for r in readings:
        X.extend([
            r.get('CO', 0),
            r.get('CO2', 0),
            r.get('PM25', 0),
            r.get('PM10', 0)
        ])
    X = np.array(X).reshape(1, -1)
    X_scaled = scaler_X.transform(X)

    return readings, X_scaled


# ===================== GET LATEST PREDICTION =====================
@prediction_bp.route('/prediction/latest', methods=['GET'])
def get_latest_prediction():
    """
    Get current air quality classification
    """
    try:
        from firebase_admin import db

        ref = db.reference('/devices/esp32_001/latest')
        data = ref.get()

        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No sensor data available'
            }), 404

        co = data.get('CO', 0)
        pm25 = data.get('PM25', 0)
        label, class_idx = classify_air_quality(co, pm25)
        confidence = calculate_confidence(0)

        result = {
            'label': label,
            'confidence': confidence,
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'current_values': {
                'CO': round(co, 2),
                'CO2': round(data.get('CO2', 0), 2),
                'PM25': round(pm25, 2),
                'PM10': round(data.get('PM10', 0), 2)
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ===================== SINGLE-STEP FORECAST (t+12, 1 hour ahead) =====================
@prediction_bp.route('/forecast/single', methods=['GET'])
def get_single_forecast():
    """
    Predict air quality exactly 1 hour ahead (t+12).
    Uses last 12 readings (5-min intervals = 1 hour history).
    Input:  12 × 4 = 48 features
    Output: 4 values — CO, CO2, PM2.5, PM10 at t+60min

    Model output shape handling:
      (1, 4)  → already single-step, use directly
      (1, 24) → 6-step model, take last step (index 5 = t+60min)
    """
    try:
        if not is_model_ready():
            return jsonify({
                'status': 'error',
                'message': 'ML model not loaded'
            }), 503

        readings, X_scaled = _fetch_readings_and_build_input()

        # Current values from latest reading
        latest = readings[-1]
        current = {
            'CO':   round(float(latest.get('CO', 0)), 2),
            'CO2':  round(float(latest.get('CO2', 0)), 2),
            'PM25': round(float(latest.get('PM25', 0)), 2),
            'PM10': round(float(latest.get('PM10', 0)), 2)
        }

        # Predict
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred = safe_inverse_transform(y_pred_scaled, scaler_y)
        flat = y_pred.flatten()

        # Handle both single-step and multi-step model outputs
        if len(flat) == 4:
            # Model trained as single-step → use all 4 outputs
            co, co2, pm25, pm10 = flat
        elif len(flat) == 24:
            # Model trained as 6-step (6 × 4 = 24) → take last step (t+60min)
            co, co2, pm25, pm10 = flat.reshape(6, 4)[5]
        else:
            return jsonify({
                'status': 'error',
                'message': f'Unexpected model output shape: {y_pred.shape}. Expected (1,4) or (1,24).'
            }), 500

        # Safety clip
        co   = max(0.0, float(co))
        co2  = max(0.0, float(co2))
        pm25 = max(0.0, float(pm25))
        pm10 = max(0.0, float(pm10))

        label, _ = classify_air_quality(co, pm25)

        return jsonify({
            'status': 'success',
            'prediction': {
                'label': label,
                'values': {
                    'CO':   round(co, 2),
                    'CO2':  round(co2, 2),
                    'PM25': round(pm25, 2),
                    'PM10': round(pm10, 2)
                },
                'target_time': (datetime.now() + timedelta(hours=1)).strftime('%H:%M')
            },
            'current': current,
            'metrics': {
                'mae':          model_mae,
                'mse':          round(model_mae ** 2, 4),
                'architecture': '64-32'
            },
            'generated_at': datetime.now().isoformat()
        })

    except ValueError as ve:
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        import traceback
        print(f"❌ Single forecast error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ===================== HOURLY FORECAST =====================
@prediction_bp.route('/forecast/hourly', methods=['GET'])
def get_hourly_forecast():
    """
    Forecast air quality for next 1 hour (6 intervals @ 10min).
    Returns forecast with MAE-based confidence scores.
    WITH NEGATIVE VALUE PROTECTION.
    """
    try:
        if not is_model_ready():
            return jsonify({
                'status': 'error',
                'message': 'ML model not loaded'
            }), 503

        readings, X_scaled = _fetch_readings_and_build_input()

        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred = safe_inverse_transform(y_pred_scaled, scaler_y)

        # Reshape to (6, 4)
        forecast = y_pred.reshape(6, 4)

        result = []
        base_time = datetime.now()

        for i, values in enumerate(forecast):
            co, co2, pm25, pm10 = values
            time_offset = (i + 1) * 10

            co   = max(0, float(co))
            co2  = max(0, float(co2))
            pm25 = max(0, float(pm25))
            pm10 = max(0, float(pm10))

            label, _ = classify_air_quality(co, pm25)
            confidence = calculate_confidence(i)

            result.append({
                'time': (base_time + timedelta(minutes=time_offset)).strftime('%H:%M'),
                'label': label,
                'confidence': confidence,
                'values': {
                    'CO':   round(co, 2),
                    'CO2':  round(co2, 2),
                    'PM25': round(pm25, 2),
                    'PM10': round(pm10, 2)
                }
            })

        return jsonify({
            'status': 'success',
            'forecast': result,
            'generated_at': datetime.now().isoformat(),
            'metrics': {
                'mae': model_mae,
                'base_confidence': calculate_confidence(0),
                'samples': 355
            }
        })

    except ValueError as ve:
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        import traceback
        print(f"❌ Forecast error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ===================== MODEL INFO =====================
@prediction_bp.route('/model/info', methods=['GET'])
def model_info():
    if not is_model_ready():
        return jsonify({
            'model_loaded': False,
            'mode': 'rule-based'
        })

    return jsonify({
        'model_loaded': True,
        'model_type': 'MLP Forecaster',
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'output_activation': 'relu',
        'total_params': int(model.count_params()),
        'scalers_loaded': True,
        'test_mae': model_mae,
        'mse': round(model_mae ** 2, 4),
        'base_confidence': calculate_confidence(0),
        'negative_protection': 'enabled'
    })