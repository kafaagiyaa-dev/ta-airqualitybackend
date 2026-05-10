"""
Prediction Routes - ML Air Quality Forecasting
Supports MLP and LSTM models with 8 features:
  CO, CO2, PM25, PM10, NO2, ozone, temp, humidity
"""

from flask import Blueprint, jsonify, current_app
import numpy as np
import os
from datetime import datetime, timedelta

prediction_bp = Blueprint('prediction', __name__)

# =====================================================
# CONSTANTS
# =====================================================
FEATURES   = ['CO', 'CO2', 'PM25', 'PM10', 'NO2', 'ozone', 'temp', 'humidity']
TEMP_IDX   = FEATURES.index('temp')   # index 6 — allowed to be negative
LOOKBACK   = 12                        # timesteps needed from Firebase

# =====================================================
# GLOBAL ML ASSETS — MLP
# =====================================================
mlp_model    = None
mlp_scaler_X = None
mlp_scaler_y = None
mlp_mae      = None   # loaded from training_metrics_mlp.json

# =====================================================
# GLOBAL ML ASSETS — LSTM
# =====================================================
lstm_model    = None
lstm_scaler_X = None
lstm_scaler_y = None
lstm_mae      = None  # loaded from training_metrics_lstm.json


# =====================================================
# LOAD MODELS
# =====================================================
def load_models():
    """Load both MLP and LSTM models with their scalers and metrics."""
    global mlp_model, mlp_scaler_X, mlp_scaler_y, mlp_mae
    global lstm_model, lstm_scaler_X, lstm_scaler_y, lstm_mae

    import tensorflow as tf
    import joblib
    import json

    model_folder = current_app.config['MODEL_FOLDER']

    # ── MLP ──────────────────────────────────────────
    mlp_model_path    = os.path.join(model_folder, 'air_quality_mlp.h5')
    mlp_scaler_X_path = os.path.join(model_folder, 'scaler_X_mlp.pkl')
    mlp_scaler_y_path = os.path.join(model_folder, 'scaler_y_mlp.pkl')
    mlp_metrics_path  = os.path.join(model_folder, 'training_metrics_mlp.json')

    try:
        if not os.path.exists(mlp_model_path):
            print(f"❌ [MLP] Model not found: {mlp_model_path}")
        else:
            mlp_model    = tf.keras.models.load_model(mlp_model_path)
            mlp_scaler_X = joblib.load(mlp_scaler_X_path)
            mlp_scaler_y = joblib.load(mlp_scaler_y_path)
            print(f"✅ [MLP] Model loaded | input: {mlp_model.input_shape} | output: {mlp_model.output_shape}")

            if os.path.exists(mlp_metrics_path):
                with open(mlp_metrics_path, 'r') as f:
                    m = json.load(f)
                mlp_mae = m['results']['test_mae_original']
                print(f"✅ [MLP] MAE (original scale): {mlp_mae}")
            else:
                mlp_mae = None
                print(f"⚠️ [MLP] Metrics file not found")
    except Exception as e:
        print(f"❌ [MLP] Load failed: {e}")
        mlp_model = mlp_scaler_X = mlp_scaler_y = None

    # ── LSTM ─────────────────────────────────────────
    lstm_model_path    = os.path.join(model_folder, 'air_quality_lstm.h5')
    lstm_scaler_X_path = os.path.join(model_folder, 'scaler_X_lstm.pkl')
    lstm_scaler_y_path = os.path.join(model_folder, 'scaler_y_lstm.pkl')
    lstm_metrics_path  = os.path.join(model_folder, 'training_metrics_lstm.json')

    try:
        if not os.path.exists(lstm_model_path):
            print(f"❌ [LSTM] Model not found: {lstm_model_path}")
        else:
            lstm_model    = tf.keras.models.load_model(lstm_model_path)
            lstm_scaler_X = joblib.load(lstm_scaler_X_path)
            lstm_scaler_y = joblib.load(lstm_scaler_y_path)
            print(f"✅ [LSTM] Model loaded | input: {lstm_model.input_shape} | output: {lstm_model.output_shape}")

            if os.path.exists(lstm_metrics_path):
                with open(lstm_metrics_path, 'r') as f:
                    m = json.load(f)
                lstm_mae = m['results']['test_mae_original']
                print(f"✅ [LSTM] MAE (original scale): {lstm_mae}")
            else:
                lstm_mae = None
                print(f"⚠️ [LSTM] Metrics file not found")
    except Exception as e:
        print(f"❌ [LSTM] Load failed: {e}")
        lstm_model = lstm_scaler_X = lstm_scaler_y = None


def is_mlp_ready():
    return mlp_model is not None and mlp_scaler_X is not None and mlp_scaler_y is not None


def is_lstm_ready():
    return lstm_model is not None and lstm_scaler_X is not None and lstm_scaler_y is not None


# =====================================================
# INPUT BUILDERS
# =====================================================
def _fetch_readings():
    """
    Fetch last 12 readings from Firebase.
    Returns list of 12 reading dicts, ordered oldest → newest.
    Raises ValueError if insufficient data.
    """
    from firebase_admin import db

    ref  = db.reference('/devices/esp32_001/readings')
    data = ref.order_by_key().limit_to_last(LOOKBACK).get()

    if not data or len(data) < LOOKBACK:
        raise ValueError(f'Insufficient data: need {LOOKBACK}, got {len(data) if data else 0}')

    return list(data.values())


def _build_mlp_input(readings):
    """
    Build MLP input: (1, 96) flat — 12 timesteps × 8 features.
    Applies scaler_X_mlp (96 means).
    """
    X = []
    for r in readings:
        for feat in FEATURES:
            X.append(r.get(feat, 0))

    X_arr    = np.array(X).reshape(1, -1)          # (1, 96)
    X_scaled = mlp_scaler_X.transform(X_arr)
    return X_scaled


def _build_lstm_input(readings):
    """
    Build LSTM input: (1, 12, 8) 3D — NOT flattened.
    Applies scaler_X_lstm (8 means, per-feature).
    """
    X = []
    for r in readings:
        row = [r.get(feat, 0) for feat in FEATURES]
        X.append(row)

    X_arr    = np.array(X)                          # (12, 8)
    X_2d     = X_arr.reshape(-1, len(FEATURES))     # (12, 8) — same, just explicit
    X_scaled = lstm_scaler_X.transform(X_2d)
    X_3d     = X_scaled.reshape(1, LOOKBACK, len(FEATURES))  # (1, 12, 8)
    return X_3d


# =====================================================
# SAFE INVERSE TRANSFORM
# =====================================================
def safe_inverse_transform(y_pred_scaled, scaler_y):
    """
    Inverse transform scaled predictions.
    Clips all features to >= 0 EXCEPT temp (index 6), which can be negative.
    """
    y_pred = scaler_y.inverse_transform(y_pred_scaled)  # (1, 8)
    for i in range(y_pred.shape[1]):
        if i != TEMP_IDX:
            y_pred[:, i] = np.maximum(y_pred[:, i], 0)
    return y_pred


# =====================================================
# HELPERS
# =====================================================
def _pred_to_dict(values):
    """Convert prediction array (8,) to feature dict."""
    return {feat: round(float(v), 4) for feat, v in zip(FEATURES, values)}


def classify_air_quality(pred_dict):
    """
    Classify air quality based on 8-feature prediction.
    Returns (label, level): label ∈ ['BAIK', 'WASPADA', 'BURUK'], level ∈ [0, 1, 2]
    """
    co    = pred_dict.get('CO',    0)
    pm25  = pred_dict.get('PM25',  0)
    pm10  = pred_dict.get('PM10',  0)
    no2   = pred_dict.get('NO2',   0)
    ozone = pred_dict.get('ozone', 0)

    if pm25 > 150 or co > 3.0 or pm10 > 350 or no2 > 400 or ozone > 240:
        return 'BURUK', 2
    elif pm25 > 55 or co > 1.5 or pm10 > 150 or no2 > 200 or ozone > 120:
        return 'WASPADA', 1
    else:
        return 'BAIK', 0


def calculate_confidence(mae, time_index=0):
    """
    Confidence score based on MAE magnitude and time decay.
    time_index 0 = current forecast step.
    """
    if mae is None:
        return 70.0
    max_mae      = 50.0   # upper bound for normalization (original scale)
    base         = max(50, min(95, (1 - mae / max_mae) * 100 * 1.1))
    decayed      = base * (1 - time_index * 0.02)
    return round(max(50, min(95, decayed)), 2)


def _metrics_payload(mae):
    return {
        'mae_original_scale': round(float(mae), 4) if mae is not None else None,
        'confidence':         calculate_confidence(mae, 0)
    }


def _latest_dict(reading):
    return {feat: round(float(reading.get(feat, 0)), 4) for feat in FEATURES}


# =====================================================
# ROUTES
# =====================================================

# ── Current classification (rule-based, no model needed) ──────────────
@prediction_bp.route('/prediction/latest', methods=['GET'])
def get_latest_prediction():
    """Current air quality status from latest Firebase reading."""
    try:
        from firebase_admin import db

        data = db.reference('/devices/esp32_001/latest').get()
        if not data:
            return jsonify({'status': 'error', 'message': 'No sensor data'}), 404

        current     = _latest_dict(data)
        label, lvl  = classify_air_quality(current)

        return jsonify({
            'status':         'success',
            'label':          label,
            'level':          lvl,
            'current_values': current,
            'timestamp':      data.get('timestamp', datetime.now().isoformat())
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── MLP single-step forecast (t+12, 1 hour ahead) ────────────────────
@prediction_bp.route('/forecast/mlp', methods=['GET'])
def get_mlp_forecast():
    """
    Predict all 8 air quality parameters 1 hour ahead using MLP.
    Input:  (1, 96) flat — 12 timesteps × 8 features
    Output: (1, 8)  — CO, CO2, PM25, PM10, NO2, ozone, temp, humidity at t+60min
    """
    try:
        if not is_mlp_ready():
            return jsonify({'status': 'error', 'message': 'MLP model not loaded'}), 503

        readings  = _fetch_readings()
        X_scaled  = _build_mlp_input(readings)

        y_scaled  = mlp_model.predict(X_scaled, verbose=0)
        y_pred    = safe_inverse_transform(y_scaled, mlp_scaler_y)
        pred_dict = _pred_to_dict(y_pred.flatten())

        label, lvl = classify_air_quality(pred_dict)
        current    = _latest_dict(readings[-1])

        return jsonify({
            'status':      'success',
            'model':       'MLP',
            'prediction':  {
                'label':       label,
                'level':       lvl,
                'values':      pred_dict,
                'target_time': (datetime.now() + timedelta(hours=1)).strftime('%H:%M')
            },
            'current':      current,
            'metrics':      _metrics_payload(mlp_mae),
            'generated_at': datetime.now().isoformat()
        })

    except ValueError as ve:
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── LSTM single-step forecast (t+12, 1 hour ahead) ───────────────────
@prediction_bp.route('/forecast/lstm', methods=['GET'])
def get_lstm_forecast():
    """
    Predict all 8 air quality parameters 1 hour ahead using LSTM.
    Input:  (1, 12, 8) 3D tensor — NOT flattened
    Output: (1, 8)  — CO, CO2, PM25, PM10, NO2, ozone, temp, humidity at t+60min
    """
    try:
        if not is_lstm_ready():
            return jsonify({'status': 'error', 'message': 'LSTM model not loaded'}), 503

        readings  = _fetch_readings()
        X_scaled  = _build_lstm_input(readings)

        y_scaled  = lstm_model.predict(X_scaled, verbose=0)
        y_pred    = safe_inverse_transform(y_scaled, lstm_scaler_y)
        pred_dict = _pred_to_dict(y_pred.flatten())

        label, lvl = classify_air_quality(pred_dict)
        current    = _latest_dict(readings[-1])

        return jsonify({
            'status':      'success',
            'model':       'LSTM',
            'prediction':  {
                'label':       label,
                'level':       lvl,
                'values':      pred_dict,
                'target_time': (datetime.now() + timedelta(hours=1)).strftime('%H:%M')
            },
            'current':      current,
            'metrics':      _metrics_payload(lstm_mae),
            'generated_at': datetime.now().isoformat()
        })

    except ValueError as ve:
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Side-by-side comparison ──────────────────────────────────────────
@prediction_bp.route('/forecast/compare', methods=['GET'])
def get_forecast_compare():
    """
    Run both MLP and LSTM on the same input, return predictions side-by-side.
    Useful for the thesis trade-off analysis.
    """
    try:
        neither_ready = not is_mlp_ready() and not is_lstm_ready()
        if neither_ready:
            return jsonify({'status': 'error', 'message': 'No models loaded'}), 503

        readings = _fetch_readings()
        current  = _latest_dict(readings[-1])
        result   = {'status': 'success', 'current': current, 'generated_at': datetime.now().isoformat()}

        if is_mlp_ready():
            X_mlp         = _build_mlp_input(readings)
            y_mlp_scaled  = mlp_model.predict(X_mlp, verbose=0)
            y_mlp         = safe_inverse_transform(y_mlp_scaled, mlp_scaler_y)
            mlp_dict      = _pred_to_dict(y_mlp.flatten())
            label_m, lv_m = classify_air_quality(mlp_dict)
            result['mlp'] = {
                'label':   label_m,
                'level':   lv_m,
                'values':  mlp_dict,
                'metrics': _metrics_payload(mlp_mae)
            }

        if is_lstm_ready():
            X_lstm        = _build_lstm_input(readings)
            y_lstm_scaled = lstm_model.predict(X_lstm, verbose=0)
            y_lstm        = safe_inverse_transform(y_lstm_scaled, lstm_scaler_y)
            lstm_dict     = _pred_to_dict(y_lstm.flatten())
            label_l, lv_l = classify_air_quality(lstm_dict)
            result['lstm'] = {
                'label':   label_l,
                'level':   lv_l,
                'values':  lstm_dict,
                'metrics': _metrics_payload(lstm_mae)
            }

        result['target_time'] = (datetime.now() + timedelta(hours=1)).strftime('%H:%M')
        return jsonify(result)

    except ValueError as ve:
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Model info ───────────────────────────────────────────────────────
@prediction_bp.route('/model/info', methods=['GET'])
def model_info():
    """Info for both loaded models."""
    info = {
        'features':    FEATURES,
        'n_features':  len(FEATURES),
        'lookback':    LOOKBACK,
        'temp_index':  TEMP_IDX
    }

    info['mlp'] = {
        'loaded':        is_mlp_ready(),
        'model_type':    'MLP',
        'input_shape':   str(mlp_model.input_shape)   if is_mlp_ready() else None,
        'output_shape':  str(mlp_model.output_shape)  if is_mlp_ready() else None,
        'total_params':  int(mlp_model.count_params()) if is_mlp_ready() else None,
        'mae_original':  round(float(mlp_mae), 4)     if mlp_mae else None,
        'architecture':  'Input(96) → Dense(64,ReLU) → Dense(32,ReLU) → Output(8,Linear)'
    }

    info['lstm'] = {
        'loaded':        is_lstm_ready(),
        'model_type':    'LSTM',
        'input_shape':   str(lstm_model.input_shape)   if is_lstm_ready() else None,
        'output_shape':  str(lstm_model.output_shape)  if is_lstm_ready() else None,
        'total_params':  int(lstm_model.count_params()) if is_lstm_ready() else None,
        'mae_original':  round(float(lstm_mae), 4)     if lstm_mae else None,
        'architecture':  'Input(12,8) → LSTM(64) → LSTM(32) → Dense(32,ReLU) → Output(8,Linear)'
    }

    return jsonify(info)