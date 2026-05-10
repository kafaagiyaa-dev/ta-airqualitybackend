"""
Sensor Routes - Data dari Firebase
8 features: CO, CO2, PM25, PM10, NO2, ozone, temp, humidity
"""

from flask import Blueprint, jsonify, request
from datetime import datetime

sensor_bp = Blueprint('sensor', __name__)

# =====================================================
# CONSTANTS
# =====================================================
FEATURES     = ['CO', 'CO2', 'PM25', 'PM10', 'NO2', 'ozone', 'temp', 'humidity']
NON_NEGATIVE = ['CO', 'CO2', 'PM25', 'PM10', 'NO2', 'ozone', 'humidity']  # temp excluded


# =====================================================
# HELPERS
# =====================================================
def _extract_features(reading):
    """Extract all 8 features from a Firebase reading dict."""
    return {feat: reading.get(feat) for feat in FEATURES}


def _classify_status(co, co2, pm25, pm10, no2, ozone):
    """
    Rule-based air quality status using 6 pollutant parameters.
    temp and humidity are environmental — not used for AQI classification.
    Returns one of: GOOD, MODERATE, UNHEALTHY, HAZARDOUS
    """
    if (co > 35 or co2 > 2000 or pm25 > 150 or pm10 > 350
            or no2 > 400 or ozone > 240):
        return 'HAZARDOUS'
    elif (co > 9 or co2 > 800 or pm25 > 76 or pm10 > 150
            or no2 > 200 or ozone > 120):
        return 'UNHEALTHY'
    elif (co > 5 or co2 > 600 or pm25 > 35 or pm10 > 50
            or no2 > 100 or ozone > 60):
        return 'MODERATE'
    else:
        return 'GOOD'


# =====================================================
# ROUTES
# =====================================================

@sensor_bp.route('/sensor-data/latest', methods=['GET'])
def get_latest():
    """Get latest sensor data (all 8 features) dari Firebase."""
    try:
        from firebase_admin import db

        data = db.reference('/devices/esp32_001/latest').get()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data available'}), 404

        features = _extract_features(data)
        status   = _classify_status(
            co=data.get('CO', 0),    co2=data.get('CO2', 0),
            pm25=data.get('PM25', 0), pm10=data.get('PM10', 0),
            no2=data.get('NO2', 0),  ozone=data.get('ozone', 0)
        )

        return jsonify({
            'status': 'success',
            'data': {
                **features,
                'pm_valid':          data.get('pm_valid', False),
                'timestamp':         data.get('timestamp'),
                'air_quality_status': status
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@sensor_bp.route('/sensor-data/history', methods=['GET'])
def get_history():
    """Get historical data (all 8 features) dari Firebase."""
    try:
        from firebase_admin import db

        limit = request.args.get('limit', 100, type=int)

        data = db.reference('/devices/esp32_001/readings') \
                 .order_by_key().limit_to_last(limit).get()

        if not data:
            return jsonify({'status': 'error', 'message': 'No historical data'}), 404

        history = []
        for _, reading in data.items():
            history.append({
                'timestamp':   reading.get('timestamp'),
                **_extract_features(reading),
                'pm_valid':    reading.get('pm_valid')
            })

        return jsonify({
            'status': 'success',
            'count':  len(history),
            'data':   history
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@sensor_bp.route('/sensor-data/export', methods=['GET'])
def export_to_csv():
    """
    Export Firebase data to CSV for model training.
    Columns match FEATURES expected by prepare_sequences_MLP.py and prepare_sequences_LSTM.py:
      timestamp, timestamp_unix, CO, CO2, PM25, PM10, NO2, ozone, temp, humidity, pm_valid
    """
    try:
        from firebase_admin import db
        import pandas as pd

        data = db.reference('/devices/esp32_001/readings').get()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data to export'}), 404

        records = []
        for _, reading in data.items():
            record = {
                'timestamp':       reading.get('timestamp'),
                'timestamp_unix':  reading.get('timestamp_unix'),
                **_extract_features(reading),
                'pm_valid':        reading.get('pm_valid')
            }
            records.append(record)

        df = pd.DataFrame(records)
        df = df.sort_values('timestamp_unix')

        # Validate all 8 feature columns are present
        missing_cols = [f for f in FEATURES if f not in df.columns]
        if missing_cols:
            return jsonify({
                'status':  'error',
                'message': f'Missing feature columns in Firebase data: {missing_cols}',
                'note':    'Ensure ESP32 firmware is pushing all 8 features.'
            }), 500

        import os
        data_folder = current_app.config.get(
            'DATA_FOLDER',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        )
        os.makedirs(data_folder, exist_ok=True)

        output_file = os.path.join(
            data_folder,
            f'sensor_data_raw.csv'           # fixed name — matches prepare_sequences scripts
        )
        df.to_csv(output_file, index=False)

        return jsonify({
            'status':  'success',
            'message': f'Exported {len(df)} records',
            'file':    output_file,
            'columns': list(df.columns),
            'date_range': {
                'start': df['timestamp'].min(),
                'end':   df['timestamp'].max()
            },
            'feature_coverage': {
                feat: int(df[feat].notna().sum()) for feat in FEATURES
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@sensor_bp.route('/sensor-info', methods=['GET'])
def get_sensor_info():
    """Sensor threshold information for all 8 parameters."""
    return jsonify({
        'status': 'success',
        'features': FEATURES,
        'thresholds': {
            'CO': {
                'unit':      'ppm',
                'good':      '< 5',
                'moderate':  '5 – 9',
                'unhealthy': '9 – 35',
                'hazardous': '> 35'
            },
            'CO2': {
                'unit':      'ppm',
                'good':      '< 600',
                'moderate':  '600 – 800',
                'unhealthy': '800 – 2000',
                'hazardous': '> 2000'
            },
            'PM25': {
                'unit':      'μg/m³',
                'good':      '< 35',
                'moderate':  '35 – 76',
                'unhealthy': '76 – 150',
                'hazardous': '> 150'
            },
            'PM10': {
                'unit':      'μg/m³',
                'good':      '< 50',
                'moderate':  '50 – 150',
                'unhealthy': '150 – 350',
                'hazardous': '> 350'
            },
            'NO2': {
                'unit':      'μg/m³',
                'good':      '< 100',
                'moderate':  '100 – 200',
                'unhealthy': '200 – 400',
                'hazardous': '> 400'
            },
            'ozone': {
                'unit':      'μg/m³',
                'good':      '< 60',
                'moderate':  '60 – 120',
                'unhealthy': '120 – 240',
                'hazardous': '> 240'
            },
            'temp': {
                'unit': '°C',
                'note': 'Environmental parameter — not used for AQI classification'
            },
            'humidity': {
                'unit': '%',
                'note': 'Environmental parameter — not used for AQI classification'
            }
        }
    })