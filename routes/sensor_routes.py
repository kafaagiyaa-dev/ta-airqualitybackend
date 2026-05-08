"""
Sensor Routes - Data dari Firebase (BUKAN terima POST dari ESP32)
"""

from flask import Blueprint, jsonify, request
from datetime import datetime

sensor_bp = Blueprint('sensor', __name__)


@sensor_bp.route('/sensor-data/latest', methods=['GET'])
def get_latest():
    """Get latest sensor data dari Firebase"""
    try:
        from firebase_admin import db
        
        ref = db.reference('/devices/esp32_001/latest')
        data = ref.get()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data available'
            }), 404
        
        # Add air quality status
        co = data.get('CO', 0)
        co2 = data.get('CO2', 0)
        pm25 = data.get('PM25', 0)
        pm10 = data.get('PM10', 0)
        
        # Determine status
        if co > 35 or co2 > 2000 or pm25 > 150 or pm10 > 350:
            status = "HAZARDOUS"
        elif co > 9 or co2 > 800 or pm25 > 76 or pm10 > 150:
            status = "UNHEALTHY"
        elif co > 5 or co2 > 600 or pm25 > 35 or pm10 > 50:
            status = "MODERATE"
        else:
            status = "GOOD"
        
        return jsonify({
            'status': 'success',
            'data': {
                'CO': co,
                'CO2': co2,
                'PM25': pm25,
                'PM10': pm10,
                'pm_valid': data.get('pm_valid', False),
                'timestamp': data.get('timestamp'),
                'air_quality_status': status
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@sensor_bp.route('/sensor-data/history', methods=['GET'])
def get_history():
    """Get historical data dari Firebase"""
    try:
        from firebase_admin import db
        
        limit = request.args.get('limit', 100, type=int)
        
        ref = db.reference('/devices/esp32_001/readings')
        data = ref.order_by_key().limit_to_last(limit).get()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No historical data'
            }), 404
        
        # Convert to list
        history = []
        for timestamp_key, reading in data.items():
            history.append({
                'timestamp': reading.get('timestamp'),
                'CO': reading.get('CO'),
                'CO2': reading.get('CO2'),
                'PM25': reading.get('PM25'),
                'PM10': reading.get('PM10'),
                'pm_valid': reading.get('pm_valid')
            })
        
        return jsonify({
            'status': 'success',
            'count': len(history),
            'data': history
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@sensor_bp.route('/sensor-data/export', methods=['GET'])
def export_to_csv():
    """Export Firebase data to CSV (untuk training)"""
    try:
        from firebase_admin import db
        import pandas as pd
        import os
        
        ref = db.reference('/devices/esp32_001/readings')
        data = ref.get()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data to export'
            }), 404
        
        # Convert to DataFrame
        records = []
        for timestamp_key, reading in data.items():
            records.append({
                'timestamp': reading.get('timestamp'),
                'timestamp_unix': reading.get('timestamp_unix'),
                'CO': reading.get('CO'),
                'CO2': reading.get('CO2'),
                'PM25': reading.get('PM25'),
                'PM10': reading.get('PM10'),
                'pm_valid': reading.get('pm_valid')
            })
        
        df = pd.DataFrame(records)
        
        # Sort by timestamp
        df = df.sort_values('timestamp_unix')
        
        # Save to CSV
        data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
        os.makedirs(data_folder, exist_ok=True)
        
        output_file = os.path.join(data_folder, f'sensor_readings_{datetime.now().strftime("%Y%m%d")}.csv')
        df.to_csv(output_file, index=False)
        
        return jsonify({
            'status': 'success',
            'message': f'Exported {len(df)} records',
            'file': output_file,
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@sensor_bp.route('/sensor-info', methods=['GET'])
def get_sensor_info():
    """Get sensor threshold information"""
    return jsonify({
        'status': 'success',
        'thresholds': {
            'CO': {
                'unit': 'ppm',
                'good': '< 5',
                'moderate': '5 - 9',
                'unhealthy': '9 - 35',
                'hazardous': '> 35'
            },
            'CO2': {
                'unit': 'ppm',
                'good': '< 600',
                'moderate': '600 - 800',
                'unhealthy': '800 - 2000',
                'hazardous': '> 2000'
            },
            'PM25': {
                'unit': 'μg/m³',
                'good': '< 35',
                'moderate': '35 - 76',
                'unhealthy': '76 - 150',
                'hazardous': '> 150'
            },
            'PM10': {
                'unit': 'μg/m³',
                'good': '< 50',
                'moderate': '50 - 150',
                'unhealthy': '150 - 350',
                'hazardous': '> 350'
            }
        }
    })