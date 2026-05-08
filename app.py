"""
TA-AirQuality Flask Backend
Main application file - FIREBASE + ML FORECASTER
"""

from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import os
import json
import logging

# Import routes
from routes.sensor_routes import sensor_bp
from routes.prediction_routes import prediction_bp, load_model, is_model_ready

app = Flask(__name__)

# =================== LOGGING ===================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =================== CORS ===================
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type"]
    }
})

# =================== CONFIG ===================
BASE_DIR = os.path.dirname(__file__)

app.config['SECRET_KEY'] = 'change-this-in-production'
app.config['DATA_FOLDER'] = os.path.join(BASE_DIR, 'data')
app.config['MODEL_FOLDER'] = os.path.join(BASE_DIR, 'models')

os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# =================== FIREBASE INIT ===================
try:
    # Prioritas 1: environment variable (Railway/production)
    service_account_env = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

    if service_account_env:
        service_account_info = json.loads(service_account_env)
        cred = credentials.Certificate(service_account_info)
        logger.info("✅ Firebase credential loaded from environment variable")

    else:
        # Prioritas 2: file lokal (development/localhost)
        cred_path = os.path.join(BASE_DIR, 'serviceAccountKey.json')
        if not os.path.exists(cred_path):
            raise FileNotFoundError(
                "serviceAccountKey.json not found dan "
                "FIREBASE_SERVICE_ACCOUNT env variable tidak di-set"
            )
        cred = credentials.Certificate(cred_path)
        logger.info("✅ Firebase credential loaded from serviceAccountKey.json")

    firebase_admin.initialize_app(cred, {
        'databaseURL': os.environ.get(
            "FIREBASE_DATABASE_URL",
            "https://ta-airquality-default-rtdb.firebaseio.com"
        )
    })

    logger.info("✅ Firebase initialized")

except Exception as e:
    logger.error(f"❌ Firebase init failed: {e}")
    raise

# =================== BLUEPRINTS ===================
app.register_blueprint(sensor_bp, url_prefix='/api')
app.register_blueprint(prediction_bp, url_prefix='/api')

# =================== ROUTES ===================
@app.route('/')
def index():
    return jsonify({
        'status': 'online',
        'service': 'TA-AirQuality API',
        'version': '3.0.0',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/health')
def health_check():
    try:
        ref = db.reference('/devices/esp32_001/latest')
        latest = ref.get()

        return jsonify({
            'status': 'healthy',
            'firebase_connected': latest is not None,
            'model_loaded': is_model_ready(),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


# =================== STARTUP ===================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 TA-AirQuality Backend Starting")
    print("=" * 60)

    with app.app_context():
        logger.info("🔄 Loading ML model...")
        if load_model():
            logger.info("✅ ML forecaster loaded")
        else:
            logger.warning("⚠️ ML model not found - using rule-based classification")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )