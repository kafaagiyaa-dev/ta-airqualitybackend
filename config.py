"""
Flask Configuration File
"""

import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # Folders
    BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
    DATA_FOLDER  = os.path.join(BASE_DIR, 'data')
    MODEL_FOLDER = os.path.join(BASE_DIR, 'models')

    # API Settings
    JSON_SORT_KEYS             = False
    JSONIFY_PRETTYPRINT_REGULAR = True

    # CORS Settings
    CORS_HEADERS = 'Content-Type'

    # Model Settings
    MODEL_NAME_MLP  = 'air_quality_mlp.h5'
    MODEL_NAME_LSTM = 'air_quality_lstm.h5'

    # Scaler Settings
    SCALER_X_MLP  = 'scaler_X_mlp.pkl'
    SCALER_Y_MLP  = 'scaler_y_mlp.pkl'
    SCALER_X_LSTM = 'scaler_X_lstm.pkl'
    SCALER_Y_LSTM = 'scaler_y_lstm.pkl'

    # Data Settings
    CSV_FILE   = 'sensor_data_raw.csv'
    EXCEL_FILE = 'sensor_log.xlsx'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG   = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG   = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG   = True
    TESTING = True


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production':  ProductionConfig,
    'testing':     TestingConfig,
    'default':     DevelopmentConfig
}