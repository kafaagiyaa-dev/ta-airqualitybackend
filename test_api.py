"""
Test script untuk Flask API
Run this to test all endpoints
"""

import requests
import json
from datetime import datetime
import time

# Base URL
BASE_URL = "http://localhost:5000"

def print_response(response, title):
    """Print formatted response"""
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    print("="*60)

def test_health():
    """Test health check endpoint"""
    response = requests.get(f"{BASE_URL}/api/health")
    print_response(response, "Health Check")

def test_post_sensor_data():
    """Test posting sensor data (simulate ESP32)"""
    data = {
        "device_id": "ESP32_TEST",
        "mq135_ppm": 150.5,
        "mq7_ppm": 25.3,
        "pm25": 35,
        "pm10": 50,
        "timestamp": int(time.time())
    }
    
    response = requests.post(
        f"{BASE_URL}/api/sensor-data",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print_response(response, "POST Sensor Data (Simulated ESP32)")

def test_get_latest():
    """Test get latest data"""
    response = requests.get(f"{BASE_URL}/api/sensor-data/latest")
    print_response(response, "GET Latest Sensor Data")

def test_get_history():
    """Test get historical data"""
    response = requests.get(f"{BASE_URL}/api/sensor-data/history?limit=10")
    print_response(response, "GET History (Last 10 records)")

def test_get_stats():
    """Test get statistics"""
    response = requests.get(f"{BASE_URL}/api/stats")
    print_response(response, "GET Statistics")

def test_prediction():
    """Test AI prediction"""
    data = {
        "mq135_ppm": 200.0,
        "mq7_ppm": 35.0,
        "pm25": 50,
        "pm10": 70
    }
    
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print_response(response, "POST Prediction (AI Model)")

def test_model_info():
    """Test model info"""
    response = requests.get(f"{BASE_URL}/api/model/info")
    print_response(response, "GET Model Info")

def run_all_tests():
    """Run all tests"""
    print("\n" + "🚀 STARTING API TESTS ".center(60, "="))
    print(f"Testing API at: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        # Basic tests
        test_health()
        
        # Post some test data
        print("\n📤 Posting test data...")
        for i in range(3):
            test_post_sensor_data()
            time.sleep(0.5)
        
        # Get data tests
        test_get_latest()
        test_get_history()
        test_get_stats()
        
        # AI prediction tests
        test_prediction()
        test_model_info()
        
        print("\n" + "✅ ALL TESTS COMPLETED ".center(60, "="))
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to Flask server!")
        print(f"Make sure Flask is running at {BASE_URL}")
        print("Run: python app.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")

if __name__ == "__main__":
    run_all_tests()