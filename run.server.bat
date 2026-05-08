@echo off
echo ================================================
echo   TA-AirQuality Flask Backend Server
echo ================================================
echo.

REM Check if virtual environment exists
if exist venv (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate
) else (
    echo [WARNING] Virtual environment not found
    echo [INFO] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo [INFO] Starting Flask server...
echo [INFO] Server will run at: http://localhost:5000
echo [INFO] Press Ctrl+C to stop the server
echo.

python app.py

pause