@echo off
title Network Packet Classifier - Admin Mode
color 0A

echo ========================================
echo    NETWORK PACKET CLASSIFIER
echo    Administrator Mode
echo ========================================
echo.

:: Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ ERROR: Administrator privileges required!
    echo.
    echo 💡 Please run this script as Administrator:
    echo    1. Right-click on this file
    echo    2. Select "Run as administrator"
    echo    3. Click "Yes" in the UAC prompt
    echo.
    pause
    exit /b 1
)

echo ✅ Running with Administrator privileges
echo 📡 Real packet capture: ENABLED
echo.

:: Change to the script's directory
cd /d "%~dp0"

:: Check if model files exist
if not exist "model.pkl" (
    echo ⚠️  WARNING: Model file not found!
    echo 💡 Please train the model first by running:
    echo    python train_model.py
    echo.
    choice /C YN /M "Do you want to train the model now?"
    if errorlevel 2 (
        echo.
        echo ❌ Cannot start without model files.
        pause
        exit /b 1
    )
    echo.
    echo 🚀 Training model...
    python train_model.py
    echo.
)

echo 🚀 Starting Network Packet Classifier...
echo 🌐 Web Interface: http://localhost:5000
echo 📡 Real-time packet capture: ACTIVE
echo 🎮 Simulation mode also available
echo.
echo Press Ctrl+C to stop the application
echo.

:: Start the application
python app.py

echo.
echo Application stopped.
pause