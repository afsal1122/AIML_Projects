@echo off
REM
REM This script creates a virtual environment, installs dependencies,
REM and runs the Streamlit app using the bundled sample data.
REM

SET VENV_DIR="venv"
SET PYTHON_EXEC="python"

REM Check for Python
%PYTHON_EXEC% --version > NUL 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Error: 'python' command not found or Python is not installed.
    echo Please install Python 3.9+ and add it to your PATH.
    pause
    exit /b 1
)

REM Create virtual environment
IF NOT EXIST %VENV_DIR% (
    echo Creating virtual environment at %VENV_DIR%...
    %PYTHON_EXEC% -m venv %VENV_DIR%
) ELSE (
    echo Virtual environment %VENV_DIR% already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
CALL %VENV_DIR%\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

REM Install requirements
echo Installing requirements from requirements.txt...
pip install -r requirements.txt

REM Run Streamlit app
echo =================================================
echo Setup complete!
echo Launching Streamlit app...
echo To stop, press Ctrl+C in this terminal.
echo =================================================
streamlit run app/streamlit_app.py

REM Deactivate on exit (optional)
CALL %VENV_DIR%\Scripts\deactivate.bat