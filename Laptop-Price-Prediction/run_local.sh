#!/bin/bash
#
# This script creates a virtual environment, installs dependencies,
# and runs the Streamlit app using the bundled sample data.

# Exit immediately if a command exits with a non-zero status.
set -e

# Define virtual environment directory
VENV_DIR="venv"

echo "Looking for Python 3..."
# Find python3 executable
if command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
elif command -v python &>/dev/null; then
    PYTHON_EXEC="python"
else
    echo "Error: Python 3 is not installed or not in PATH."
    exit 1
fi
echo "Using $($PYTHON_EXEC --version)"

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    $PYTHON_EXEC -m venv $VENV_DIR
else
    echo "Virtual environment $VENV_DIR already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Run Streamlit app
echo "================================================="
echo "Setup complete!"
echo "Launching Streamlit app..."
echo "To stop, press Ctrl+C in this terminal."
echo "================================================="
streamlit run app/streamlit_app.py

# Deactivate on exit (optional)
deactivate