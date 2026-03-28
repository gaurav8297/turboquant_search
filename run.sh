#!/bin/bash
# Run TurboQuant Search demo
# Creates a virtual environment and installs dependencies on first run.

set -e
cd "$(dirname "$0")"

VENV_DIR=".venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Installing dependencies..."
    "$VENV_DIR/bin/pip" install --upgrade pip -q
    "$VENV_DIR/bin/pip" install -r requirements.txt -q
    echo "Done. Starting app..."
fi

# Activate and run
exec "$VENV_DIR/bin/python" app.py
