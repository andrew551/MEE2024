#!/bin/bash
# Run MEE2024, automatically creating a virtualenv if needed

APP_NAME="mee2024"
ENV_DIR="$HOME/.mee2024env"

# Check if virtualenv exists
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment at $ENV_DIR..."
    python3 -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"
    echo "Installing MEE2024 via pip..."
    pip install --upgrade pip
    pip install git+https://github.com/andrew551/MEE2024.git
else
    source "$ENV_DIR/bin/activate"
fi

# Run the app
echo "Launching MEE2024..."
mee2024
