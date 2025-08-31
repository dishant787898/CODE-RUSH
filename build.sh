#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Starting build process..."

# Force Python version if available
if command -v pyenv 1>/dev/null 2>&1; then
  echo "Using pyenv to set Python version..."
  pyenv install -s 3.9.16
  pyenv local 3.9.16
fi

# Create required directories
mkdir -p static/uploads
mkdir -p static/images/nft
mkdir -p models

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Build completed successfully!"
# Touch files to ensure directories exist
touch static/uploads/.gitkeep
touch static/images/nft/.gitkeep

echo "Build completed successfully!"
