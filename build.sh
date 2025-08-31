#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies if needed
apt-get update -y || true
apt-get install -y libgl1-mesa-glx libglib2.0-0 || true

# Install Python dependencies with pre-built wheels when possible
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir -r requirements.txt

# Create required directories
mkdir -p static/uploads
mkdir -p static/images/nft

# This is where models would be downloaded, handled by the app code
mkdir -p models

# Touch files to ensure directories exist
touch static/uploads/.gitkeep
touch static/images/nft/.gitkeep

echo "Build completed successfully!"
