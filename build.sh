#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Starting build process..."

# Create required directories
mkdir -p static/uploads
mkdir -p static/images/nft
mkdir -p models

# Install Python dependencies with more flexible versioning
pip install --upgrade pip
pip install -r requirements.txt

echo "Build completed successfully!"
# This is where models would be downloaded, handled by the app code
mkdir -p models

# Touch files to ensure directories exist
touch static/uploads/.gitkeep
touch static/images/nft/.gitkeep

echo "Build completed successfully!"
