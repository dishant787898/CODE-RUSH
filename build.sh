#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Create required directories
mkdir -p static/uploads
mkdir -p static/images/nft

# This is where models would be downloaded, handled by the app code
mkdir -p models
