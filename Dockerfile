FROM python:3.9.16-slim

WORKDIR /app

# Install system dependencies with updated package names
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with specific versions
RUN pip install --no-cache-dir --upgrade pip setuptools==65.6.3 wheel==0.40.0 && \
    pip install --no-cache-dir -r requirements.txt

# Create required directories
RUN mkdir -p static/uploads static/images/nft models

# Copy application files
COPY . .

# Set environment variables
ENV ENVIRONMENT=production
ENV PORT=8000

# Run the download script as part of the build
RUN python download_models.py

# Expose port for the application
EXPOSE 8000

# Start command
CMD gunicorn --bind 0.0.0.0:$PORT app:app
