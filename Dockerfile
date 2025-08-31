FROM python:3.9-slim

WORKDIR /app

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lite.txt .
RUN pip install --no-cache-dir -r requirements-lite.txt

# Create required directories
RUN mkdir -p static/uploads static/images/nft models

COPY . .

ENV PORT=8000

# Expose port and run the application
EXPOSE 8000
CMD gunicorn --bind 0.0.0.0:$PORT wsgi:app
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
