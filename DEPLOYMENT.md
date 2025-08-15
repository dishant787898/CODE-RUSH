# EcoSortAI Deployment Guide

This document provides comprehensive instructions for deploying the EcoSortAI web application in various environments.

## Prerequisites

- Python 3.9+ installed
- Git installed (optional, for version control)
- Docker installed (optional, for containerized deployment)

## Step 1: Set up the Environment

First, run the script to create necessary directories and placeholder images:

```bash
python create_dirs.py
```

### Create and configure .env file

Copy the example .env file and update it with your settings:

```bash
cp .env.example .env
```

Edit `.env` with a secure secret key.

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Choose a Deployment Method

### Method A: Windows Deployment (Recommended for Windows Users)

Windows users should use Waitress instead of Gunicorn:

```bash
# First ensure waitress is installed
pip install waitress

# Then run the application with waitress
python run_waitress.py
```

Access at http://localhost:8000

### Method B: Local Development

Run directly with Flask (for development only):

```bash
python app.py
```

Access at http://localhost:5000

### Method C: Production on Linux/macOS with Gunicorn (Not for Windows)

> ⚠️ Note: Gunicorn is not compatible with Windows. Windows users should use Method A above.

```bash
# For Linux/macOS only
pip install gunicorn
gunicorn --bind 0.0.0.0:8000 wsgi:app
```

Access at http://localhost:8000

### Method D: Docker Deployment (Works on all platforms with Docker installed)

Build and run with Docker Compose:

```bash
docker-compose build
docker-compose up -d
```

Access at http://localhost:5000

## Step 4: Heroku Deployment

1. Install Heroku CLI and login:
   ```bash
   heroku login
   ```

2. Create a new Heroku app:
   ```bash
   heroku create ecosort-ai
   ```

3. Configure environment variables:
   ```bash
   heroku config:set SECRET_KEY=your_secure_secret_key_change_this
   ```

4. Push your code to Heroku:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push heroku master
   ```

5. Ensure at least one instance is running:
   ```bash
   heroku ps:scale web=1
   ```

## Step 5: Configure your Domain (Optional)

### Using Nginx as a Reverse Proxy

1. Install Nginx:
   ```bash
   sudo apt update
   sudo apt install nginx
   ```

2. Create a configuration file:
   ```
   server {
       listen 80;
       server_name yourdomain.com www.yourdomain.com;

       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. Enable the site and restart Nginx:
   ```bash
   sudo ln -s /etc/nginx/sites-available/ecosortai /etc/nginx/sites-enabled/
   sudo systemctl restart nginx
   ```

## Step 6: Set Up SSL with Certbot (Optional)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed with `pip install -r requirements.txt`

2. **Port already in use**: Change the port or kill the process using the port:
   ```bash
   # Find process
   netstat -ano | findstr :5000
   # Kill process (replace PID with actual process ID)
   taskkill /F /PID PID
   ```

3. **Model loading errors**: Verify model paths in the application and ensure models are present in the models directory.

### Health Check

Test the API endpoint to verify the application is running:
```bash
curl http://localhost:5000/
```
