import os
import webbrowser
from threading import Timer
from waitress import serve
from app import app

def open_browser():
    """Open the browser after a short delay"""
    webbrowser.open('http://localhost:8000')

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/images/nft', exist_ok=True)
    
    port = 8000
    host = '0.0.0.0'
    
    print("="*60)
    print(f"Starting EcoSortAI on http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    # Open browser automatically
    Timer(1.5, open_browser).start()
    
    # Start waitress server
    serve(app, host=host, port=port, threads=4)
