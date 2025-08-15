import os
import sys
import platform
import subprocess

def check_python_version():
    major, minor, _ = sys.version_info
    current_version = f"{major}.{minor}"
    print(f"Current Python version: {current_version}")
    
    if major == 3 and minor >= 11:
        print("\n⚠️ WARNING: TensorFlow has compatibility issues with Python 3.11+")
        print("It's recommended to use Python 3.9 or 3.10 for best compatibility.")
        
        if platform.system() == "Windows":
            print("\nFor Windows, you can:")
            print("1. Install Python 3.10 from https://www.python.org/downloads/release/python-3109/")
            print("2. Create a virtual environment with: python3.10 -m venv venv")
        else:
            print("\nFor Linux/macOS:")
            print("1. Install Python 3.10 using your package manager")
            print("2. Create a virtual environment: python3.10 -m venv venv")
        
        choice = input("\nDo you want to continue with the current version anyway? (y/n): ")
        if choice.lower() != 'y':
            print("Setup aborted. Please install a compatible Python version.")
            sys.exit(1)
    
    return True

def create_virtual_environment():
    if os.path.exists("venv"):
        print("Virtual environment already exists")
    else:
        print("Creating virtual environment...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", "venv"])
            print("Virtual environment created successfully!")
        except Exception as e:
            print(f"Failed to create virtual environment: {e}")
            sys.exit(1)

def install_dependencies():
    print("Installing dependencies...")
    
    # Get the correct pip path based on platform
    if platform.system() == "Windows":
        pip_path = os.path.join("venv", "Scripts", "pip")
    else:
        pip_path = os.path.join("venv", "bin", "pip")
    
    try:
        # Upgrade pip first
        subprocess.check_call([pip_path, "install", "--upgrade", "pip"])
        
        # Install dependencies
        subprocess.check_call([pip_path, "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except Exception as e:
        print(f"Failed to install dependencies: {e}")
        
        # Additional troubleshooting for TensorFlow
        if "tensorflow" in str(e).lower():
            print("\nTensorFlow installation error. Trying alternative installation...")
            try:
                # Try installing an older version of TensorFlow
                subprocess.check_call([pip_path, "install", "tensorflow==2.10.0"])
                print("TensorFlow installed successfully using alternative version!")
            except:
                print("\n⚠️ TensorFlow installation still failed. Try these steps:")
                print("1. Manually install TensorFlow: venv/Scripts/pip install tensorflow==2.10.0")
                print("2. If that fails, consider using a CPU-only version: venv/Scripts/pip install tensorflow-cpu==2.10.0")
                print("3. For Python 3.11+, you might need to use TensorFlow 2.11+ or use Python 3.10")

def setup_directories():
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/images/nft", exist_ok=True)
    print("Created necessary directories!")

def main():
    print("===== EcoSortAI Environment Setup =====")
    check_python_version()
    create_virtual_environment()
    install_dependencies()
    setup_directories()
    
    print("\n✅ Setup completed! To activate the environment:")
    if platform.system() == "Windows":
        print("    venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")
    
    print("\nThen run the application with: python app.py")

if __name__ == "__main__":
    main()
