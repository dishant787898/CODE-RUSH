import os
import sys
import requests
import hashlib
import zipfile
import shutil
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf

# Define the Cast layer to match what's in app.py
class Cast(Layer):
    def __init__(self, dtype='float32', **kwargs):
        super(Cast, self).__init__(**kwargs)
        self._dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, dtype=self._dtype)

    def get_config(self):
        config = super(Cast, self).get_config()
        config.update({'dtype': self._dtype})
        return config

    @property
    def dtype(self):
        return self._dtype

# Model info: name, URL, and expected MD5 hash for verification
MODELS = {
    "vgg16_waste_classification_tf.h5": {
        "url": "https://example.com/models/vgg16_waste_classification_tf.h5",
        "md5": "00000000000000000000000000000000",  # Replace with actual MD5
        "size_mb": 85
    },
    "waste_classification22_model.h5": {
        "url": "https://example.com/models/waste_classification22_model.h5",
        "md5": "00000000000000000000000000000000",  # Replace with actual MD5
        "size_mb": 97
    },
    "inceptionv3_waste_classification_tf.h5": {
        "url": "https://example.com/models/inceptionv3_waste_classification_tf.h5",
        "md5": "00000000000000000000000000000000",  # Replace with actual MD5
        "size_mb": 90
    }
}

def create_models_dir():
    """Create models directory if it doesn't exist"""
    os.makedirs("models", exist_ok=True)
    print("✓ Models directory created/verified")

def download_file(url, filename, expected_md5=None):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(filename, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(filename)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                f.write(data)
                bar.update(len(data))
        
        # Verify MD5 if provided
        if expected_md5:
            file_md5 = hashlib.md5(open(filename, 'rb').read()).hexdigest()
            if file_md5 != expected_md5:
                print(f"⚠️ Warning: MD5 checksum mismatch for {filename}")
                print(f"Expected: {expected_md5}")
                print(f"Got: {file_md5}")
                return False
        
        return True
    
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def extract_models_if_zip(zip_path):
    """Extract models from zip file if provided"""
    if not zip_path.endswith('.zip'):
        return False
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"Extracting models from {zip_path}...")
            zip_ref.extractall("models")
        return True
    except Exception as e:
        print(f"❌ Error extracting models: {e}")
        return False

def verify_models():
    """Verify that all required models exist and can be loaded"""
    missing_models = []
    valid_models = []
    
    for model_file in MODELS:
        model_path = os.path.join("models", model_file)
        
        if not os.path.exists(model_path):
            missing_models.append(model_file)
            continue
        
        # Try to load model
        try:
            custom_objects = {'Cast': Cast}
            model = load_model(model_path, custom_objects=custom_objects)
            valid_models.append(model_file)
            print(f"✓ Successfully loaded: {model_file}")
        except Exception as e:
            print(f"❌ Error loading {model_file}: {e}")
            missing_models.append(model_file)
    
    return valid_models, missing_models

def copy_models_from_path(source_path):
    """Copy models from a specified directory"""
    if not os.path.isdir(source_path):
        print(f"❌ {source_path} is not a directory")
        return False
    
    success = False
    for model_file in MODELS:
        source_file = os.path.join(source_path, model_file)
        if os.path.exists(source_file):
            try:
                shutil.copy2(source_file, os.path.join("models", model_file))
                print(f"✓ Copied {model_file} from {source_path}")
                success = True
            except Exception as e:
                print(f"❌ Error copying {model_file}: {e}")
    
    return success

def create_dummy_models():
    """Create minimal dummy models for testing purposes"""
    print("\nCreating minimal dummy models for testing...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    
    for model_file in MODELS:
        try:
            # Create a very basic model
            model = Sequential([
                Input(shape=(224, 224, 3)),
                Cast(dtype='float32'),
                tf.keras.layers.GlobalAveragePooling2D(),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy')
            model.save(os.path.join("models", model_file))
            print(f"✓ Created dummy model: {model_file}")
        except Exception as e:
            print(f"❌ Error creating dummy model {model_file}: {e}")

def main():
    print("===== EcoSortAI Model Setup =====")
    
    # Create models directory
    create_models_dir()
    
    # Check if models already exist
    valid_models, missing_models = verify_models()
    
    if len(valid_models) == len(MODELS):
        print("\n✅ All models are already set up and valid!")
        return
    
    print(f"\nMissing models: {', '.join(missing_models)}")
    
    # Ask user how they want to provide the models
    print("\nHow would you like to provide the missing models?")
    print("1. Download from remote server (requires internet)")
    print("2. Copy from local directory")
    print("3. Extract from ZIP file")
    print("4. Create dummy models (for testing only)")
    print("q. Quit")
    
    choice = input("\nEnter choice [1-4 or q]: ")
    
    if choice == "1":
        print("\nDownloading models...")
        for model_file in missing_models:
            model_info = MODELS[model_file]
            download_file(
                model_info["url"], 
                os.path.join("models", model_file),
                model_info["md5"]
            )
    
    elif choice == "2":
        source_path = input("Enter the directory containing the model files: ")
        copy_models_from_path(source_path)
    
    elif choice == "3":
        zip_path = input("Enter the path to the ZIP file containing the models: ")
        extract_models_if_zip(zip_path)
    
    elif choice == "4":
        create_dummy_models()
    
    elif choice.lower() == "q":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Exiting...")
        return
    
    # Verify models again
    valid_models, missing_models = verify_models()
    
    if len(valid_models) == len(MODELS):
        print("\n✅ All models are now set up and valid!")
    else:
        print(f"\n⚠️ There are still missing models: {', '.join(missing_models)}")
        print("Please make sure to provide all required models before running the application.")

if __name__ == "__main__":
    # Check if TensorFlow is installed
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow is not installed. Please install it first.")
        sys.exit(1)
        
    # Check if we have tqdm for progress bars
    try:
        import tqdm
    except ImportError:
        print("Installing tqdm for progress bars...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        
    main()
