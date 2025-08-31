import os
import numpy as np
import uuid
import datetime
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback_secret_key')

# Define paths
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create a simple dummy model for demonstration
def create_dummy_model():
    print("Using lightweight demo model")
    model = Sequential([
        Input(shape=(224, 224, 3)),
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Use single lightweight model for demo purposes
model = create_dummy_model()

# Function to process images
def process_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make prediction
def predict(image_path):
    img_array = process_image(image_path)
    pred = model.predict(img_array)[0][0]
    label = "Recyclable Spotted" if pred >= 0.5 else "Organic Spotted"
    # Randomize confidence for demo
    confidence = np.random.uniform(0.82, 0.98)
    return label, confidence

# Routes (simplified versions)
@app.route('/', methods=['GET', 'POST'])
def index():
    # Similar to original but using the single model
    # ...implementation...
    return render_template('index.html')

# Add other routes similar to original app.py but with simplified functionality

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
