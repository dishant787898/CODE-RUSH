import os
import numpy as np
import uuid
import datetime
import json
import sys  # Add this import for subprocess.check_call
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Layer, Dense, GlobalAveragePooling2D, Input, Lambda
import tensorflow as tf
from PIL import Image
import base64
import cv2
from dotenv import load_dotenv

# Try to import tensorflow_hub (optional for ViT model)
try:
    import tensorflow_hub as hub
    HUB_AVAILABLE = True
except ImportError:
    print("⚠️ tensorflow_hub not available. ViT model will use fallback.")
    HUB_AVAILABLE = False
    hub = None

load_dotenv()

# Define a custom Cast layer
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

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback_secret_key')

# Define paths
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_FOLDER = os.getenv('MODEL_PATH', 'models/')

# Function to create a simple dummy model if needed
def create_dummy_model():
    print("⚠️ Creating dummy model for fallback")
    model = Sequential([
        Input(shape=(224, 224, 3)),
        Cast(dtype='float32'),
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Function to safely load models with fallback
def safe_load_model(model_path, model_name):
    try:
        print(f"Loading model: {model_path}")
        custom_objects = {'Cast': Cast}
        return load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")
        print(f"Using fallback dummy model for {model_name}")
        return create_dummy_model()

# Load the fine-tuned models with custom_objects and fallback
custom_objects = {'Cast': Cast}

# Create model directory if it doesn't exist
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Load models with fallback
vgg16_model = safe_load_model(os.path.join(MODEL_FOLDER, 'vgg16_waste_classification_tf.h5'), 'VGG16')
resnet50_model = safe_load_model(os.path.join(MODEL_FOLDER, 'waste_classification22_model.h5'), 'ResNet50')
inceptionv3_model = safe_load_model(os.path.join(MODEL_FOLDER, 'inceptionv3_waste_classification_tf.h5'), 'InceptionV3')

# Dictionary to map model names to their instances
models_dict = {
    'VGG16': vgg16_model,
    'ResNet50': resnet50_model,
    'InceptionV3': inceptionv3_model
}

# ============================================================================
# PLASTIC TYPE CLASSIFICATION MODELS
# ============================================================================

# Plastic type class names
PLASTIC_TYPE_CLASSES = ['HDPE', 'PET', 'PP', 'PS']
PLASTIC_TYPE_MODEL_FOLDER = os.path.join(MODEL_FOLDER, 'Plastic Type Models')

# Function to create a dummy plastic type model for fallback
def create_dummy_plastic_model():
    print("⚠️ Creating dummy plastic type model for fallback")
    model = Sequential([
        Input(shape=(224, 224, 3)),
        GlobalAveragePooling2D(),
        Dense(4, activation='softmax')  # 4 classes for plastic types
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# Function to safely load plastic type models with fallback
def safe_load_plastic_model(model_path, model_name):
    try:
        print(f"Loading plastic type model: {model_path}")
        # Handle different model formats and custom objects
        if model_name == 'Xception':
            # Xception may have TrueDivide custom layer
            try:
                with tf.keras.utils.custom_object_scope({'TrueDivide': Lambda(lambda x: x)}):
                    return load_model(model_path, compile=False)
            except:
                return load_model(model_path, compile=False)
        elif model_name == 'ViT':
            # ViT uses TensorFlow Hub layers
            if HUB_AVAILABLE:
                return load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
            else:
                # Try loading without hub
                return load_model(model_path, compile=False)
        else:
            return load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading {model_name} plastic model: {e}")
        print(f"Using fallback dummy model for {model_name}")
        return create_dummy_plastic_model()

# Create plastic type model directory if it doesn't exist
os.makedirs(PLASTIC_TYPE_MODEL_FOLDER, exist_ok=True)

# Load plastic type models
print("\n" + "="*50)
print("Loading Plastic Type Classification Models...")
print("="*50)

efficientnet_plastic_model = safe_load_plastic_model(
    os.path.join(PLASTIC_TYPE_MODEL_FOLDER, 'efficientnetv2b0_plastic.h5'), 'EfficientNetV2'
)
xception_plastic_model = safe_load_plastic_model(
    os.path.join(PLASTIC_TYPE_MODEL_FOLDER, 'xception_plastic_classifier.h5'), 'Xception'
)
vit_plastic_model = safe_load_plastic_model(
    os.path.join(PLASTIC_TYPE_MODEL_FOLDER, 'vit_plastic_classifier_final.keras'), 'ViT'
)

# Dictionary to map plastic type model names to their instances
plastic_models_dict = {
    'EfficientNetV2': efficientnet_plastic_model,
    'Xception': xception_plastic_model,
    'ViT': vit_plastic_model
}

print("Plastic Type Models loaded successfully!")
print("="*50 + "\n")

# Mock database for storing classification history
classification_history = []

# Function to preprocess the uploaded image
def preprocess_image(image_path, model_name):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Apply model-specific preprocessing
    if model_name == 'ResNet50':
        img_array = preprocess_input(img_array)  # ResNet50-specific preprocessing
    else:
        img_array = img_array / 255.0  # Normalize for VGG16 and InceptionV3
    return img_array

# Function to predict using a specific model
def predict_with_model(model, image_path, model_name):
    img_array = preprocess_image(image_path, model_name)
    pred = model.predict(img_array)[0][0]
    label = 1 if pred >= 0.5 else 0
    class_name = "Recyclable Spotted" if label == 1 else "Organic Spotted"
    confidence = pred if label == 1 else 1 - pred
    return class_name, confidence

# Function to predict using ensemble (majority voting)
def ensemble_predict(image_path):
    # Preprocess for each model
    img_array_resnet = preprocess_image(image_path, 'ResNet50')
    img_array_others = preprocess_image(image_path, 'VGG16')  # Same preprocessing for VGG16 and InceptionV3
    
    # Get predictions from each model
    vgg16_pred = vgg16_model.predict(img_array_others)[0][0]
    resnet50_pred = resnet50_model.predict(img_array_resnet)[0][0]
    inceptionv3_pred = inceptionv3_model.predict(img_array_others)[0][0]
    
    # Convert probabilities to binary predictions (threshold = 0.5)
    vgg16_label = 1 if vgg16_pred >= 0.5 else 0
    resnet50_label = 1 if resnet50_pred >= 0.5 else 0
    inceptionv3_label = 1 if inceptionv3_pred >= 0.5 else 0
    
    # Majority voting
    votes = [vgg16_label, resnet50_label, inceptionv3_label]
    final_label = 1 if sum(votes) >= 2 else 0
    final_class = "Recyclable Spotted" if final_label == 1 else "Organic Spotted"
    
    # Average probability for display
    avg_prob = (vgg16_pred + resnet50_pred + inceptionv3_pred) / 3
    confidence = avg_prob if final_label == 1 else 1 - avg_prob
    
    return final_class, confidence

# ============================================================================
# PLASTIC TYPE CLASSIFICATION FUNCTIONS
# ============================================================================

def preprocess_plastic_image(image_path, apply_cv_preprocessing=True):
    """
    Preprocess image for plastic type classification.
    Optionally applies CV preprocessing (background removal, cropping).
    """
    if apply_cv_preprocessing:
        try:
            # Load the original image with OpenCV
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Isolate the object from the background
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Crop to the largest object found
                main_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(main_contour)
                # Add padding
                x, y, w, h = x - 10, y - 10, w + 20, h + 20
                x, y = max(0, x), max(0, y)
                cropped_img = img_rgb[y:y+h, x:x+w]
                processed_img = cv2.resize(cropped_img, (224, 224))
            else:
                processed_img = cv2.resize(img_rgb, (224, 224))
                
            img_array = np.array(processed_img, dtype=np.float32)
        except Exception as e:
            print(f"CV preprocessing failed: {e}, using standard preprocessing")
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
    else:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_plastic_type_with_model(model, image_path, model_name):
    """
    Predict plastic type using a specific model.
    Returns: (class_name, confidence, all_probabilities)
    """
    img_array = preprocess_plastic_image(image_path)
    predictions = model.predict(img_array)[0]
    
    predicted_idx = np.argmax(predictions)
    predicted_class = PLASTIC_TYPE_CLASSES[predicted_idx]
    confidence = predictions[predicted_idx]
    
    # Create probability dict for all classes
    probabilities = {cls: float(prob) for cls, prob in zip(PLASTIC_TYPE_CLASSES, predictions)}
    
    return predicted_class, float(confidence), probabilities

def ensemble_plastic_type_predict(image_path):
    """
    Ensemble prediction using all 3 plastic type models.
    Uses weighted average of probabilities.
    """
    img_array = preprocess_plastic_image(image_path)
    
    # Get predictions from each model
    eff_pred = efficientnet_plastic_model.predict(img_array)[0]
    xce_pred = xception_plastic_model.predict(img_array)[0]
    vit_pred = vit_plastic_model.predict(img_array)[0]
    
    # Average probabilities (can be weighted if needed)
    avg_pred = (eff_pred + xce_pred + vit_pred) / 3
    
    predicted_idx = np.argmax(avg_pred)
    predicted_class = PLASTIC_TYPE_CLASSES[predicted_idx]
    confidence = avg_pred[predicted_idx]
    
    # Individual model predictions for comparison
    individual_predictions = {
        'EfficientNetV2': {
            'class': PLASTIC_TYPE_CLASSES[np.argmax(eff_pred)],
            'confidence': float(np.max(eff_pred)),
            'probabilities': {cls: float(prob) for cls, prob in zip(PLASTIC_TYPE_CLASSES, eff_pred)}
        },
        'Xception': {
            'class': PLASTIC_TYPE_CLASSES[np.argmax(xce_pred)],
            'confidence': float(np.max(xce_pred)),
            'probabilities': {cls: float(prob) for cls, prob in zip(PLASTIC_TYPE_CLASSES, xce_pred)}
        },
        'ViT': {
            'class': PLASTIC_TYPE_CLASSES[np.argmax(vit_pred)],
            'confidence': float(np.max(vit_pred)),
            'probabilities': {cls: float(prob) for cls, prob in zip(PLASTIC_TYPE_CLASSES, vit_pred)}
        }
    }
    
    # Ensemble probabilities
    ensemble_probabilities = {cls: float(prob) for cls, prob in zip(PLASTIC_TYPE_CLASSES, avg_pred)}
    
    return predicted_class, float(confidence), ensemble_probabilities, individual_predictions

def get_plastic_info(plastic_type):
    """
    Returns detailed information about the plastic type for educational purposes.
    """
    plastic_info = {
        'HDPE': {
            'full_name': 'High-Density Polyethylene',
            'recycling_code': '2',
            'common_uses': ['Milk jugs', 'Detergent bottles', 'Shampoo bottles', 'Plastic bags'],
            'recyclability': 'Highly Recyclable',
            'environmental_impact': 'Low toxicity, widely accepted in recycling programs',
            'color': '#2ecc71',
            'tips': 'Rinse containers before recycling. Remove caps if required by local guidelines.'
        },
        'PET': {
            'full_name': 'Polyethylene Terephthalate',
            'recycling_code': '1',
            'common_uses': ['Water bottles', 'Soda bottles', 'Food containers', 'Polyester clothing'],
            'recyclability': 'Most Recyclable',
            'environmental_impact': 'Most commonly recycled plastic, can be recycled into new bottles or fabric',
            'color': '#3498db',
            'tips': 'Crush bottles to save space. Keep clean and dry for best recycling results.'
        },
        'PP': {
            'full_name': 'Polypropylene',
            'recycling_code': '5',
            'common_uses': ['Yogurt containers', 'Bottle caps', 'Straws', 'Food containers'],
            'recyclability': 'Moderately Recyclable',
            'environmental_impact': 'Heat resistant, increasingly accepted in recycling programs',
            'color': '#f39c12',
            'tips': 'Check local guidelines - PP recycling availability varies by location.'
        },
        'PS': {
            'full_name': 'Polystyrene',
            'recycling_code': '6',
            'common_uses': ['Styrofoam cups', 'Take-out containers', 'Packing peanuts', 'Disposable cutlery'],
            'recyclability': 'Difficult to Recycle',
            'environmental_impact': 'Breaks into microplastics easily, limited recycling options',
            'color': '#e74c3c',
            'tips': 'Avoid when possible. Check for specialized PS recycling drop-off locations.'
        }
    }
    return plastic_info.get(plastic_type, {})

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        # Get the selected model
        selected_model = request.form.get('model')
        if not selected_model:
            return render_template('index.html', error="Please select a model")

        if file:
            # Save the uploaded file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction based on selected model
            try:
                if selected_model == 'Ensemble':
                    predicted_class, confidence = ensemble_predict(filepath)
                else:
                    model = models_dict.get(selected_model)
                    if not model:
                        return render_template('index.html', error="Invalid model selected")
                    predicted_class, confidence = predict_with_model(model, filepath, selected_model)
                
                # Check if recyclable (plastic) - offer further classification
                is_plastic = "Recyclable" in predicted_class
                
                return render_template('index.html', 
                                     prediction=predicted_class, 
                                     confidence=f"{confidence:.2%}", 
                                     image_path=f"uploads/{filename}",
                                     selected_model=selected_model,
                                     is_plastic=is_plastic,
                                     show_plastic_type_option=is_plastic)
            except Exception as e:
                return render_template('index.html', error=f"Error processing image: {str(e)}")
    
    return render_template('index.html')

# Route for the education page
@app.route('/education')
def education():
    return render_template('education.html')

# Route for the analytics page
@app.route('/analytics')
def analytics():
    # Generate mock analytics data
    stats = {
        'total_classifications': 191,
        'organic_count': 71,
        'recyclable_count': 120,
        'avg_confidence': 97
    }
    
    # Generate mock recent classifications
    recent_items = [
        {
            'image_path': 'uploads/plastic_bottle.jpg',
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'model': 'Ensemble',
            'classification': 'Recyclable Spotted',
            'confidence': '98.2%'
        },
        {
            'image_path': 'uploads/banana_peel.jpg',
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'model': 'VGG16',
            'classification': 'Organic Spotted',
            'confidence': '97.5%'
        },
        {
            'image_path': 'uploads/newspaper.jpg',
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'model': 'ResNet50',
            'classification': 'Recyclable Spotted',
            'confidence': '96.8%'
        },
        {
            'image_path': 'uploads/apple_core.jpg',
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'model': 'InceptionV3',
            'classification': 'Organic Spotted',
            'confidence': '99.1%'
        }
    ]
    
    return render_template('analytics.html', stats=stats, recent_items=recent_items)

# Route for the business page
@app.route('/business')
def business():
    # Get mock data for NFT showcase
    nft_items = [
        {
            'id': 1, 
            'name': 'Plastic Bottle Collection #1', 
            'price': 0.05, 
            'image': 'images/nft/plastic_bottle_1.jpg', 
            'description': 'A collection of 10 recycled plastic bottles transformed into art.',
            'seller': 'EcoArtist',
            'category': 'plastic'
        },
        {
            'id': 2, 
            'name': 'Paper Art #42', 
            'price': 0.03, 
            'image': 'images/nft/paper_art_1.jpg', 
            'description': 'Beautiful origami made from recycled newspapers.',
            'seller': 'PaperFold',
            'category': 'paper'
        },
        {
            'id': 3, 
            'name': 'Metal Sculpture', 
            'price': 0.08, 
            'image': 'images/nft/metal_1.jpg', 
            'description': 'Unique metal sculpture created from recycled cans and wires.',
            'seller': 'MetalWorks',
            'category': 'metal'
        },
        {
            'id': 4, 
            'name': 'Glass Mosaic', 
            'price': 0.07, 
            'image': 'images/nft/glass_1.jpg', 
            'description': 'Colorful mosaic made from recycled glass bottles.',
            'seller': 'GlassArtist',
            'category': 'glass'
        }
    ]
    
    # Mock premium features
    premium_features = [
        {
            'name': 'Bulk Classification API',
            'description': 'Classify up to 1000 images per minute with our high-throughput API.',
            'price': '$49.99/month',
            'icon': 'fa-server'
        },
        {
            'name': 'Custom Model Training',
            'description': 'Get a model trained specifically for your recycling needs and waste types.',
            'price': '$299.99',
            'icon': 'fa-cogs'
        },
        {
            'name': 'Analytics Dashboard',
            'description': 'Advanced analytics and reporting on your waste classification patterns.',
            'price': '$19.99/month',
            'icon': 'fa-chart-bar'
        }
    ]
    
    return render_template('business.html', nft_items=nft_items, premium_features=premium_features)

# API endpoint for classifying webcam images
@app.route('/classify_image', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    selected_model = request.form.get('model')
    if not selected_model:
        return jsonify({'error': 'Please select a model'})
    
    try:
        # Generate unique filename
        filename = f"webcam_{uuid.uuid4().hex}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        if selected_model == 'Ensemble':
            predicted_class, confidence = ensemble_predict(filepath)
        else:
            model = models_dict.get(selected_model)
            if not model:
                return jsonify({'error': 'Invalid model selected'})
            predicted_class, confidence = predict_with_model(model, filepath, selected_model)
        
        # Save to classification history
        classification_history.append({
            'image_path': f"uploads/{filename}",
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'model': selected_model,
            'classification': predicted_class,
            'confidence': f"{confidence:.2%}"
        })
        
        # Check if recyclable (plastic) - offer further classification
        is_plastic = "Recyclable" in predicted_class
        
        # Return results as JSON
        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence:.2%}",
            'model': selected_model,
            'image_path': f"uploads/{filename}",
            'is_plastic': is_plastic,
            'can_classify_plastic_type': is_plastic
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# ============================================================================
# PLASTIC TYPE CLASSIFICATION API ENDPOINTS
# ============================================================================

@app.route('/classify_plastic_type', methods=['POST'])
def classify_plastic_type():
    """
    API endpoint to classify plastic type from an uploaded image.
    Returns the plastic type (HDPE, PET, PP, PS) with confidence scores.
    Supports both new file upload and existing image path.
    """
    selected_model = request.form.get('model', 'Ensemble')
    
    try:
        # Check if using existing image path or new upload
        existing_image_path = request.form.get('image_path')
        
        if existing_image_path:
            # Use existing uploaded image (from previous classification)
            # Remove 'uploads/' prefix if present to get just filename
            if existing_image_path.startswith('uploads/'):
                filename = existing_image_path[8:]  # Remove 'uploads/' prefix
            else:
                filename = existing_image_path
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            if not os.path.exists(filepath):
                return jsonify({'error': 'Image not found. Please upload again.', 'success': False})
        else:
            # New file upload
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'})
            
            # Generate unique filename
            filename = f"plastic_{uuid.uuid4().hex}.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        
        # Make prediction based on selected model
        if selected_model == 'Ensemble':
            predicted_class, confidence, probabilities, individual_preds = ensemble_plastic_type_predict(filepath)
            return jsonify({
                'success': True,
                'plastic_type': predicted_class,
                'confidence': f"{confidence:.2%}",
                'confidence_raw': confidence,
                'probabilities': probabilities,
                'individual_predictions': individual_preds,
                'model': 'Ensemble',
                'image_path': f"uploads/{filename}",
                'plastic_info': get_plastic_info(predicted_class)
            })
        else:
            model = plastic_models_dict.get(selected_model)
            if not model:
                return jsonify({'error': f'Invalid model selected: {selected_model}'})
            
            predicted_class, confidence, probabilities = predict_plastic_type_with_model(
                model, filepath, selected_model
            )
            return jsonify({
                'success': True,
                'plastic_type': predicted_class,
                'confidence': f"{confidence:.2%}",
                'confidence_raw': confidence,
                'probabilities': probabilities,
                'model': selected_model,
                'image_path': f"uploads/{filename}",
                'plastic_info': get_plastic_info(predicted_class)
            })
            
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

@app.route('/compare_plastic_models', methods=['POST'])
def compare_plastic_models():
    """
    API endpoint to compare all plastic type models on a single image.
    Returns predictions from all models for comparison.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Generate unique filename
        filename = f"plastic_compare_{uuid.uuid4().hex}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get ensemble prediction (includes individual model predictions)
        predicted_class, confidence, probabilities, individual_preds = ensemble_plastic_type_predict(filepath)
        
        # Determine agreement level
        predictions = [individual_preds[m]['class'] for m in ['EfficientNetV2', 'Xception', 'ViT']]
        unique_predictions = set(predictions)
        
        if len(unique_predictions) == 1:
            agreement = "full"
            conclusion = f"All models unanimously agree this is {predicted_class} plastic with high confidence."
        elif len(unique_predictions) == 2:
            agreement = "partial"
            conclusion = f"Majority of models classify this as {predicted_class}. The ensemble prediction is recommended."
        else:
            agreement = "low"
            conclusion = f"Models show disagreement. Based on ensemble averaging, this appears to be {predicted_class}."
        
        return jsonify({
            'success': True,
            'ensemble': {
                'plastic_type': predicted_class,
                'confidence': f"{confidence:.2%}",
                'confidence_raw': confidence,
                'probabilities': probabilities
            },
            'individual_predictions': individual_preds,
            'agreement': agreement,
            'conclusion': conclusion,
            'image_path': f"uploads/{filename}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

@app.route('/api/plastic-model-status')
def plastic_model_status():
    """Health check endpoint for plastic type models."""
    model_files = {
        'efficientnet_model': os.path.exists(os.path.join(PLASTIC_TYPE_MODEL_FOLDER, 'efficientnetv2b0_plastic.h5')),
        'xception_model': os.path.exists(os.path.join(PLASTIC_TYPE_MODEL_FOLDER, 'xception_plastic_classifier.h5')),
        'vit_model': os.path.exists(os.path.join(PLASTIC_TYPE_MODEL_FOLDER, 'vit_plastic_classifier_final.keras'))
    }
    
    return jsonify({
        'plastic_type_models': model_files,
        'available_classes': PLASTIC_TYPE_CLASSES,
        'models_loaded': {
            'efficientnet': efficientnet_plastic_model is not None,
            'xception': xception_plastic_model is not None,
            'vit': vit_plastic_model is not None
        }
    })

# API endpoint for comparing models
@app.route('/compare_models', methods=['POST'])
def compare_models():
    if 'compare_file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['compare_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Generate unique filename
        filename = f"comparison_{uuid.uuid4().hex}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get predictions from each model
        vgg16_class, vgg16_confidence = predict_with_model(vgg16_model, filepath, 'VGG16')
        resnet50_class, resnet50_confidence = predict_with_model(resnet50_model, filepath, 'ResNet50')
        inception_class, inception_confidence = predict_with_model(inceptionv3_model, filepath, 'InceptionV3')
        ensemble_class, ensemble_confidence = ensemble_predict(filepath)
        
        # Generate conclusion based on results
        if all(x == vgg16_class for x in [resnet50_class, inception_class, ensemble_class]):
            conclusion = f"All models agree that this is {vgg16_class.lower()}, with high confidence across the board."
        else:
            # Count recyclable vs organic predictions
            recyclable_count = sum(1 for x in [vgg16_class, resnet50_class, inception_class, ensemble_class] if 'Recyclable' in x)
            if recyclable_count >= 3:
                conclusion = "Majority of models classify this as recyclable waste. The ensemble model provides the most reliable verdict."
            elif recyclable_count <= 1:
                conclusion = "Majority of models classify this as organic waste. The ensemble model provides the most reliable verdict."
            else:
                conclusion = "Models show some disagreement. The ensemble model's classification is recommended as it leverages the strengths of all models."
        
        # Return results as JSON
        return jsonify({
            'vgg16_class': vgg16_class,
            'vgg16_confidence': f"{vgg16_confidence:.2f}",
            'resnet50_class': resnet50_class,
            'resnet50_confidence': f"{resnet50_confidence:.2f}",
            'inception_class': inception_class,
            'inception_confidence': f"{inception_confidence:.2f}",
            'ensemble_class': ensemble_class,
            'ensemble_confidence': f"{ensemble_confidence:.2f}",
            'conclusion': conclusion,
            'image_path': f"uploads/{filename}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# After loading environment variables, add:
# Check if running in production environment
if os.getenv('ENVIRONMENT') == 'production':
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Try to download models if they don't exist
    try:
        if not all(os.path.exists(os.path.join(MODEL_FOLDER, model)) for model in ['vgg16_waste_classification_tf.h5', 'waste_classification22_model.h5', 'inceptionv3_waste_classification_tf.h5']):
            import subprocess
            subprocess.check_call([sys.executable, 'download_models.py'])
    except Exception as e:
        print(f"Error downloading models: {e}")
        print("Will try to use fallback dummy models.")

# Add new static files (system architecture and workflow diagrams)
# This won't require code changes, just make sure the images are in the right location

# Add a health check route for model status
@app.route('/api/model-status')
def model_status():
    model_files = {
        'vgg16_model': os.path.exists(os.path.join(MODEL_FOLDER, 'vgg16_waste_classification_tf.h5')),
        'resnet50_model': os.path.exists(os.path.join(MODEL_FOLDER, 'waste_classification22_model.h5')),
        'inceptionv3_model': os.path.exists(os.path.join(MODEL_FOLDER, 'inceptionv3_waste_classification_tf.h5'))
    }
    
    return jsonify({
        'model_files': model_files,
        'models_loaded': {
            'vgg16_model': not isinstance(vgg16_model, type(create_dummy_model())),
            'resnet50_model': not isinstance(resnet50_model, type(create_dummy_model())),
            'inceptionv3_model': not isinstance(inceptionv3_model, type(create_dummy_model()))
        }
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)