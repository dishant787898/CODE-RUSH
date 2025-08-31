import os
import numpy as np
import uuid
import datetime
from flask import Flask, request, render_template, jsonify
from PIL import Image
import sys
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback_secret_key')

# Define paths
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Demo prediction function that mimics model behavior
def mock_predict(image_path, model_type):
    """Mock prediction function for demonstration purposes"""
    try:
        # Basic image processing to extract features
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        
        # Generate a consistent but pseudo-random prediction based on image data
        img_array = np.array(img)
        avg_color = np.mean(img_array)
        
        # Add a bias based on the model type to simulate different model behaviors
        model_biases = {
            'VGG16': 0.1,
            'ResNet50': -0.05,
            'InceptionV3': 0.02,
            'Ensemble': 0.0
        }
        bias = model_biases.get(model_type, 0)
        
        # Calculate a probability (0-1) based on image features
        # This is a simplified demo - not real prediction
        prob = (avg_color / 255.0 * 0.5) + 0.25 + bias
        
        # Ensure within bounds
        prob = max(0.1, min(0.9, prob))
        
        # Determine class based on threshold
        class_name = "Recyclable Spotted" if prob > 0.5 else "Organic Spotted"
        confidence = prob if prob > 0.5 else 1 - prob
        
        return class_name, confidence
    except Exception as e:
        print(f"Error in mock prediction: {e}")
        return "Unknown", 0.5

# Routes - simplified from the full app
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
        selected_model = request.form.get('model') or 'Ensemble'
        
        if file:
            # Save the uploaded file
            filename = f"upload_{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction using the mock function
            try:
                predicted_class, confidence = mock_predict(filepath, selected_model)
                
                return render_template('index.html', 
                                     prediction=predicted_class, 
                                     confidence=f"{confidence:.2%}", 
                                     image_path=f"uploads/{filename}",
                                     selected_model=selected_model,
                                     demo_mode=True)
            except Exception as e:
                return render_template('index.html', error=f"Error processing image: {str(e)}")
    
    return render_template('index.html', demo_mode=True)

# Add remaining routes from original app but with mock data
@app.route('/education')
def education():
    return render_template('education.html')

@app.route('/analytics')
def analytics():
    # Same mock data as original app
    stats = {
        'total_classifications': 191,
        'organic_count': 71,
        'recyclable_count': 120,
        'avg_confidence': 97
    }
    
    # Same mock recent classifications as original app
    recent_items = [
        {
            'image_path': 'uploads/plastic_bottle.jpg',
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'model': 'Ensemble',
            'classification': 'Recyclable Spotted',
            'confidence': '98.2%'
        },
        # ...other mock items...
    ]
    
    return render_template('analytics.html', stats=stats, recent_items=recent_items)

@app.route('/business')
def business():
    # Same mock data as original app
    # ...
    return render_template('business.html', nft_items=[], premium_features=[])

@app.route('/api/model-status')
def model_status():
    return jsonify({
        'status': 'lite_mode',
        'message': 'Running in demonstration mode with mock predictions'
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
