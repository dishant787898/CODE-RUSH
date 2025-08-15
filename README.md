# EcoSortAI - Intelligent Waste Classification

EcoSortAI is a waste classification system that uses ensemble deep learning models to accurately categorize waste items as either organic or recyclable.

## Features

- Multiple pre-trained deep learning models (VGG16, ResNet50, InceptionV3)
- Ensemble approach for improved accuracy (98%+)
- User-friendly web interface for image upload and classification
- Real-time webcam classification
- Educational resources about waste management
- Business solutions including NFT marketplace for recycled materials

## Tech Stack

- **Backend**: Flask (Python)
- **ML Framework**: TensorFlow/Keras
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Chart.js
- **Animations**: AOS, Particles.js

## Installation & Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run model setup: `python setup_models.py`
4. Start the application: `python app.py`

## Model Architecture

The system uses three deep learning models:
- VGG16: 96.5% accuracy
- ResNet50: 96.0% accuracy
- InceptionV3: 97.0% accuracy
- Ensemble (majority voting): 98.0% accuracy

For more details, see the system architecture and workflow documentation in the application.
