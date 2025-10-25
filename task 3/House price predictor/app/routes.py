"""
Flask routes for house price prediction API
"""

from flask import Blueprint, render_template, request, jsonify
from app.model_loader import predict_price
from app.utils import validate_input
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
main = Blueprint('main', __name__)

@main.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    """
    Predict house price based on input features
    
    Expected JSON payload:
    {
        "square_feet": 2000,
        "bedrooms": 3,
        "bathrooms": 2,
        "age": 10,
        "lot_size": 5000,
        "garage_spaces": 2,
        "neighborhood_quality": 7,
        "distance_to_city_center": 5.5
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No input data provided',
                'status': 'error'
            }), 400
        
        # Validate input
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({
                'error': error_msg,
                'status': 'error'
            }), 400
        
        # Make prediction
        prediction, confidence = predict_price(data)
        
        logger.info(f"Prediction made: ${prediction:,.2f}")
        
        # Return prediction
        return jsonify({
            'status': 'success',
            'prediction': round(prediction, 2),
            'formatted_prediction': f"${prediction:,.2f}",
            'confidence_score': round(confidence, 2),
            'input_features': data
        }), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error during prediction',
            'details': str(e),
            'status': 'error'
        }), 500

@main.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'house-price-predictor',
        'version': '1.0.0'
    }), 200

@main.route('/api/info', methods=['GET'])
def api_info():
    """Return API information and expected input format"""
    return jsonify({
        'service': 'House Price Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'Web interface',
            '/predict': 'POST - Make price prediction',
            '/health': 'GET - Health check',
            '/api/info': 'GET - API information'
        },
        'input_format': {
            'square_feet': 'int (800-5000)',
            'bedrooms': 'int (1-5)',
            'bathrooms': 'int (1-4)',
            'age': 'int (0-50)',
            'lot_size': 'int (2000-20000)',
            'garage_spaces': 'int (0-3)',
            'neighborhood_quality': 'int (1-10)',
            'distance_to_city_center': 'float (0.5-30)'
        },
        'example_request': {
            'square_feet': 2500,
            'bedrooms': 4,
            'bathrooms': 3,
            'age': 5,
            'lot_size': 8000,
            'garage_spaces': 2,
            'neighborhood_quality': 8,
            'distance_to_city_center': 3.5
        }
    }), 200