"""
Flask Web Application for Rainfall Prediction
Provides a REST API and serves the frontend interface.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from data_preprocessing import WeatherDataPreprocessor
from ml_model import RainfallPredictor
from llm_explainer import RainfallExplainer
from dataset_generator import generate_sample_dataset
import numpy as np
import os
import json

app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessor
model = None
preprocessor = None
explainer = None

def initialize_model():
    """Initialize and train the model if not already loaded."""
    global model, preprocessor, explainer
    
    if model is None:
        print("Initializing model...")
        
        # Generate training data
        df = generate_sample_dataset(n_samples=200, random_state=42)
        
        # Preprocess data
        preprocessor = WeatherDataPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
        
        # Train model
        model = RainfallPredictor(random_state=42)
        model.train(X_train, y_train, feature_names=preprocessor.get_feature_names())
        
        # Initialize explainer
        explainer = RainfallExplainer()
        
        print("Model initialized successfully!")

@app.route('/')
def home():
    """Serve the main frontend page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict rainfall based on weather data.
    
    Expected JSON payload:
    {
        "temperature": float,
        "humidity": float,
        "pressure": float,
        "wind_speed": float
    }
    
    Returns:
    {
        "prediction": "Rain Expected" | "No Rain Expected",
        "confidence": float,
        "rain_probability": float,
        "no_rain_probability": float,
        "explanation": {
            "main_insight": str,
            "confidence_level": str,
            "factors": [...]
        },
        "raw_data": {...}
    }
    """
    try:
        # Ensure model is initialized
        initialize_model()
        # Get data from request
        data = request.get_json()
        
        # Validate input
        required_fields = ['temperature', 'humidity', 'pressure', 'wind_speed']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract values
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        pressure = float(data['pressure'])
        wind_speed = float(data['wind_speed'])
        
        # Validate ranges
        if not (0 <= humidity <= 100):
            return jsonify({'error': 'Humidity must be between 0 and 100'}), 400
        if not (-50 <= temperature <= 50):
            return jsonify({'error': 'Temperature must be between -50 and 50°C'}), 400
        if not (900 <= pressure <= 1100):
            return jsonify({'error': 'Pressure must be between 900 and 1100 hPa'}), 400
        if not (0 <= wind_speed <= 100):
            return jsonify({'error': 'Wind speed must be between 0 and 100 km/h'}), 400
        
        # Prepare data for prediction
        weather_array = np.array([[temperature, humidity, pressure, wind_speed]])
        
        # Transform using preprocessor
        weather_scaled = preprocessor.scaler.transform(weather_array)
        
        # Get prediction details
        prediction_details = model.get_prediction_details(weather_scaled)
        
        # Generate explanation
        raw_weather = {
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed
        }
        explanation = explainer.explain_prediction(prediction_details, raw_weather)
        
        # Prepare response
        response = {
            'prediction': explanation['prediction'],
            'confidence': explanation['confidence_percentage'],
            'rain_probability': explanation['probability_rain'],
            'no_rain_probability': explanation['probability_no_rain'],
            'explanation': {
                'main_insight': explanation['main_insight'],
                'confidence_level': explanation['confidence'],
                'factors': explanation['factors']
            },
            'raw_data': raw_weather
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the trained model."""
    try:
        if model is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        info = {
            'feature_names': preprocessor.get_feature_names(),
            'feature_importance': {
                name: float(importance) 
                for name, importance in zip(
                    preprocessor.get_feature_names(), 
                    np.abs(model.model.coef_[0])
                )
            },
            'model_type': 'Logistic Regression',
            'training_samples': 160,
            'test_samples': 40
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200

if __name__ == '__main__':
    # Initialize model before starting server
    initialize_model()
    
    # Run Flask app
    print("\n" + "="*70)
    print("RAINFALL PREDICTION WEB APP")
    print("="*70)
    print("\nServer starting...")
    print("Access the application at: http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  POST /api/predict       - Make rainfall predictions")
    print("  GET  /api/model-info    - Get model information")
    print("  GET  /api/health        - Check server health")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
