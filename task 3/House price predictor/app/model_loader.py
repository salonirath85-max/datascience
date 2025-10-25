"""
Load trained model and make predictions
"""

import joblib
import pandas as pd
import numpy as np
import os

# Load model artifacts
MODEL_PATH = 'model/house_price_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
FEATURES_PATH = 'model/feature_names.pkl'

# Global variables to store loaded models
_model = None
_scaler = None
_feature_names = None

def load_model():
    """Load model, scaler, and feature names"""
    global _model, _scaler, _feature_names
    
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please run 'python model/train_model.py' first."
            )
        
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _feature_names = joblib.load(FEATURES_PATH)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Expected features: {len(_feature_names)}")
    
    return _model, _scaler, _feature_names

def engineer_features(input_data):
    """
    Apply same feature engineering as training
    
    Args:
        input_data (dict): Raw input features
    
    Returns:
        pd.DataFrame: Engineered features
    """
    df = pd.DataFrame([input_data])
    
    # Create derived features
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Create age categories
    age = df['age'].values[0]
    if age <= 10:
        df['house_age_category_mid'] = 0
        df['house_age_category_old'] = 0
    elif age <= 25:
        df['house_age_category_mid'] = 1
        df['house_age_category_old'] = 0
    else:
        df['house_age_category_mid'] = 0
        df['house_age_category_old'] = 1
    
    # Drop original age since we have categories
    # (age is still kept as a feature in the model)
    
    return df

def predict_price(input_data):
    """
    Predict house price from input features
    
    Args:
        input_data (dict): Dictionary containing house features
    
    Returns:
        tuple: (predicted_price, confidence_score)
    """
    model, scaler, feature_names = load_model()
    
    # Engineer features
    df = engineer_features(input_data)
    
    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reorder columns to match training
    df = df[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    # Calculate confidence (using prediction variance from trees)
    # Get predictions from all trees
    tree_predictions = np.array([tree.predict(X_scaled)[0] 
                                 for tree in model.estimators_])
    
    # Confidence based on consistency (inverse of coefficient of variation)
    std = np.std(tree_predictions)
    mean_pred = np.mean(tree_predictions)
    
    if mean_pred > 0:
        cv = std / mean_pred  # Coefficient of variation
        confidence = max(0, min(100, 100 * (1 - cv)))  # Convert to 0-100 scale
    else:
        confidence = 50.0
    
    return float(prediction), float(confidence)

def batch_predict(input_list):
    """
    Make predictions for multiple houses
    
    Args:
        input_list (list): List of input dictionaries
    
    Returns:
        list: List of (price, confidence) tuples
    """
    return [predict_price(data) for data in input_list]