"""
Utility functions for input validation and data processing
"""

def validate_input(data):
    """
    Validate input data for prediction
    
    Args:
        data (dict): Input features
    
    Returns:
        tuple: (is_valid, error_message)
    """
    required_features = [
        'square_feet', 'bedrooms', 'bathrooms', 'age',
        'lot_size', 'garage_spaces', 'neighborhood_quality',
        'distance_to_city_center'
    ]
    
    # Check all required features are present
    missing_features = [f for f in required_features if f not in data]
    if missing_features:
        return False, f"Missing required features: {', '.join(missing_features)}"
    
    # Validate data types and ranges
    validations = {
        'square_feet': (int, 800, 5000, "Square feet must be between 800 and 5000"),
        'bedrooms': (int, 1, 5, "Bedrooms must be between 1 and 5"),
        'bathrooms': (int, 1, 4, "Bathrooms must be between 1 and 4"),
        'age': (int, 0, 50, "Age must be between 0 and 50 years"),
        'lot_size': (int, 2000, 20000, "Lot size must be between 2000 and 20000 sq ft"),
        'garage_spaces': (int, 0, 3, "Garage spaces must be between 0 and 3"),
        'neighborhood_quality': (int, 1, 10, "Neighborhood quality must be between 1 and 10"),
        'distance_to_city_center': (float, 0.5, 30, "Distance must be between 0.5 and 30 miles")
    }
    
    for feature, (dtype, min_val, max_val, error_msg) in validations.items():
        value = data[feature]
        
        # Type check
        if not isinstance(value, (int, float)):
            return False, f"{feature} must be a number"
        
        # Convert to expected type
        try:
            if dtype == int:
                value = int(value)
            else:
                value = float(value)
            data[feature] = value
        except (ValueError, TypeError):
            return False, f"{feature} must be a valid {dtype.__name__}"
        
        # Range check
        if not (min_val <= value <= max_val):
            return False, error_msg
    
    return True, None

def format_price(price):
    """Format price with currency symbol and commas"""
    return f"${price:,.2f}"

def calculate_price_per_sqft(price, square_feet):
    """Calculate price per square foot"""
    return price / square_feet if square_feet > 0 else 0

def get_feature_descriptions():
    """Return human-readable descriptions of features"""
    return {
        'square_feet': 'Total living area in square feet',
        'bedrooms': 'Number of bedrooms',
        'bathrooms': 'Number of bathrooms',
        'age': 'Age of the house in years',
        'lot_size': 'Size of the property lot in square feet',
        'garage_spaces': 'Number of garage parking spaces',
        'neighborhood_quality': 'Quality rating of neighborhood (1-10)',
        'distance_to_city_center': 'Distance to city center in miles'
    }