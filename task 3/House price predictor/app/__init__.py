"""
Flask application factory
"""

from flask import Flask
from flask_cors import CORS

def create_app(config=None):
    """
    Create and configure Flask application
    
    Args:
        config (dict): Optional configuration dictionary
    
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)
    
    # Default configuration
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    # Apply custom config if provided
    if config:
        app.config.update(config)
    
    # Enable CORS for API endpoints
    CORS(app)
    
    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)
    
    # Log startup
    @app.before_request
    def log_startup():
        app.logger.info("House Price Predictor API started")
    
    return app