🏠 House Price Predictor
A complete end-to-end machine learning project for predicting house prices using Random Forest Regression, deployed as a Flask web application.

📋 Project Overview
This project demonstrates a full data science pipeline:

Data Collection: Synthetic dataset generation simulating real-world housing data
Data Preprocessing: Cleaning, outlier removal, and feature engineering
Model Training: Random Forest Regressor with hyperparameter tuning
Model Evaluation: Performance metrics and feature importance analysis
API Development: RESTful API using Flask
Web Interface: Interactive UI for predictions
Testing: Comprehensive unit tests
Deployment: Production-ready Flask application
🎯 Features
8 Input Features: Square feet, bedrooms, bathrooms, age, lot size, garage spaces, neighborhood quality, distance to city center
Feature Engineering: Automatic creation of derived features (total rooms, age categories)
High Accuracy: R² score of ~0.95 on test data
Confidence Scoring: Prediction confidence based on model variance
RESTful API: Clean API endpoints for integration
Modern UI: Responsive web interface with real-time predictions
Input Validation: Comprehensive validation with helpful error messages
🚀 Quick Start
Prerequisites
Python 3.8 or higher
pip package manager
Installation
Clone the repository
bash
git clone <repository-url>
cd house-price-predictor
Create virtual environment
bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
Install dependencies
bash
pip install -r requirements.txt
Train the model
bash
python model/train_model.py
Expected output:

Generates synthetic dataset (5,000 samples)
Preprocesses and cleans data
Engineers features
Trains Random Forest model
Saves model artifacts to model/ directory
Run the Flask application
bash
python run.py
The application will be available at: http://localhost:5000

📁 Project Structure
house-price-predictor/
│
├── app/                          # Flask application
│   ├── __init__.py              # App factory
│   ├── routes.py                # API endpoints
│   ├── model_loader.py          # Model loading and prediction
│   └── utils.py                 # Helper functions
│
├── model/                        # ML model artifacts
│   ├── train_model.py           # Training script
│   ├── house_price_model.pkl    # Trained model
│   ├── scaler.pkl               # Feature scaler
│   └── feature_names.pkl        # Feature list
│
├── data/                         # Dataset storage
│   ├── raw/                     # Original data
│   │   └── housing_data.csv
│   └── processed/               # Cleaned data
│       └── housing_data_clean.csv
│
├── templates/                    # HTML templates
│   └── index.html               # Web interface
│
├── tests/                        # Unit tests
│   └── test_routes.py           # Route tests
│
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── run.py                       # Application entry point
🔌 API Endpoints
1. Home Page
GET /
Returns the web interface

2. Predict Price
POST /predict
Content-Type: application/json
Request Body:

json
{
  "square_feet": 2500,
  "bedrooms": 4,
  "bathrooms": 3,
  "age": 5,
  "lot_size": 8000,
  "garage_spaces": 2,
  "neighborhood_quality": 8,
  "distance_to_city_center": 3.5
}
Response:

json
{
  "status": "success",
  "prediction": 625000.50,
  "formatted_prediction": "$625,000.50",
  "confidence_score": 94.5,
  "input_features": {...}
}
3. Health Check
GET /health
Response:

json
{
  "status": "healthy",
  "service": "house-price-predictor",
  "version": "1.0.0"
}
4. API Information
GET /api/info
Returns API documentation and example requests

🧪 Testing
Run unit tests:

bash
python -m pytest tests/
Or using unittest:

bash
python -m unittest tests/test_routes.py
📊 Model Performance
Algorithm: Random Forest Regressor
Training Samples: 4,000
Test Samples: 1,000
R² Score: ~0.95
RMSE: ~$30,000
MAE: ~$20,000
Feature Importance
Top contributing features:

Square Feet (35%)
Neighborhood Quality (25%)
Lot Size (15%)
Bathrooms (10%)
Distance to City Center (8%)
🎨 Web Interface
The application includes a modern, responsive web interface with:

Real-time form validation
Animated prediction display
Confidence score visualization
Error handling with user-friendly messages
Mobile-responsive design
🔧 Configuration
Environment Variables
Create a .env file for custom configuration:

PORT=5000
FLASK_ENV=development
Model Retraining
To retrain the model with different parameters, edit model/train_model.py:

python
model = RandomForestRegressor(
    n_estimators=200,      # Increase for better accuracy
    max_depth=25,          # Adjust tree depth
    random_state=42
)
🚢 Deployment
Production Deployment
Using Gunicorn (recommended):
bash
gunicorn -w 4 -b 0.0.0.0:5000 run:app
Using Docker:
dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "run:app"]
Deploy to Cloud Platforms:
Heroku: Include Procfile
AWS: Use Elastic Beanstalk
Google Cloud: Use App Engine
Azure: Use App Service
📈 Future Enhancements
 Add more advanced models (XGBoost, Neural Networks)
 Implement model versioning
 Add user authentication
 Include real-time data updates
 Add visualization dashboard
 Implement A/B testing
 Add batch prediction capability
 Include model explainability (SHAP values)
🤝 Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch
Make your changes
Add tests
Submit a pull request
📝 License
This project is licensed under the MIT License.

👥 Authors
Data Science Team
🙏 Acknowledgments
Scikit-learn for machine learning tools
Flask for web framework
The open-source community
📞 Support
For issues or questions:

Open an issue on GitHub
Contact: support@houseprice.com
Built with ❤️ using Python, Flask, and Scikit-learn

