"""
House Price Prediction Model Training Script
Collects data, preprocesses, trains model, and saves artifacts
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

# Create directories
os.makedirs('model', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("=" * 60)
print("HOUSE PRICE PREDICTION MODEL TRAINING")
print("=" * 60)

# Step 1: Generate synthetic dataset (simulating real-world data collection)
print("\n[1/6] Generating synthetic housing dataset...")
np.random.seed(42)

n_samples = 5000
data = {
    'square_feet': np.random.randint(800, 5000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 5, n_samples),
    'age': np.random.randint(0, 50, n_samples),
    'lot_size': np.random.randint(2000, 20000, n_samples),
    'garage_spaces': np.random.randint(0, 4, n_samples),
    'neighborhood_quality': np.random.randint(1, 11, n_samples),
    'distance_to_city_center': np.random.uniform(0.5, 30, n_samples),
}

df = pd.DataFrame(data)

# Generate realistic price based on features
base_price = 50000
df['price'] = (
    base_price +
    df['square_feet'] * 150 +
    df['bedrooms'] * 15000 +
    df['bathrooms'] * 20000 +
    df['lot_size'] * 5 +
    df['garage_spaces'] * 12000 +
    df['neighborhood_quality'] * 25000 -
    df['age'] * 2000 -
    df['distance_to_city_center'] * 1500 +
    np.random.normal(0, 30000, n_samples)
)

# Add some outliers and missing values to simulate real data
df.loc[np.random.choice(df.index, 50), 'price'] *= 1.5
df.loc[np.random.choice(df.index, 20), 'age'] = np.nan

# Save raw data
df.to_csv('data/raw/housing_data.csv', index=False)
print(f"   ✓ Generated {len(df)} samples")
print(f"   ✓ Features: {list(df.columns[:-1])}")

# Step 2: Data Preprocessing
print("\n[2/6] Preprocessing data...")

# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)

# Remove outliers (IQR method)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

print(f"   ✓ Handled missing values")
print(f"   ✓ Removed {len(df) - len(df_clean)} outliers")
print(f"   ✓ Final dataset: {len(df_clean)} samples")

# Save processed data
df_clean.to_csv('data/processed/housing_data_clean.csv', index=False)

# Step 3: Feature Engineering
print("\n[3/6] Engineering features...")

# Create derived features
df_clean['price_per_sqft'] = df_clean['price'] / df_clean['square_feet']
df_clean['total_rooms'] = df_clean['bedrooms'] + df_clean['bathrooms']
df_clean['house_age_category'] = pd.cut(df_clean['age'], 
                                         bins=[0, 10, 25, 50], 
                                         labels=['new', 'mid', 'old'])

# One-hot encode categorical features
df_encoded = pd.get_dummies(df_clean, columns=['house_age_category'], drop_first=True)

print(f"   ✓ Created derived features")
print(f"   ✓ Total features: {len(df_encoded.columns) - 1}")

# Step 4: Prepare train/test split
print("\n[4/6] Splitting dataset...")

X = df_encoded.drop(['price', 'price_per_sqft'], axis=1)
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   ✓ Training set: {len(X_train)} samples")
print(f"   ✓ Test set: {len(X_test)} samples")

# Step 5: Train model with preprocessing pipeline
print("\n[5/6] Training Random Forest model...")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

print(f"   ✓ Model trained successfully")

# Step 6: Evaluate model
print("\n[6/6] Evaluating model performance...")

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\n   Training Metrics:")
print(f"   ├─ R² Score: {train_r2:.4f}")
print(f"   └─ RMSE: ${train_rmse:,.2f}")

print(f"\n   Test Metrics:")
print(f"   ├─ R² Score: {test_r2:.4f}")
print(f"   ├─ RMSE: ${test_rmse:,.2f}")
print(f"   └─ MAE: ${test_mae:,.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 5 Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   ├─ {row['feature']}: {row['importance']:.4f}")

# Save model and preprocessing objects
print("\n[SAVING] Serializing model artifacts...")

joblib.dump(model, 'model/house_price_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(X.columns.tolist(), 'model/feature_names.pkl')

print("   ✓ Saved: model/house_price_model.pkl")
print("   ✓ Saved: model/scaler.pkl")
print("   ✓ Saved: model/feature_names.pkl")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nNext step: Run 'python run.py' to start the Flask application")