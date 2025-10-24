"""
ETL Pipeline for Data Preprocessing, Transformation, and Loading
Author: Your Name
Date: October 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ETLPipeline:
    """
    Automated ETL Pipeline for data preprocessing and transformation
    """
    
    def __init__(self, config=None):
        """
        Initialize ETL Pipeline with configuration
        
        Args:
            config (dict): Configuration parameters for the pipeline
        """
        self.config = config or {}
        self.data = None
        self.processed_data = None
        self.scaler = None
        self.label_encoders = {}
        self.metadata = {}
        
    # ==================== EXTRACT ====================
    
    def extract_csv(self, filepath, **kwargs):
        """Extract data from CSV file"""
        logger.info(f"Extracting data from CSV: {filepath}")
        try:
            self.data = pd.read_csv(filepath, **kwargs)
            logger.info(f"Successfully loaded {len(self.data)} rows")
            return self.data
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def extract_json(self, filepath, **kwargs):
        """Extract data from JSON file"""
        logger.info(f"Extracting data from JSON: {filepath}")
        try:
            self.data = pd.read_json(filepath, **kwargs)
            logger.info(f"Successfully loaded {len(self.data)} rows")
            return self.data
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise
    
    def extract_excel(self, filepath, sheet_name=0, **kwargs):
        """Extract data from Excel file"""
        logger.info(f"Extracting data from Excel: {filepath}")
        try:
            self.data = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
            logger.info(f"Successfully loaded {len(self.data)} rows")
            return self.data
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            raise
    
    def extract_from_dataframe(self, df):
        """Extract data from existing DataFrame"""
        logger.info("Loading data from DataFrame")
        self.data = df.copy()
        logger.info(f"Successfully loaded {len(self.data)} rows")
        return self.data
    
    # ==================== TRANSFORM ====================
    
    def get_data_profile(self):
        """Generate data profile and statistics"""
        if self.data is None:
            logger.warning("No data loaded")
            return None
        
        profile = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'duplicates': self.data.duplicated().sum(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        logger.info("Data Profile Generated:")
        logger.info(f"  Shape: {profile['shape']}")
        logger.info(f"  Duplicates: {profile['duplicates']}")
        logger.info(f"  Memory Usage: {profile['memory_usage']:.2f} MB")
        
        return profile
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        """
        Handle missing values in the dataset
        
        Args:
            strategy (str): 'mean', 'median', 'most_frequent', 'constant'
            fill_value: Value to use when strategy is 'constant'
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        if self.data is None:
            raise ValueError("No data loaded")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            if strategy == 'constant' and fill_value is not None:
                self.data[numeric_cols] = self.data[numeric_cols].fillna(fill_value)
            else:
                imputer = SimpleImputer(strategy=strategy)
                self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            cat_strategy = 'most_frequent' if strategy != 'constant' else 'constant'
            imputer = SimpleImputer(strategy=cat_strategy, fill_value=fill_value)
            self.data[categorical_cols] = imputer.fit_transform(self.data[categorical_cols])
        
        logger.info("Missing values handled successfully")
        return self.data
    
    def remove_duplicates(self, subset=None, keep='first'):
        """Remove duplicate rows"""
        logger.info("Removing duplicate rows")
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates(subset=subset, keep=keep)
        removed = initial_count - len(self.data)
        logger.info(f"Removed {removed} duplicate rows")
        return self.data
    
    def handle_outliers(self, columns=None, method='iqr', threshold=1.5):
        """
        Handle outliers using IQR or Z-score method
        
        Args:
            columns (list): Columns to check for outliers
            method (str): 'iqr' or 'zscore'
            threshold (float): IQR multiplier or Z-score threshold
        """
        logger.info(f"Handling outliers using {method} method")
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
                self.data[col] = self.data[col].clip(lower_bound, upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers = (z_scores > threshold).sum()
                self.data.loc[z_scores > threshold, col] = self.data[col].median()
            
            logger.info(f"  {col}: {outliers} outliers handled")
        
        return self.data
    
    def encode_categorical(self, columns=None, method='label'):
        """
        Encode categorical variables
        
        Args:
            columns (list): Columns to encode
            method (str): 'label' or 'onehot'
        """
        logger.info(f"Encoding categorical variables using {method} encoding")
        
        if columns is None:
            columns = self.data.select_dtypes(include=['object']).columns
        
        for col in columns:
            if method == 'label':
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"  {col}: Label encoded ({len(le.classes_)} classes)")
                
            elif method == 'onehot':
                dummies = pd.get_dummies(self.data[col], prefix=col, drop_first=True)
                self.data = pd.concat([self.data.drop(col, axis=1), dummies], axis=1)
                logger.info(f"  {col}: One-hot encoded ({dummies.shape[1]} features)")
        
        return self.data
    
    def scale_features(self, columns=None, method='standard'):
        """
        Scale numerical features
        
        Args:
            columns (list): Columns to scale
            method (str): 'standard' or 'minmax'
        """
        logger.info(f"Scaling features using {method} scaling")
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        
        self.data[columns] = self.scaler.fit_transform(self.data[columns])
        logger.info(f"Scaled {len(columns)} features")
        
        return self.data
    
    def create_features(self, operations):
        """
        Create new features based on operations
        
        Args:
            operations (dict): Dictionary of new feature names and their operations
                Example: {'total': lambda df: df['col1'] + df['col2']}
        """
        logger.info("Creating new features")
        
        for feature_name, operation in operations.items():
            self.data[feature_name] = operation(self.data)
            logger.info(f"  Created feature: {feature_name}")
        
        return self.data
    
    def select_features(self, features):
        """Select specific features"""
        logger.info(f"Selecting {len(features)} features")
        self.data = self.data[features]
        return self.data
    
    def split_data(self, target_column, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        
        Args:
            target_column (str): Name of the target variable
            test_size (float): Proportion of test set
            random_state (int): Random seed
        """
        logger.info(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
        
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"  Train set: {len(X_train)} samples")
        logger.info(f"  Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    # ==================== LOAD ====================
    
    def load_to_csv(self, filepath, index=False):
        """Save processed data to CSV"""
        logger.info(f"Loading data to CSV: {filepath}")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(filepath, index=index)
        logger.info("Data saved successfully")
    
    def load_to_json(self, filepath, orient='records'):
        """Save processed data to JSON"""
        logger.info(f"Loading data to JSON: {filepath}")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.data.to_json(filepath, orient=orient, indent=2)
        logger.info("Data saved successfully")
    
    def load_to_excel(self, filepath, sheet_name='Sheet1', index=False):
        """Save processed data to Excel"""
        logger.info(f"Loading data to Excel: {filepath}")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.data.to_excel(filepath, sheet_name=sheet_name, index=index)
        logger.info("Data saved successfully")
    
    def save_metadata(self, filepath='metadata.json'):
        """Save pipeline metadata"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': self.data.shape if self.data is not None else None,
            'columns': list(self.data.columns) if self.data is not None else None,
            'label_encoders': {k: v.classes_.tolist() for k, v in self.label_encoders.items()},
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {filepath}")
    
    # ==================== PIPELINE ====================
    
    def run_pipeline(self, steps):
        """
        Run the complete ETL pipeline
        
        Args:
            steps (list): List of tuples (method_name, kwargs)
        """
        logger.info("=" * 50)
        logger.info("Starting ETL Pipeline")
        logger.info("=" * 50)
        
        for i, (method_name, kwargs) in enumerate(steps, 1):
            logger.info(f"\nStep {i}: {method_name}")
            method = getattr(self, method_name)
            method(**kwargs)
        
        logger.info("\n" + "=" * 50)
        logger.info("ETL Pipeline Completed Successfully")
        logger.info("=" * 50)
        
        return self.data


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ETLPipeline()
    
    # Example: Create sample data for demonstration
    sample_data = pd.DataFrame({
        'age': [25, 30, np.nan, 45, 50, 35, 28, 42, 55, 31],
        'salary': [50000, 60000, 55000, 80000, np.nan, 65000, 52000, 75000, 90000, 58000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'IT', 'Finance', 'HR', 'IT', 'Finance'],
        'experience': [2, 5, 3, 10, 15, 7, 3, 12, 20, 6],
        'performance': ['Good', 'Excellent', 'Good', 'Excellent', 'Average', 'Good', 'Average', 'Excellent', 'Excellent', 'Good']
    })
    
    # Save sample data
    sample_data.to_csv('data/raw/sample_data.csv', index=False)
    
    # Define pipeline steps
    pipeline_steps = [
        ('extract_csv', {'filepath': 'data/raw/sample_data.csv'}),
        ('get_data_profile', {}),
        ('handle_missing_values', {'strategy': 'mean'}),
        ('remove_duplicates', {}),
        ('encode_categorical', {'columns': ['department', 'performance'], 'method': 'label'}),
        ('handle_outliers', {'method': 'iqr', 'threshold': 1.5}),
        ('scale_features', {'method': 'standard'}),
        ('load_to_csv', {'filepath': 'data/processed/processed_data.csv'}),
        ('save_metadata', {'filepath': 'data/processed/metadata.json'})
    ]
    
    # Run pipeline
    processed_data = pipeline.run_pipeline(pipeline_steps)
    
    print("\n" + "=" * 50)
    print("Processed Data Preview:")
    print("=" * 50)
    print(processed_data.head())
    print(f"\nShape: {processed_data.shape}")
    print(f"Columns: {list(processed_data.columns)}")