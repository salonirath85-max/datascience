ETL Pipeline - Data Preprocessing & Transformation
A comprehensive Python-based ETL (Extract, Transform, Load) pipeline for automated data preprocessing, transformation, and loading using Pandas and Scikit-learn.

🚀 Features
Multiple Data Sources: Support for CSV, JSON, Excel, and DataFrame inputs
Data Profiling: Automatic data quality assessment and statistics
Missing Value Handling: Multiple strategies (mean, median, mode, constant)
Outlier Detection: IQR and Z-score methods
Feature Engineering: Create custom features with lambda functions
Encoding: Label encoding and one-hot encoding for categorical variables
Scaling: StandardScaler and MinMaxScaler for numerical features
Data Splitting: Train-test split functionality
Multiple Output Formats: Export to CSV, JSON, and Excel
Logging: Comprehensive logging system for pipeline tracking
Metadata Management: Save and track pipeline configurations
📋 Prerequisites
Python 3.8 or higher
pip (Python package manager)
🔧 Installation
Clone the repository:
bash
git clone https://github.com/yourusername/etl-pipeline.git
cd etl-pipeline
Create a virtual environment (recommended):
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install required packages:
bash
pip install -r requirements.txt
📁 Project Structure
etl-pipeline/
│
├── etl_pipeline.py          # Main ETL pipeline script
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── .gitignore             # Git ignore file
│
├── data/
│   ├── raw/               # Raw input data
│   └── processed/         # Processed output data
│
├── notebooks/
│   └── etl_demo.ipynb    # Jupyter notebook with examples
│
└── logs/
    └── etl_pipeline.log  # Pipeline execution logs
💻 Usage
Basic Example
python
from etl_pipeline import ETLPipeline
import pandas as pd

# Initialize pipeline
pipeline = ETLPipeline()

# Load data
pipeline.extract_csv('data/raw/your_data.csv')

# Get data profile
pipeline.get_data_profile()

# Transform data
pipeline.handle_missing_values(strategy='mean')
pipeline.remove_duplicates()
pipeline.encode_categorical(method='label')
pipeline.scale_features(method='standard')

# Save processed data
pipeline.load_to_csv('data/processed/output.csv')
pipeline.save_metadata('data/processed/metadata.json')
Advanced Pipeline
python
# Define complete pipeline steps
pipeline_steps = [
    ('extract_csv', {'filepath': 'data/raw/input.csv'}),
    ('get_data_profile', {}),
    ('handle_missing_values', {'strategy': 'median'}),
    ('remove_duplicates', {}),
    ('handle_outliers', {'method': 'iqr', 'threshold': 1.5}),
    ('encode_categorical', {'columns': ['category1', 'category2'], 'method': 'label'}),
    ('scale_features', {'method': 'standard'}),
    ('load_to_csv', {'filepath': 'data/processed/output.csv'}),
    ('save_metadata', {'filepath': 'data/processed/metadata.json'})
]

# Run pipeline
pipeline = ETLPipeline()
processed_data = pipeline.run_pipeline(pipeline_steps)
📊 Available Methods
Extract Methods
extract_csv() - Load data from CSV
extract_json() - Load data from JSON
extract_excel() - Load data from Excel
extract_from_dataframe() - Load from existing DataFrame
Transform Methods
get_data_profile() - Generate data statistics
handle_missing_values() - Impute missing values
remove_duplicates() - Remove duplicate rows
handle_outliers() - Detect and handle outliers
encode_categorical() - Encode categorical variables
scale_features() - Scale numerical features
create_features() - Create custom features
select_features() - Select specific columns
split_data() - Split into train/test sets
Load Methods
load_to_csv() - Save to CSV
load_to_json() - Save to JSON
load_to_excel() - Save to Excel
save_metadata() - Save pipeline metadata
📝 Example Dataset
The repository includes a sample dataset demonstrating the pipeline capabilities:

python
sample_data = pd.DataFrame({
    'age': [25, 30, None, 45, 50],
    'salary': [50000, 60000, 55000, 80000, None],
    'department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
    'experience': [2, 5, 3, 10, 15]
})
🔍 Logging
The pipeline generates detailed logs for each operation:

2025-10-21 10:30:45 - INFO - Starting ETL Pipeline
2025-10-21 10:30:45 - INFO - Extracting data from CSV: data/raw/sample.csv
2025-10-21 10:30:45 - INFO - Successfully loaded 1000 rows
2025-10-21 10:30:46 - INFO - Handling missing values with strategy: mean
2025-10-21 10:30:46 - INFO - Missing values handled successfully
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👤 Author
Your Name - @yourhandle

Project Link: https://github.com/yourusername/etl-pipeline

🙏 Acknowledgments
Pandas documentation
Scikit-learn documentation
Python logging module
📧 Contact
For questions or feedback, please reach out to your.email@example.com

