# Credit Risk Prediction System

This project implements a machine learning-based credit risk prediction system using the German Credit dataset. The system helps financial institutions assess the creditworthiness of loan applicants by predicting whether they are good or bad credit risks.

## Features

- Interactive Streamlit web interface
- Real-time credit risk prediction
- Model performance metrics visualization
- Feature importance analysis
- SHAP values for model interpretation
- Data exploration capabilities

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## Data Source

The German Credit dataset is sourced from the UCI Machine Learning Repository. It contains 1000 instances with 20 features describing various aspects of loan applicants.

## Model Details

The system uses a Random Forest Classifier with the following features:
- 100 decision trees
- StandardScaler for feature normalization
- One-hot encoding for categorical variables
- 80-20 train-test split

## Performance Metrics

The model provides the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

## Contributing

Feel free to submit issues and enhancement requests. 