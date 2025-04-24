import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("Credit Risk Prediction System")
st.markdown("""
This application predicts the credit risk of loan applicants using machine learning.
The model is trained on the German Credit dataset and provides insights into key factors affecting credit risk.
""")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

# Function to load and preprocess data
@st.cache_data
def load_data():
    # Load the German Credit dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    column_names = [
        'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings_account', 'employment', 'installment_rate', 'personal_status',
        'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
        'housing', 'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
        'credit_risk'
    ]
    data = pd.read_csv(url, sep=' ', names=column_names)
    return data

# Load data
data = load_data()

# Display raw data
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(data)

# Data preprocessing
def preprocess_data(df):
    # Convert categorical variables to numerical
    categorical_columns = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Split features and target
    X = df_encoded.drop('credit_risk', axis=1)
    y = df_encoded['credit_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

# Train model
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Main function
def main():
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    
    # Display metrics
    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    with col2:
        st.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
    with col3:
        st.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    with col4:
        st.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance.head(10), x='Importance', y='Feature', 
                 title='Top 10 Most Important Features')
    st.plotly_chart(fig)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(fig)

if __name__ == "__main__":
    main() 