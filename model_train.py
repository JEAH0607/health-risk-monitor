import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Synthetic Data Generation
def generate_data(n_samples=1000):
    np.random.seed(42)
    
    # Features: Age, Glucose, SystolicBP, BMI, ActivityLevel (0-Low, 1-Moderate, 2-High)
    age = np.random.randint(20, 80, n_samples)
    glucose = np.random.normal(100, 20, n_samples) # Mean 100, SD 20
    systolic_bp = np.random.normal(120, 15, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    activity = np.random.randint(0, 3, n_samples)
    
    # Target: Health Risk (0: Low, 1: High)
    # Logic: High risk if combinatorial factors are high
    risk_score = (
        (age / 80) * 0.3 + 
        (glucose / 200) * 0.4 + 
        (systolic_bp / 180) * 0.4 + 
        (bmi / 40) * 0.3 - 
        (activity * 0.1)
    )
    
    # Threshold for risk
    target = (risk_score > 0.65).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Glucose': glucose,
        'SystolicBP': systolic_bp,
        'BMI': bmi,
        'ActivityLevel': activity,
        'Risk': target
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_data()
    print(f"Data generated. Shape: {df.shape}")
    print(f"Class distribution:\n{df['Risk'].value_counts()}")
    
    # 2. Preprocessing
    X = df.drop('Risk', axis=1)
    y = df['Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model Training
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluation
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc:.2f}")
    
    # 5. Save Model using absolute paths
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    columns_path = os.path.join(BASE_DIR, 'model_columns.pkl')
    joblib.dump(model, model_path)
    joblib.dump(X.columns.tolist(), columns_path)
    print(f"Model saved to '{model_path}'")
