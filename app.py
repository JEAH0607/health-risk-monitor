from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import os
import traceback
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
COLUMNS_PATH = os.path.join(BASE_DIR, 'model_columns.pkl')

# Feature names in correct order
FEATURE_NAMES = ['Age', 'Glucose', 'SystolicBP', 'BMI', 'ActivityLevel']

def generate_and_train_model():
    """Generate synthetic data and train model if not exists"""
    print("Training model...")
    np.random.seed(42)
    n_samples = 1000
    
    age = np.random.randint(20, 80, n_samples)
    glucose = np.random.normal(100, 20, n_samples)
    systolic_bp = np.random.normal(120, 15, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    activity = np.random.randint(0, 3, n_samples)
    
    risk_score = (
        (age / 80) * 0.3 + 
        (glucose / 200) * 0.4 + 
        (systolic_bp / 180) * 0.4 + 
        (bmi / 40) * 0.3 - 
        (activity * 0.1)
    )
    target = (risk_score > 0.65).astype(int)
    
    df = pd.DataFrame({
        'Age': age, 'Glucose': glucose, 'SystolicBP': systolic_bp,
        'BMI': bmi, 'ActivityLevel': activity, 'Risk': target
    })
    
    X = df[FEATURE_NAMES]
    y = df['Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trained_model = RandomForestClassifier(n_estimators=100, random_state=42)
    trained_model.fit(X_train, y_train)
    
    joblib.dump(trained_model, MODEL_PATH)
    joblib.dump(FEATURE_NAMES, COLUMNS_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return trained_model

# Load or train model
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")
    else:
        model = generate_and_train_model()
except Exception as e:
    print(f"Error loading model: {e}")
    model = generate_and_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        age = float(request.form.get('age', 0))
        glucose = float(request.form.get('glucose', 0))
        bp = float(request.form.get('bp', 0))
        bmi = float(request.form.get('bmi', 0))
        activity = int(request.form.get('activity', 0))
        
        # Create feature array in correct order
        features = np.array([[age, glucose, bp, bmi, activity]])
        
        # Predict Risk
        risk_prob = model.predict_proba(features)[0][1]  # Probability of High Risk
        risk_class = "High Risk" if risk_prob > 0.5 else "Low Risk"
        
        # Calculate Health Score (0-100)
        health_score = int((1 - risk_prob) * 100)
        
        # LIME Explanation
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array([
                [50, 100, 120, 25, 1], 
                [30, 90, 110, 22, 2], 
                [70, 150, 160, 30, 0],
                [40, 110, 130, 28, 1],
                [60, 140, 150, 35, 0]
            ]),
            feature_names=FEATURE_NAMES,
            class_names=['Low Risk', 'High Risk'],
            mode='classification'
        )
        
        exp = explainer.explain_instance(
            data_row=features[0], 
            predict_fn=model.predict_proba,
            num_features=3
        )
        
        explanation_list = exp.as_list()
        
        return render_template('index.html', 
                               prediction=risk_class,
                               probability=f"{risk_prob*100:.1f}%",
                               score=health_score,
                               explanation=explanation_list,
                               scroll='results')
                               
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"Prediction error: {error_msg}")
        print(traceback.format_exc())
        return render_template('index.html', error=error_msg)

if __name__ == '__main__':
    app.run(debug=True)

