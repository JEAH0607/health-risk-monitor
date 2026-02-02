from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
COLUMNS_PATH = os.path.join(BASE_DIR, 'model_columns.pkl')

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
    
    X = df.drop('Risk', axis=1)
    y = df['Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), COLUMNS_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model, X.columns.tolist()

# Load or train model
if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
else:
    model, model_columns = generate_and_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        data = {
            'Age': float(request.form['age']),
            'Glucose': float(request.form['glucose']),
            'SystolicBP': float(request.form['bp']),
            'BMI': float(request.form['bmi']),
            'ActivityLevel': int(request.form['activity'])
        }
        
        # Create DataFrame for prediction to match training format
        query_df = pd.DataFrame([data])
        
        # Reorder columns to match training
        query_df = query_df[model_columns]
        
        # Predict Risk
        risk_prob = model.predict_proba(query_df)[0][1] # Probability of Class 1 (High Risk)
        risk_class = "High Risk" if risk_prob > 0.65 else "Low Risk"
        
        # Calculate Health Score (0-100)
        # Higher risk = Lower score. 
        # If risk is 0, score is 100. If risk is 1, score is 0.
        health_score = int((1 - risk_prob) * 100)
        
        # LIME Explanation
        # We need a training set summary for LIME. 
        # In a real app, we'd persist the explainer. For this demo, we'll create a simple one.
        # We'll use a small synthetic background for the explainer initialization (fast & simple)
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array([[50, 100, 120, 25, 1], [30, 90, 110, 22, 2], [70, 150, 160, 30, 0]]), # Dummy background
            feature_names=model_columns,
            class_names=['Low Risk', 'High Risk'],
            mode='classification'
        )
        
        exp = explainer.explain_instance(
            data_row=query_df.iloc[0], 
            predict_fn=model.predict_proba,
            num_features=3
        )
        
        # Get top factors
        explanation_list = exp.as_list()
        
        return render_template('index.html', 
                               prediction=risk_class,
                               probability=f"{risk_prob*100:.1f}%",
                               score=health_score,
                               explanation=explanation_list,
                               scroll='results') # Anchor to scroll to
                               
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
