from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular

app = Flask(__name__)

# Load Model and Columns
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

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
