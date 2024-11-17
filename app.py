from flask import Flask, request, render_template, jsonify
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load and train the model
def train_model():
    # File paths
    cleveland_path = "DataSet/preprocessed/preprocessed_cleveland.csv"
    hungarian_path = "DataSet/preprocessed/preprocessed_hungarian.csv"
    switzerland_path = "DataSet/preprocessed/preprocessed_switzerland.csv"
    
    # Load data
    cleveland_data = pd.read_csv(cleveland_path)
    hungarian_data = pd.read_csv(hungarian_path)
    switzerland_data = pd.read_csv(switzerland_path)
    
    # Combine all datasets
    combined_data = pd.concat([cleveland_data, hungarian_data, switzerland_data], ignore_index=True)
    
    # Prepare data
    X = combined_data.drop('target', axis=1)
    y = combined_data['target']
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = SVC(kernel='linear', random_state=42, probability=True)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Загварын нарийвчлал: {accuracy:.2f}')
    
    return model

# Initialize model
try:
    print("Загвар сургаж байна...")
    model = train_model()
    print("Загвар сургалт дууслаа!")
except Exception as e:
    print(f"Загвар сургах явцад алдаа гарлаа: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Get values from the form
            features = {
                'age': float(request.form['age']),
                'sex': float(request.form['sex']),
                'cp': float(request.form['cp']),
                'trestbps': float(request.form['trestbps']),
                'chol': float(request.form['chol']),
                'fbs': float(request.form['fbs']),
                'restecg': float(request.form['restecg']),
                'thalach': float(request.form['thalach']),
                'exang': float(request.form['exang']),
                'oldpeak': float(request.form['oldpeak']),
                'slope': float(request.form['slope']),
                'ca': float(request.form['ca']),
                'thal': float(request.form['thal'])
            }
            
            # Create DataFrame for prediction
            features_df = pd.DataFrame([features])
            
            # Make prediction
            prediction = model.predict(features_df)[0]
            prediction_proba = model.predict_proba(features_df)[0]
            
            # Create response message
            response = {
                'prediction': int(prediction),
                'prediction_probability': f"{max(prediction_proba) * 100:.2f}%",
                'message': get_prediction_message(prediction),
                'features': features
            }
            
            return render_template('result.html', **response)
            
    except Exception as e:
        return render_template('error.html', error=str(e))

def get_prediction_message(prediction):
    messages = {
        0: "Зүрхний өвчин илрээгүй",
        1: "1-р түвшний зүрхний өвчин илэрсэн",
        2: "2-р түвшний зүрхний өвчин илэрсэн",
        3: "3-р түвшний зүрхний өвчин илэрсэн",
        4: "4-р түвшний зүрхний өвчин илэрсэн"
    }
    return messages.get(prediction, "Тодорхойгүй таамаглал")

if __name__ == '__main__':
    app.run(debug=True)