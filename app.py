from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model_path = "./main/svm_model.pkl"
scaler_path = "./main/scaler.pkl"
model, scaler = None, None

try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    print("Trained model loaded successfully!")

    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading the model or scaler: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
        
            data = {
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
                'thal': float(request.form['thal']),
            }

            features_df = pd.DataFrame([data])

           
            scaled_features = scaler.transform(features_df)

           
            prediction = model.predict(scaled_features)[0]
            prediction_proba = model.predict_proba(scaled_features)[0]

            
            messages = {
                0: "No heart disease detected",
                1: "Stage 1 heart disease detected",
                2: "Stage 2 heart disease detected",
                3: "Stage 3 heart disease detected",
                4: "Stage 4 heart disease detected"
            }
            prediction_message = messages.get(prediction, "Unknown prediction")

           
            response = {
                'prediction': prediction,
                'prediction_probability': f"{max(prediction_proba) * 100:.2f}%",
                'message': prediction_message,
                'features': data
            }

            return render_template('result.html', **response)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 