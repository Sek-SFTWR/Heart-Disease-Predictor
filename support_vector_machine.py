# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# File paths
data_paths = {
    'cleveland': "../DataSet/preprocessed/preprocessed_cleveland.csv",
    'hungarian': "../DataSet/preprocessed/preprocessed_hungarian.csv",
    'switzerland': "../DataSet/preprocessed/preprocessed_switzerland.csv"
}

# Load data
data = pd.read_csv(data_paths['hungarian'])
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
model = SVC(kernel='linear', class_weight='balanced', random_state=42, probability=True)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the trained model and scaler
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

with open("svm_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Scaler and model saved successfully!")

