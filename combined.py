from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd


# File paths
cleveland_path = "../DataSet/preprocessed/preprocessed_cleveland.csv"
hungarian_path = "../DataSet/preprocessed/preprocessed_hungarian.csv"
switzerland_path = "../DataSet/preprocessed/preprocessed_switzerland.csv"
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
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
model = SVC(kernel='linear',random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')  