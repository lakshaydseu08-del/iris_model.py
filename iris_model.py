import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train a K-Nearest Neighbors (KNN) model
# Using n_neighbors=5 as an example hyperparameter
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 4. Evaluate the model's performance
y_pred = model.predict(X_test)

print("--- Model Performance Evaluation ---")
# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\n")

# Classification report (precision, recall, f1-score, support for each class)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 5. Save the trained model as a pickle file
model_pkl_file = "iris_classifier_model.pkl"
with open(model_pkl_file, 'wb') as file:
    pickle.dump(model, file)

print(f"\nModel successfully saved to {model_pkl_file}")

# 6. (Optional) Load the model back to verify
with open(model_pkl_file, 'rb') as file:
    loaded_model = pickle.load(file)

# Verify the loaded model makes the same predictions
loaded_y_pred = loaded_model.predict(X_test)
print(f"\nLoaded model accuracy: {accuracy_score(y_test, loaded_y_pred):.2f}")
