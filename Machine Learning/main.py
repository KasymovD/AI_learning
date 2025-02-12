# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels (0: Setosa, 1: Versicolor, 2: Virginica)

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data (optional but recommended for some ML models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Show detailed performance report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Predict a new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example input
new_sample_scaled = scaler.transform(new_sample)
predicted_class = model.predict(new_sample_scaled)
print(f"\nPredicted class for new sample: {iris.target_names[predicted_class[0]]}")
