# Random Forest Classifier

# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Importing Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values  # Age, Estimated Salary
Y = dataset.iloc[:, 4].values       # Purchased

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
classifier = RandomForestClassifier(
    n_estimators=200,
    criterion='entropy',
    random_state=0
)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Performance Results
print("Training Accuracy:", accuracy_score(y_train, classifier.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test F1 Score:", f1_score(y_test, y_pred, average="weighted"))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
