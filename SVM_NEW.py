# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
Dataset = pd.read_csv("cars_dataset.csv")
X = Dataset.iloc[:, :-1].values   # Features
y = Dataset.iloc[:, -1].values    # Target variable

# Label Encoding for Categorical Features
from sklearn.preprocessing import LabelEncoder
encoders = []
for i in range(X.shape[1]):
    le = LabelEncoder()
    X[:, i] = le.fit_transform(X[:, i])
    encoders.append(le)

# Encode Target also if string labels
y = LabelEncoder().fit_transform(y)

# Splitting Data into Train & Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling (after split — avoids leakage)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM Classifier Model
from sklearn.svm import SVC
classifier = SVC(C=10.0, kernel="linear")
classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

print("Training Accuracy:", accuracy_score(y_train, classifier.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test F1 Score:", f1_score(y_test, y_pred, average="weighted"))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

