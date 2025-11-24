# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("diabetes_dataset.csv")

# Feature and target variables
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

# Evaluation
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print("Training Accuracy:", round(train_accuracy, 2))
print("Test Accuracy:", round(test_accuracy, 2))
print("Test F1 Score:", round(test_f1, 2))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
