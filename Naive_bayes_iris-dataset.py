# Naive Bayes Classifier using Iris CSV

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset properly (first row is column header)
url = "iris.csv"
df = pd.read_csv(url)

print("Sample Data:")
print(df.head(), "\n")

# Split features and label
X = df.iloc[:, :-1]   # all columns except last
y = df.iloc[:, -1]    # last column as class label

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model training
model = GaussianNB()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
