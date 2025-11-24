# Load dataset and explore it
import pandas as pd

data = pd.read_json('News_Category_Dataset_v3.json', lines=True)


# Combine text fields
data['text'] = data['headline'] + " " + data['short_description']

# Filter relevant categories
relevant_categories = ['WORLD NEWS', 'ENVIRONMENT', 'PARENTING']
data = data[data['category'].isin(relevant_categories)]

# Encode categories numerically
category_mapping = {
    'WORLD NEWS': 0,
    'ENVIRONMENT': 1,
    'PARENTING': 2
}
data['label'] = data['category'].map(category_mapping)

# Separate input & target
X = data['text']
y = data['label']

# Split train & test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train Naive Bayes
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train_vect, y_train)

# Predictions
y_pred = classifier.predict(X_test_vect)
y_train_pred = classifier.predict(X_train_vect)

# Metrics
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='weighted')

print("Training Accuracy:", round(train_accuracy, 2))
print("Test Accuracy:", round(test_accuracy, 2))
print("Test F1 Score:", round(test_f1, 2))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=category_mapping.keys()))
