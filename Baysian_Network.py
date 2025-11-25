import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, K2Score, HillClimbSearch
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings("ignore")  # Hides pandas/pgmpy warnings

# Load dataset
data = pd.read_csv("heart.csv")
features = ['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'target']
subset_data = data[features].copy()

print("\nSample Dataset:")
print(subset_data.head())

# Discretization
subset_data.loc[:, 'age'] = pd.cut(subset_data['age'], bins=3, labels=['Young', 'Middle', 'Old'])
subset_data.loc[:, 'thalach'] = pd.cut(subset_data['thalach'], bins=3, labels=['LowHR', 'MedHR', 'HighHR'])
subset_data.loc[:, 'oldpeak'] = pd.cut(subset_data['oldpeak'], bins=3, labels=['LowST', 'MedST', 'HighST'])

# Split dataset
train_data, test_data = train_test_split(subset_data, test_size=0.30, random_state=42)

# Structure Learning: Hill Climb + K2 Score
hc = HillClimbSearch(train_data)
best_model = hc.estimate(scoring_method=K2Score(train_data))

print("\nLearned Structure Edges:")
print(list(best_model.edges()))

# Train Bayesian Network
model = BayesianNetwork(best_model.edges())
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

# Inference
infer = VariableElimination(model)

# Evaluate Model Accuracy
y_true = test_data['target'].values
preds = []

for _, row in test_data.iterrows():
    evidence = row.drop('target').to_dict()
    pred = infer.query(['target'], evidence=evidence).values.argmax()
    preds.append(pred)

accuracy = accuracy_score(y_true, preds)
print("\nModel Prediction Accuracy:", round(accuracy * 100, 2), "%")

# Sample Prediction
sample_evidence = {
    'age': 'Middle',
    'sex': 1,
    'cp': 1,
    'thalach': 'HighHR',
    'exang': 0,
    'oldpeak': 'LowST'
}

print("\nEvidence for Prediction:")
print(sample_evidence)

result = infer.query(variables=['target'], evidence=sample_evidence)
print("\nHeart Disease Probability:")
print(result)

prediction = result.values.argmax()
print("\nPredicted Heart Disease Class:", prediction, "(1 = Disease, 0 = No Disease)")
