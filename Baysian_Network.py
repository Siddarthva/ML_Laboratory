import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load dataset
data = pd.read_csv('heart.csv')

# Select important features
subset_data = data[['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'target']]
print(subset_data.head())

# Define network structure
model = DiscreteBayesianNetwork([
    ('age', 'target'),
    ('sex', 'target'),
    ('cp', 'target'),
    ('thalach', 'target'),
    ('exang', 'target'),
    ('oldpeak', 'target')
])

# Train the model
model.fit(subset_data, estimator=MaximumLikelihoodEstimator)

# Inference
inference = VariableElimination(model)

# Evidence (input values)
evidence = {
    'age': 63,
    'sex': 1,
    'cp': 1,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 2.3
}

# Predict probability of heart disease
result = inference.query(variables=['target'], evidence=evidence)
print(result)