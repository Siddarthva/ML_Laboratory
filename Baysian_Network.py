import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("heart.csv")
df = data[['age','sex','cp','thalach','exang','oldpeak','target']].copy()

df['age'] = pd.cut(df['age'], 3, labels=['Young','Middle','Old'])
df['thalach'] = pd.cut(df['thalach'], 3, labels=['LowHR','MedHR','HighHR'])
df['oldpeak'] = pd.cut(df['oldpeak'], 3, labels=['LowST','MedST','HighST'])

model = BayesianNetwork([
    ('age','target'),('sex','target'),('cp','target'),
    ('thalach','target'),('exang','target'),('oldpeak','target')
])

model.fit(df, estimator=MaximumLikelihoodEstimator)
infer = VariableElimination(model)

result = infer.query(
    ['target'], 
    {'age':'Middle','sex':1,'cp':1,'thalach':'MedHR','exang':0,'oldpeak':'MedST'}
)

print(result)
