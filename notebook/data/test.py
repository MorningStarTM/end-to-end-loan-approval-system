import pandas as pd
import numpy as np
import dill
from sklearn.metrics import accuracy_score

df = pd.read_csv("artifacts//test.csv")
sample = df.drop(columns=[' loan_status', 'loan_id'], axis=1)
y = df[' loan_status']
#status_mapping = {' Rejected': 0, ' Approved': 1}
#y = y.map(status_mapping)

file_path = "artifacts/preprocessor.pkl"
model_path = "artifacts/model.pkl"

with open(file_path, "rb") as obj:
    preprocess = dill.load(obj)


with open(model_path, "rb") as obj1:
    model = dill.load(obj1)

test = preprocess.transform(sample)
yhat = model.predict(test)
print(accuracy_score(y, yhat))