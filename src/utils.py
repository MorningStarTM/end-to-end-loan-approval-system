import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
import dill
from sklearn.metrics import accuracy_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X,y,X_test,y_test,models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X, y)

            model.set_params(**gs.best_params_)
            model.fit(X, y)

            y_train_pred = model.predict(X)
            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as obj:
            return dill.load(obj)
    except Exception as e:
        raise CustomException(e, sys)
    
    
        