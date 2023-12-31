import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Gradient Boosting" : GradientBoostingClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier()
            }

            params = {
                "Logistic Regression":{
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs']
                },

                "K-Neighbors Classifier":{
                    'n_neighbors': [3, 5, 7],  # Adjust the number of neighbors
                    'weights': ['uniform', 'distance'],
                },

                'Decision Tree Classifier': {
                    'criterion': ['gini', 'entropy'],  # Splitting criterion
                    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
                },

                'Gradient Boosting': {
                    'n_estimators': [50, 100, 200],  # Number of boosting stages
                    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
                    'max_depth': [3, 4, 5],  # Maximum depth of individual trees
                },

                'CatBoosting Classifier': {
                    'iterations': [100, 200, 300],  # Number of boosting iterations
                    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
                    'depth': [4, 6, 8],  # Depth of the trees
                },

                'AdaBoost Classifier': {
                    'n_estimators': [50, 100, 200],  # Number of boosting stages
                    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
                },

                "Random Forest Classifier":{
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'n_estimators': [50, 100, 200]
                }


            }

            model_report:dict = evaluate_model(X=X_train, y=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Found best model on both training and testing")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            accuracy_scores = accuracy_score(y_test, predicted)
            return accuracy_scores

        except Exception as e:
            raise CustomException(e, sys)