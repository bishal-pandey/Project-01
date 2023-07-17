import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge,Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer

from  src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, model_evalute
import warnings

class ModelTrainerConfig:
    def __init__(self):
        self.trained_model_file_path = os.path.join("artificate","model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_data, test_data):
        try:
            logging.info("Model Trainig Initiated")
            X_train, y_train, X_test, y_test = (train_data[:,:-1],train_data[:,-1],
                                                test_data[:,:-1], test_data[:,-1])
            
            models = {
                
                # "Lasso":Lasso(),
                # "Ridge":Ridge(),
                # "SVR":SVR(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "LinearRegression":LinearRegression(),
                # "k-Neighbour":KNeighborsRegressor(),
                
                "Xgboost":XGBRegressor(),
            }
            params = {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                
                
                
            }
            
            logging.info("model Evalution begin")
            result = model_evalute(x_train = X_train, y_train = y_train,x_test = X_test, y_test = y_test, models=models,params = params)

            best_models_score = max(sorted(result.values()))
            
            best_model_name = list(models.keys())[list(result.values()).index(best_models_score)]
            
            best_model = models[best_model_name]

            if (best_models_score < 0.6):
                raise CustomException("No best model found", sys)
            
            logging.info("Best model found")
            
            save_obj(file_path=self.model_trainer_config, obj=best_model)
            logging.info("best model is save")


            predicted = best_model.predict(X_test)
            r2_scores = r2_score(y_test, predicted)

            return r2_scores


        except Exception as e:
            raise CustomException(e, sys)