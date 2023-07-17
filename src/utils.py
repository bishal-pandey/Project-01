import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


from src.exception import CustomException

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)
    except:
        pass

def model_evalute(x_train,y_train, x_test, y_test, models):
    try:
        result = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_r2_score = r2_score(y_train, y_train_pred)
            test_r2_score = r2_score(y_test, y_test_pred)

            result[list(models.keys())[i]] = test_r2_score
            return result
        
    except Exception as e:
        raise CustomException





