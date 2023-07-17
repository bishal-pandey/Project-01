import sys
import os
from  src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

class DataInjectionConfig:
    def __init__(self) :
        self.train_data_path = os.path.join("artificate","train_data.csv")
        self.test_data_path = os.path.join("artificate","test_data.csv")
        self.raw_data_path = os.path.join("artificate","data.csv")

class DataInjection:
    def __init__(self):
        self.injection_config = DataInjectionConfig()
    
    def initiate_data_injection(self):
        logging.info("Data Injection")
        try:
            df = pd.read_csv(r"D:\Programming file\Python\Project-01\Notebook\data\stud.csv")
            logging.info("Data Injection complete")

            os.makedirs(os.path.dirname(self.injection_config.raw_data_path),exist_ok=True)

            df.to_csv(self.injection_config.raw_data_path, index=False, header=True)
            logging.info("Train_test_split")
            train_set, test_set = train_test_split(df, test_size=0.2)
            train_set.to_csv(self.injection_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.injection_config.test_data_path, index=False, header=True)

            logging.info("data Injection completed")

            return (
                self.injection_config.train_data_path,
                self.injection_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataInjection()
    train_data, test_data = obj.initiate_data_injection()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    score = model_trainer.initiate_model_training(train_arr, test_arr)
    print(score)