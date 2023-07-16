import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.utils import save_obj


class DataTransformConfig:
    def __init__(self) -> None:
        self.preprocessor_file_path = os.path.join("artificate", "preprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_transform_config = DataTransformConfig()
    
    def get_data_transformer_obj(self):
        try:
            numerical_features = ["writing_score","reading_score"]
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education',
                                     'lunch', 'test_preparation_course']
            num_pipeline = Pipeline([
                ("inputer", SimpleImputer(strategy='median')),
                ("scalar",StandardScaler())

                ])
            
            cat_pipeline = Pipeline([
                ("Imputer",SimpleImputer(strategy="most_frequent")),
                ("OneHotEncoder",OneHotEncoder(handle_unknown="ignore"))
            ])
            logging.info("Numerical StandardScaler and categorical OneHotEncoder completed")

            transformer = ColumnTransformer([
                ("Num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ])

            return transformer
        

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data  = pd.read_csv(test_path)
            logging.info("Read train and test data")

            transformer_object = self.get_data_transformer_obj()
            target_column = "math_score"

            X_train_data = train_data.drop(target_column, axis=1)
            train_target_data = train_data[target_column]

            X_test_data = test_data.drop(target_column, axis=1)
            test_target_data = test_data[target_column]

            logging.info("applying  transformation on train and test data")
            tranf_X_train_data = transformer_object.fit_transform(X_train_data)
            tranf_X_test_data = transformer_object.fit_transform(X_test_data)

            save_obj(file_path=self.data_transform_config.preprocessor_file_path,obj=transformer_object)

            return (
                tranf_X_train_data,
                tranf_X_test_data,
                self.data_transform_config.preprocessor_file_path

            )

        except Exception as e:
            raise CustomException(e, sys)
