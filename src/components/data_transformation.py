import sys

from dataclasses import dataclass

import numpy as np

import pandas as pd

import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.components.utils import save_object

from src.exception import  CustomException
from src.logger import logging

class DataTransformationConfig:
    preprocessor_ob_file_path= os.path.join("artifacts", "preprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """This function is responsible for data transformation"""
        try:
            numerical_columns  = ["writing_score", "reading_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            
            num_pipeline = Pipeline(
                steps=[
                    # impute missing values with median and then scale the features
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    # impute missing values with most frequent value and then apply one hot encoding
                    # replace with mode
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    # handle unknown categories by ignoring them during transformation
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    # scale the features without centering to avoid issues with sparse matrices
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error(f"Error in Data Transformation: {e}")
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name="math_score"

            numerical_columns= ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)

            target_feature_train_df = train_df[target_column_name]

            logging.info( 
                f"Applying preprocessing object on training dataframe and testing dataframe."       
                )
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr=preprocessing_obj.transform(test_df[input_feature_train_df.columns])
            target_feature_test_df = test_df[target_column_name]
            
            
            # explain
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                
                self.data_transformation_config.preprocessor_ob_file_path
            )
            

        
        
        except Exception as e:
            raise CustomException(e, sys)