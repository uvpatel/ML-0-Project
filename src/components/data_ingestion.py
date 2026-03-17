# read and data and train test

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")



class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()      
        logging.info("Data Ingestion initiated")

    def initiate_data_ingestion(self, file_path="notebook/data/stud.csv"):
        logging.info("Data Ingestion method started")
        try:
            df = pd.read_csv(file_path)
            logging.info("Data read as pandas dataframe")

            # combine train and test data and save raw data
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            logging.info("Train and Test data combined and saved as raw data")
            

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False)

            
            logging.info("Data Ingestion completed successfully")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error(f"Error in Data Ingestion: {e}")
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    