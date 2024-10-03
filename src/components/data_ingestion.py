import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging  # Assuming logging is set up correctly
from src.exception import CustomException  # Assuming CustomException is defined
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initialize_data_ingestion(self):
        logging.info('Data Ingestion Process has started...')
        try:
            # Reading the dataset
            df = pd.read_csv('dataset/train_data.csv')  # Ensure this path is correct
            logging.info(f'Data has been read successfully. Shape of the dataset: {df.shape}')

            # Creating directory for saving the files
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Saving the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f'Raw data saved at: {self.ingestion_config.raw_data_path}')

            # Splitting the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f'Data split completed. Training set size: {train_set.shape}, Test set size: {test_set.shape}')

            # Saving the train set
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info(f'Training data saved at: {self.ingestion_config.train_data_path}')

            # Saving the test set
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f'Test data saved at: {self.ingestion_config.test_data_path}')

            return self.ingestion_config.test_data_path, self.ingestion_config.train_data_path
        
        except Exception as e:
            # Log the error and raise custom exception
            logging.error(f"Error occurred during data ingestion: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    test_path, train_path = obj.initialize_data_ingestion()
    data_transformation = DataTransformation()
    train_data, test_data,_ = data_transformation.initialize_data_transformer_obj(train_path, test_path)
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_data,test_data))
