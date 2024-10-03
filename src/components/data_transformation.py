import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from src.utils import save_object, select_top_55_features

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformerConfig()
        self.label_encoder = LabelEncoder()

    def get_data_transformer_object(self):
        try:
            # le = LabelEncoder()
            feature_pipeline = Pipeline(
                steps=[
                    (
                        "replace_infinity",
                        FunctionTransformer(
                            lambda X: X.replace([np.inf, -np.inf], np.nan),
                            validate=False,
                        ),
                    ),
                    (
                        "drop_nulls",
                        FunctionTransformer(lambda X: X.dropna(), validate=False),
                    ),
                ]
            )

            # Combine the transformations using ColumnTransformer
            preprocessing_pipeline = ColumnTransformer(
                transformers=[
                    # Step 1: Apply the feature pipeline to all remaining columns except 'label'
                    (
                        "features",
                        feature_pipeline,
                        slice(0,None),
                    ),  # Apply to all features except 'label'
                ],
                remainder="passthrough",
            )

            return (preprocessing_pipeline)
        

        except Exception as e:
            raise CustomException(e, sys)

    def initialize_data_transformer_obj(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Completed with Reading Datasets")

            unnecessary_cols = [
                "Dst Port",
                "Protocol",
                "Timestamp",
                "Src IP",
                "Src Port",
                "Dst IP",
                "Flow ID",
            ]
            train_df.drop(columns=unnecessary_cols, inplace=True)
            test_df.drop(columns=unnecessary_cols, inplace=True)
            logging.info('Droped the Unnecessary Columns')

            target_column_name = "label"

            preprocessing_pipeline= self.get_data_transformer_object()
            logging.info('Loaded the PreProcess Object')


            label_encoding = LabelEncoder()
            train_df[target_column_name] = label_encoding.fit_transform(train_df[target_column_name]) 
            test_df[target_column_name] = label_encoding.transform(test_df[target_column_name])


            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            train_arr=preprocessing_pipeline.fit_transform(train_df)
            test_arr=preprocessing_pipeline.transform(test_df)

            feature_columns = train_df.columns

            logging.info('Convert the transformed arrays back into DataFrames')
            train_transformed_df = pd.DataFrame(train_arr, columns=feature_columns)
            test_transformed_df = pd.DataFrame(test_arr, columns=feature_columns)

            logging.info('Selecting the top 55 cols')
            imp_cols = select_top_55_features(train_transformed_df)

            logging.info('Updated train and test Datasets with new updated Feature')
            train_transformed_df = train_transformed_df[imp_cols]
            test_transformed_df = test_transformed_df[imp_cols]
            logging.info(f"Traing Dataframe Columns: {train_transformed_df.columns}\nTesting Dataframe Columns: {test_transformed_df.columns}")

            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessing_pipeline, 
            )

            return (
                train_transformed_df,
                test_transformed_df,
                self.data_transformation_config.preprocessor_obj_file,
            )
        except Exception as e:
            raise CustomException(e, sys)
