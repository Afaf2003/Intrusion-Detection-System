import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, plot_importance  # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from dataclasses import dataclass

from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    model_train_obj_file_path: str = os.path.join("artifacts", "model_trained.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_transformed_df, test_transformed_df):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_transformed_df.iloc[:, :-1],  # Select all columns except the last for features
                train_transformed_df.iloc[:, -1],   # Select the last column for target (label)
                test_transformed_df.iloc[:, :-1],   # Same for test data
                test_transformed_df.iloc[:, -1]     # Select the last column for test target
            )


            # List of Model that can be tried for this Perticular Problem
            # models = {
            #     "Random Forest": RandomForestClassifier(),
            #     "Decision Tree": DecisionTreeClassifier(),
            #     "Gradient Boosting": GradientBoostingClassifier(),
            #     "XGBRegressor": XGBClassifier()
            # }
            XGB_model = XGBClassifier(
                random_state=42,
                use_label_encoder=False,
                eval_metric="mlogloss",
                colsample_bytree=0.8,
                learning_rate=0.1,
                max_depth=7,
                n_estimators=200,
                subsample=1.0,
            )
            logging.info(f"model Started to Trained on both training and testing dataset")
            result = evaluate_models(X_train, y_train, X_test, y_test, XGB_model)
            save_object(
                file_path=self.model_trainer_config.model_train_obj_file_path,
                obj=XGB_model
            )
            logging.info(f"Completetd with Model Training")
            return result
        except Exception as e:
            raise CustomException(e, sys)
