import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.exception import CustomException

def save_object(file_path, obj,):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            

    except Exception as e:
        raise CustomException(e, sys)
    

def select_top_55_features(train_transformed_data, target_column='label', top_n=55):
    try:
        # Separate the features and the target
        X = train_transformed_data.drop(columns=[target_column])
        y = train_transformed_data[target_column]
        feature_names = X.columns  # Store feature names before scaling

        # Step 1: Scale the feature data
        sc = StandardScaler()
        scale_X = sc.fit_transform(X)

        # Step 2: Fit LDA model
        lda = LinearDiscriminantAnalysis()
        lda.fit(scale_X, y)

        # Step 3: Compute feature importance scores using LDA coefficients
        lda_coefficients = np.exp(np.mean(np.abs(lda.coef_), axis=0))  # Mean of abs(coef_)
        
        # Step 4: Create a DataFrame to store feature names and their corresponding scores
        df_feature_score = pd.DataFrame({
            'Feature': feature_names,
            'Score': lda_coefficients
        })

        # Step 5: Sort the DataFrame by scores in descending order and select top N features
        imp_feature = df_feature_score.sort_values('Score', ascending=False).head(top_n)

        # Step 6: Return the list of top N important feature names
        top_important_columns = imp_feature['Feature'].tolist()

        return top_important_columns

    except Exception as e:
        print(f"Error in select_top_55_features: {e}")
        return []