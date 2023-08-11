import os,sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self,trainset,testset):
        try:
            logging.info("Data Transformation initiated.")
            train_df = pd.read_csv(trainset)
            test_df = pd.read_csv(testset)

            logging.info("Reading of train and test data completed.")
            logging.info("Obtaining the preprocessor.")
            preprocessor_obj = StandardScaler()
            
            target_column_name = 'GLD'
            numerical_columns = ['SPX','USO','SLV','EUR/USD']

            input_feat_train_df = train_df.drop(columns=[target_column_name,'Date'])
            target_feat_train_df = train_df[target_column_name]

            input_feat_test_df = test_df.drop(columns=[target_column_name,'Date'])
            target_feat_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feat_train_arr = preprocessor_obj.fit_transform(input_feat_train_df)
            input_feat_test_arr = preprocessor_obj.transform(input_feat_test_df)

            train_arr = np.c_[
                input_feat_train_arr,np.array(target_feat_train_df)
            ]
            test_arr = np.c_[
                input_feat_test_arr,np.array(target_feat_test_df)
            ]
            logging.info("Transformation complete saving the preprocessing object.")
            save_object(self.data_transformation_config.preprocessor_path,preprocessor_obj)
            return(train_arr,test_arr)

        except Exception as e:
            raise CustomException(e,sys)