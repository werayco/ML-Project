from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from source.logger import logging
from source.exception import CustomException
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import os
import sys
import numpy as np
from source.utils import trans_data_pickle

@dataclass 
class DataTransConfig:
    savingpath:str=os.path.join("PickleFiles","Preprocessor.pkl")

class DataTrans:
    def __init__(self):
        self.transform=DataTransConfig()

    def Process(self):
        
        try:
            # your_df=pd.read_csv("C:\\Users\\LENOVO-PC\\Videos\\Project001\\car-sales-extended-missing-data.csv")
            numerical_columns=["Doors","Odometer (KM)","Price"]
            categorical_column=["Make","Colour"]
            logging.info("Categorical and numerical data identified!")
            # or we can get your desired by usng
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="mean")), ("standard",StandardScaler(with_mean=False))
            ])
            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder()),("standard",StandardScaler(with_mean=False))
            ])
            preprocesor=ColumnTransformer([("numerical",num_pipeline,numerical_columns),("categorical",cat_pipeline,categorical_column)])

            return preprocesor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def read_train_test(self,train_path,test_path):
        try:
            train_csv=pd.read_csv(train_path)
            test_csv=pd.read_csv(test_path)

            target_col_name="Price"

            # Below are the targets of each data frame i.e Train and test Df's
            target_feat_train=train_csv[target_col_name]
            features_train=train_csv.drop(columns=[target_col_name],axis=1)

            target_feat_test=test_csv[target_col_name]
            features_test=test_csv.drop(columns=[target_col_name],axis=1)

            # Calling the Process method
            preprocessor_obj=self.Process()

            # transformming the respective dataframes
            input_trans_train=preprocessor_obj.fit_transform(train_csv)
            input_trans_test=preprocessor_obj.transform(test_csv)

            # now lets join the data back into features + target but this time we'd use array
            train_array=np.c_[input_trans_train,np.array(target_feat_train)]
            test_array=np.c_[input_trans_test,np.array(target_feat_test)]

            # 
            trans_data_pickle(self.transform.savingpath,preprocessor_obj)
            logging.info("Succesfully saved Preprocessor Pickle file in the 'PickleFiles' Directory")
            
            return (train_array,
                    test_array,
                    self.transform.savingpath)
        
        except Exception as e:
                raise CustomException(e,sys)
    
        
         
