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
from utils import trans_data_pickle

@dataclass 
class SavingPath:
    savingpath:str=os.path.join("Mypath","Saved.pkl")

class DaraTransConfig:
    def __init__(self):
        self.transform=SavingPath()

    def Process(self):
        
        try:
            your_df=pd.read_csv("C:\Users\LENOVO-PC\Videos\car-sales-extended-missing-data.csv")
            numerical_columns=["Doors","Odometer (KM)"]
            categorical_column=["Make","Colour"]
            logging.info("Categorical and numerical data identified!")
            # or we can get your desired by usng
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="mean")), ("standard",StandardScaler(with_mean=False))
            ])
            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most-frequent")),("encoder",OneHotEncoder()),("standard",StandardScaler(with_mean=False))
            ])
            preprocesor=ColumnTransformer(("numerical",num_pipeline,numerical_columns),("categorical"),cat_pipeline,categorical_column)
            return preprocesor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def read_train_test(self,train_path,test_path):
        try:
            preprocessor_obj=self.Process()
            train_csv=pd.read_csv()
            test_csv=pd.read_csv()
            target_feat_test=test_csv["Price"]
            target_feat_train=test_csv["Price"]

            features=train_csv.drop(columns=["Price"],axis=1)
            logging.info("sucessfully loaded the train and test data from ingestion")
            input_trans_train=preprocessor_obj.fit_transform(features)
            input_trans_test=preprocessor_obj.transform(features)

            train_arr=np.c_[input_trans_train, np.array(target_feat_train)]
            test_arr=np.c_[input_trans_test, np.array(target_feat_test)]

            logging.info("saved")

            trans_data_pickle(self.transform.savingpath,preprocessor_obj)
            return (train_arr,test_arr,self.transform.savingpath)




