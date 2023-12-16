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
            numerical_columns=['area','bedrooms','bathrooms','stories']
            categorical_columns=['guestroom','basement','mainroad','furnishingstatus']
            logging.info("Categorical and numerical data identified!")
            # or we can get your desired by using
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="mean")), ("standard",StandardScaler(with_mean=False))
            ])
            cat_pipeline=Pipeline(steps=[
                 ("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder()),("standard",StandardScaler(with_mean=False))
             ])
            preprocesor=ColumnTransformer([("numerical",num_pipeline,numerical_columns),("cat",cat_pipeline,categorical_columns)])

            return preprocesor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def read_train_test(self,train_path,test_path):
        try:
            train_csv=pd.read_csv(train_path)
            test_csv=pd.read_csv(test_path)

            preprocessor_obj=self.Process()
            target_column = "price"

            input_dataa_train=train_csv[['area','bedrooms','bathrooms','stories','guestroom','basement','mainroad','furnishingstatus']]

            input_dataa_test=test_csv[['area','bedrooms','bathrooms','stories','guestroom','basement','mainroad','furnishingstatus']]
            target_feature_test_df=test_csv[target_column]

            input_feature_train_arr=preprocessor_obj.fit_transform(input_dataa_train)
            input_feature_test_arr=preprocessor_obj.transform(input_dataa_test)

           
            # now lets join the data back into features + target but this time we'd use array
            train_array=np.c_[input_feature_train_arr,np.array(train_csv["price"])]
            test_array=np.c_[input_feature_test_arr,np.array(test_csv["price"])]

            # 
            trans_data_pickle(self.transform.savingpath,preprocessor_obj)
            logging.info("Succesfully saved Preprocessor Pickle file in the 'PickleFiles' Directory")
            
            return (train_array,
                    test_array,
                    self.transform.savingpath)
        
        except Exception as e:
                raise CustomException(e,sys)
    
        
         
