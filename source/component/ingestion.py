import pandas as pd
from dataclasses import dataclass
import os
import sys
import dill
import sys 
from source.logger import logging
from source.exception import CustomException
from sklearn.model_selection import train_test_split
import numpy as np
from source.component.transformation import DataTransConfig
from source.component.transformation import DataTrans

from source.component.model_trainer import ModelTrainerConfig
from source.component.model_trainer import ModelTrainer




# our mission is to take a csv file then split it into train and split csv respectively

@dataclass
class DataIngestionConfig:
    train_path=os.path.join("Machine Learning Project", "train.csv")
    test_path=os.path.join("Machine Learning Project", "test.csv")
    real_path=os.path.join("Machine Learning Project", "real.csv")

class DataIngestion:
    def __init__(self):
        self.config=DataIngestionConfig()
    
    def DataIngestionProcess(self):
        try:
            real_data = pd.read_csv("C:\\Users\\LENOVO-PC\\Videos\\Project001\\car-sales-extended-missing-data.csv")
            logging.info("just finished reading the raw data")

            # now lets use the train test split to  seperate our data into train and test
            train_data,test_data=train_test_split(real_data,random_state=43,test_size=0.3)
            logging.info("data sucessfully splited")

            path_train_01_MLP=os.path.dirname(self.config.train_path)
            path_test_01_MLP=os.path.dirname(self.config.test_path)

            logging.info("creating the MLP folder for train and test")
            os.makedirs(path_train_01_MLP,exist_ok=True)
            os.makedirs(path_test_01_MLP,exist_ok=True)

            # now lets save the train data into a csv file
            train_data.to_csv(self.config.train_path,index=False,header=True)
            test_data.to_csv(self.config.test_path,index=False,header=True)
            real_data.to_csv(self.config.real_path,index=False,header=True)
            logging.info("sucessfully saved the train data,test data and raw data as csv's directory")

        except Exception as e:
            raise CustomException(e,sys)
        
        return (self.config.train_path,self.config.test_path,self.config.real_path)



if __name__=="__main__":
    DataIngestion_obj=DataIngestion()
    train_path,test_path,_=DataIngestion_obj.DataIngestionProcess()


    trans_obj=DataTrans()
    train_arr,test_arr,_=trans_obj.read_train_test(train_path,test_path)


    model_trainer_obj=ModelTrainer()
    print(model_trainer_obj.ModelTrainerProcess(train_arr,test_arr))


    


