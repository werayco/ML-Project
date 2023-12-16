import pandas as pd
from dataclasses import dataclass
import os
import sys
import dill
import sys 
from source.logger import logging
from source.exception import CustomException
from sklearn.model_selection import train_test_split,KFold
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
            real_data = pd.read_csv("C:\\Users\\LENOVO-PC\\Videos\\Project001\\Housing.csv")
            logging.info("Data frame read successfully!")

            # Assuming your data has a 'target' column that represents the class for stratification

            logging.info("Using train_test_split to split the data..")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust the number of splits as needed
            for train_index, test_index in kf.split(real_data):
                train_data, test_data = real_data.iloc[train_index], real_data.iloc[test_index]

            logging.info("Data Splitted successfully!")


            path_train_01_MLP=os.path.dirname(self.config.train_path)
            path_test_01_MLP=os.path.dirname(self.config.test_path)

            logging.info("creating the MLP folder for train and test")
            os.makedirs(path_train_01_MLP,exist_ok=True)
            os.makedirs(path_test_01_MLP,exist_ok=True)

            logging.info("Saving the respective train and test dataframe to csv...")
            # now lets save the train data into a csv file
            train_data.to_csv(self.config.train_path,index=False,header=True)
            test_data.to_csv(self.config.test_path,index=False,header=True)
            real_data.to_csv(self.config.real_path,index=False,header=True)
            logging.info("sucessfully saved the train data,test data and raw data as csv's in the MLP directory!")

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


    


