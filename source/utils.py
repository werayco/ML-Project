import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import dill
import joblib as jb
import pickle as pkl
import matplotlib.pyplot as plt
import logging
import sys
import os
from source.exception import CustomException


def trans_data_pickle(file_path,name_of_object_to_be_saved):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name)
        with open(file_path,"wb") as file_path:
            dill.dump(name_of_object_to_be_saved,file_path)

    except Exception as e:
        raise CustomException(e,sys)
        

