import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import dill
import joblib as jb
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import logging
import sys
import os
import pickle as plkk
from source.exception import CustomException
from sklearn.metrics import r2_score
from typing import List
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,cross_val_score,KFold

def trans_data_pickle(file_path,name_of_object_to_be_saved):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,"wb") as file_path:
            dill.dump(name_of_object_to_be_saved,file_path)

    except Exception as e:
        raise CustomException(e,sys)
        

def best_model(x_train,y_train,x_test,y_test,models):
    try:
        model_plus_scores={}
        for model_name,model in models.items():
            # para=params[list(models.keys())]
            # gs = GridSearchCV(model,para,cv=3)

            # gs.fit(x_train,y_train)
            # model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            y_pred_01=model.predict(x_test)
            r2scores=r2_score(y_true=y_test,y_pred=y_pred_01)

            model_plus_scores[model_name]=r2scores
            return model_plus_scores


    except Exception as e:
        raise CustomException(e,sys)
    
def pickle_opener(model_path):
    try:
        modell=plkk.load(model_path)
        return modell
    except Exception as e:
        raise CustomException(e,sys)
    
def pickle_loader(model_path):
    try:
        with open(model_path,"rb") as file_obj:
            dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
        



