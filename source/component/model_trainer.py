from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from dataclasses import dataclass
import os
from source.utils import best_model,trans_data_pickle
from source.exception import CustomException
import sys
from source.logger import logging
from sklearn.metrics import r2_score
import numpy as np

@dataclass 
class ModelTrainerConfig:
    path_for_model:str=os.path.join("PickleFiles","BestModel.pkl")

class ModelTrainer:
    def __init__(self):
        self.config=ModelTrainerConfig()

    def ModelTrainerProcess(self,train_arr,test_arr):

        try:
            models={"LinearRegression":LinearRegression(),"Catboost":CatBoostRegressor(),
            "KNN":KNeighborsRegressor(n_neighbors=3),"DecisonTree":DecisionTreeRegressor()}
            x_train,y_train,X_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
            # Check for NaN values in y_train
            nan_rows_train = np.isnan(y_train)

            # Check for NaN values in y_test
            nan_rows_test = np.isnan(y_test)

            y_trainn = np.nan_to_num(y_train, nan=np.nanmean(y_train))
            y_testt = np.nan_to_num(y_test, nan=np.nanmean(y_test))

            parameters={
                "Decision Tree":{
                    "criterion":["squared_error","friedman_mse","absolute_error","poisson"]
                },
                "Random_forest":{"n_estimators":[8,16,32,64,128]
                },
                "Gradient Boosting":{
                    "learning_rate":[.1,.01,.05,.001],
                    "subsample":[0.6,0.7,0.75,0.8,0.85,0.9],
                    "n_estimators":[8,16,32,64,128,256]
                }
            }
            model_scores:dict=best_model(x_train,y_train=y_trainn,x_test=X_test,y_test=y_testt,models=models)



            # best model score from the dict
            best_model_score=max(sorted(list(model_scores.values())))

            best_model_name=list(model_scores.keys())[list(model_scores.values()).index(best_model_score)]
            # the above code will return a string

            best_m=models[best_model_name]
            logging.info(f"your model {best_model_name} is instantiated")

            logging.info(f"the best model is {best_model_name}")
            # now lets instantiate the best model here and fit the model


            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            trans_data_pickle(
                file_path=self.config.path_for_model,
                name_of_object_to_be_saved=best_m
            )
            
            # fitter =best_m.fit(x_train,y_train)
            y_pred=best_m.predict(X_test)
            score=r2_score(y_testt,y_pred)
            return score

        except Exception as e:
            raise CustomException(e,sys)
        

# if __name__=="__main__":
#     ModelTrain_obj=ModelTrainer()
