from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from dataclasses import dataclass
import os
from utils import best_model,trans_data_pickle
from source.exception import CustomException
import sys
from source.logger import logging
from sklearn.metrics import r2_score

@dataclass 
class ModelTrainerConfig:
    path_for_model:str=os.path.join(os.getcwd(),"Mypath","BestModel.pkl")

class ModelTrainer:
    def __init__(self):
        self.config=ModelTrainerConfig()

    def ModelTrainerProcess(self,train_arr,test_arr):
        try:
            models={"LinearRegression":LinearRegression(),"Catboost":CatBoostRegressor(),
            "KNN":KNeighborsRegressor(n_neighbors=3),"DecisonTree":DecisionTreeRegressor()}
            x_train,y_train,x_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]

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
            model_scores:dict=best_model(x_train,y_train,x_test,y_test,model=models,params=parameters)



            # best model score from the dict
            best_model_score=max(sorted(list(model_scores.values())))

            best_model_name=list(model_scores.keys())[list(model_scores.values()).index(best_model_score)]
            # the above code will return a string

            logging.info(f"the best model is {best_model_name}")
            # now lets instantiate the best model here and fit the model

            best_model=models[best_model_name]
            logging.info(f"your model {best_model_name} is instantiated")

            fitter = best_model.fit(x_train,y_train)
            logging.info(f"{best_model_name} is fitted to the data")
            
            y_pred=best_model.predict(x_test)
            score=r2_score(y_true=y_test,y_pred=y_pred)

            if best_model_score<0.6:
                raise CustomException("No best module found")
            
            trans_data_pickle(
                file_path=self.config.path_for_model,
                name_of_object_to_be_saved=best_model
            )
            return best_model_name

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    ModelTrain_obj=ModelTrainer()