from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from dataclasses import dataclass
import os
from sklearn.naive_bayes import GaussianNB
from source.utils import best_model,trans_data_pickle
from source.exception import CustomException
import sys
from source.logger import logging
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

@dataclass 
class ModelTrainerConfig:
    path_for_model:str=os.path.join("PickleFiles","BestModel.pkl")

class ModelTrainer:
    def __init__(self):
        self.config=ModelTrainerConfig()

    def ModelTrainerProcess(self,train_arr,test_arr):

        try:
            models = {
                "Linear_Regression": LinearRegression(),
                "Ridge_Regression": Ridge(),
                "Lasso_Regression": Lasso(),
                "K_Neighbors_Regressor": KNeighborsRegressor(),
                "Random_Forest": RandomForestRegressor(),
                "Gradient_Boosting_Regressor": GradientBoostingRegressor(),
                "SVR": SVR(),
                "Decision_Tree": DecisionTreeRegressor(),
                "Naive_Bayes": GaussianNB()
            }
            parameters = {
                "Linear_Regression": {},
                "Ridge_Regression": {'alpha': [0.1, 1.0, 10.0]},
                "Lasso_Regression": {'alpha': [0.1, 1.0, 10.0]},
                "K_Neighbors_Regressor": {'n_neighbors': [3, 5, 7, 9]},
                "Random_Forest": {'n_estimators': [100, 200, 300],
                                'max_depth': [None, 5, 10, 20],
                                'bootstrap': [True, False]},
                "Gradient_Boosting_Regressor": {'n_estimators': [50, 100, 200],
                                                'learning_rate': [0.05, 0.1, 0.2],
                                                'max_depth': [3, 4, 5]},
                "SVR": {'kernel': ['linear', 'rbf'],
                        'C': [0.1, 1, 10],
                        'gamma': ['scale', 'auto']},
                "Decision_Tree": {'max_depth': [None, 5, 10, 20]},
                "Naive_Bayes": {}  # GaussianNB does not have tunable parameters
            }
            x_train,y_train,x_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]

            model_scores:dict=best_model(x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=parameters)

            # best model score from the dict
            best_model_score=max(sorted(list(model_scores.values())))

            best_model_name=list(model_scores.keys())[list(model_scores.values()).index(best_model_score)]
            # the above code will return a string

            best_m=models[best_model_name]
            logging.info(f"your model {best_model_name} is instantiated")

            logging.info(f"the best model is {best_model_name}")
            # now lets instantiate the best model here and fit the model


            
            trans_data_pickle(
                file_path=self.config.path_for_model,
                name_of_object_to_be_saved=best_m
            )
            
            # fitter =best_m.fit(x_train,y_train)
            y_pred=best_m.predict(x_train)
            score=r2_score(y_train,y_pred)
            return score

        except Exception as e:
            raise CustomException(e,sys)
        

# if __name__=="__main__":
#     ModelTrain_obj=ModelTrainer()
