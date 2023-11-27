import os,pandas as pd
import sys
import dill
import sys 
from source.utils import pickle_opener
from source.exception import CustomException



class Predict:
    def __init__(self) -> None:
        pass
    def predict(self,features):
        try:
            model_path=pickle_opener("artficats\model.pkl")
            preprocessor=pickle_opener("artifacts\preprocessor")
            y_pred=model_path.predict(preprocessor.transform(features))
            return y_pred
        except Exception as e:
            raise CustomException(e,sys)
        
class DataFrameCreator:
    def __init__(self,age,sex,name,height):
        self.age=age
        self.sex=sex
        self.height=height
        self.name=name

    def DataFrame(self):
        data={"age":[self.age],
              "sex":[self.sex],
              "name":[self.name],
              "height":[self.height]}
        return pd.DataFrame(data=data)
       
if __name__=="__main__":
    Predict_Obj=Predict()
    Predict_Obj_pre=Predict_Obj.predict()

    Df_obj=DataFrameCreator()
    pd_df=Df_obj.DataFrame()
        