import os
import pandas as pd
import sys
import pickle as plk
from source.utils import pickle_opener, pickle_loader
from source.exception import CustomException

    
class PredictPipeline:
    def __init__(self):
        pass

    def predictt(self, features):
        try:
            model_path1 = "source/component/PickleFiles/BestModel.pkl"
            processor_path = "source/component/PickleFiles/Preprocessor.pkl"
            model=pickle_loader(model_path=model_path1)
            preprocessor = pickle_loader(model_path=processor_path)
            transformed = preprocessor.transform(features)
            y_pred = model.predict(transformed)
            return y_pred
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,furnishingstatus):
        self.area = area
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms  
        self.stories = stories
        self.mainroad=mainroad
        self.guestroom=guestroom
        self.basement=basement
        self.furnishingstatus=furnishingstatus

        # numerical_columns=['area','bedrooms','bathrooms','stories']
        # categorical_columns=['guestroom','basement','mainroad','furnishingstatus']

    def DataFrame(self):
        data = {
            "area": [self.area],
            "bedrooms": [self.bedrooms],
            "bathrooms": [self.bathrooms],
            "stories": [self.stories],
            "mainroad":[self.mainroad],
            "guestroom":[self.mainroad],
            "basement":[self.basement],
            "furnishingstatus":[self.furnishingstatus]
        }
        return pd.DataFrame(data)
