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
            with open("source/component/PickleFiles/BestModel.pkl", "rb") as model_file:
                model_path = plk.load(model_file)
            
            with open("source/component/PickleFiles/Preprocessor.pkl", "rb") as preprocessor_file:
                preprocessor = plk.load(preprocessor_file)
            
            transformed = preprocessor.transform(features)
            y_pred = model_path.predict(transformed)
            return y_pred
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Priceperweek, Population, Monthlyincome, Averageparkingpermonth):  # Adjusted column name here
        self.Priceperweek = Priceperweek
        self.Population = Population
        self.Monthlyincome = Monthlyincome  # Adjusted column name here
        self.Averageparkingpermonth = Averageparkingpermonth

    def DataFrame(self):
        data = {
            "Priceperweek": [self.Priceperweek],
            "Population": [self.Population],
            "Monthlyincome": [self.Monthlyincome],  # Adjusted column name here
            "Averageparkingpermonth": [self.Averageparkingpermonth]
        }
        return pd.DataFrame(data)
