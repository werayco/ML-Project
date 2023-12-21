from flask import Flask,render_template,jsonify,request,url_for
import numpy as np
import pandas as pd
import sys

import os
import sys
import pickle as plk
from source.utils import pickle_opener, pickle_loader

    
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

# intialzing the flask to this script
app=Flask(__name__)

logging.info("get request granted")

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def Predictor():
    if request.method == "POST":
        area = request.form.get("area")
        bedrooms = request.form.get("bedrooms")
        bathrooms = request.form.get("bathrooms")
        stories = request.form.get("stories")
        guestroom = request.form.get("guestroom")
        basement = request.form.get("basement")
        mainroad = request.form.get("mainroad") 
        furnishingstatus = request.form.get("furnishingstatus")

        # Create CustomData object
        data_001 = CustomData(area, bedrooms, bathrooms, stories, guestroom, basement, mainroad, furnishingstatus)
        pred_df = data_001.DataFrame()

        # Make predictions
        pred_obj = PredictPipeline()
        results = pred_obj.predictt(pred_df)

        return render_template("home.html", results=results[0])
    else:
        return render_template("index.html")





if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0")
        


