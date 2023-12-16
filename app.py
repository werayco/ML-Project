from flask import Flask,render_template,jsonify,request,url_for
import numpy as np
import pandas as pd
from source.exception import CustomException
from source.logger import logging
import sys
from source.pipeline.predict_pipeline import CustomData,PredictPipeline

# intialzing the flask to this script
application=Flask(__name__)
app = application

logging.info("get request granted")

@app.route('/')
def index():
    return render_template("index.html")


# ... (other imports)

@app.route("/predictdata", methods=["GET", "POST"])
def Predictor():
    if request.method == "POST":
        # Process the form data and perform prediction
        area = request.form.get("area")
        bedrooms = request.form.get("bedrooms")
        bathrooms = request.form.get("bathrooms")
        stories = request.form.get("stories")
        guestroom = request.form.get("guestroom")
        basement = request.form.get("basement")
        mainroad = request.form.get("mainroad")  # Fix variable name typo here
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
    app.run(debug=True,host="0.0.0.0",port=8080)
        


