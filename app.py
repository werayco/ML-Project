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


@app.route("/predictdata",methods=["GET","POST"])
def Predictor():
        if request.method=="GET":
            return render_template("index.html")
        else:
            data_001=CustomData(Priceperweek=request.form.get("Priceperweek"),Population = request.form.get("Population"),Monthlyincome=request.form.get("Montlyincome"),
            Averageparkingpermonth=request.form.get("Averageparkingpermonth"))
            pred_df=data_001.DataFrame()
            # lets transform and predict
            pred_obj=PredictPipeline()
            print(pred_obj)
            results = pred_obj.predictt(pred_df)
            return render_template("home.html", results=results[0])


if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0")
        


