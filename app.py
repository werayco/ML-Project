from flask import Flask,render_template,jsonify,request,url_for
import numpy as np
import pandas as pd
from source.logger import logging
from source.pipeline.predict_pipeline import DataFrameCreator,Predict

# intialzing the flask to this script
app=Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URL"]="sqlite://myname.db"

@app.route("")
def index_html():
    return render_template("index.html")

@app.route("predictor",methods=["POST","GET"])
def predictor():
    if request.method=="GET":
        return render_template("index.html")
    elif request.method=="POST":
        age=request.form.get("age")
        sex = request.form.get("sex")
        name=request.form.get("name")
        height=request.form.get("height")
        data_001=DataFrameCreator(age=age,sex=sex,name=name,height=height)
        pred=data_001.DataFrame()

        


