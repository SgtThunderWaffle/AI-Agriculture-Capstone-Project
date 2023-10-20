import sys
sys.path.append('..')

from flask import Flask
from flask import render_template, flash, redirect, url_for, session, request, jsonify
import app
from app.DataPreprocessing import DataPreprocessing
from app.ML_Class import Active_ML_Model, AL_Encoder, ML_Model
from app.SamplingMethods import lowestPercentage
from app.forms import LabelForm
from app.ImagePreprocessing import ImagePreprocessing
from flask_bootstrap import Bootstrap
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import random
import shutil
import numpy as np
import boto3
from io import StringIO

def createMLModel(data):
    
    ml_classifier = RandomForestClassifier()
    preprocess = DataPreprocessing(True)
    
    data_x = data.iloc[:,:-1].values
    data_y = data.iloc[:,-1:].values
    
    data_x = preprocess.fit_transform(data_x)
    ml_model = ml_classifier.fit(data_x, data_y.ravel())
    
    #ml_model = ML_Model(data, RandomForestClassifier(), DataPreprocessing(True))
    return ml_model

def getData(path):
    """
    Gets and returns the csvOut.csv as a DataFrame.

    Returns
    -------
    data : Pandas DataFrame
        The data that contains the features for each image.
    """
    data = pd.read_csv(path, index_col = 0, header = 0)
    data.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35']

    data_mod = data.astype({'8': 'int32','9': 'int32','10': 'int32','12': 'int32','14': 'int32'})
    return data_mod.iloc[:, :]
    
def getRandomPictureData(count):

    images = []
    if len(os.listdir("../TestImages/")) != 0:
        shutil.rmtree("../TestImages/")
        os.mkdir("../TestImages/")
    for i in range(0,count):
        images.append("../images_drone/images_drone/" + str(random.choice(os.listdir("../images_drone/images_drone/"))))
    for image in images:
        shutil.copy(image,"../TestImages/")
    
    ImagePreprocessing("../TestImages/",False)
    
    
    

data = getData("../app/csvOut.csv")
ml_model = createMLModel(data)

getRandomPictureData(10)
testData = getData("./csvOut.csv")
testInput = DataPreprocessing(True).fit_transform(testData.iloc[:,:-1])
print(testData)
print(ml_model.predict(testInput))