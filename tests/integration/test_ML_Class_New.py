"""
Created November 2023
@author: Patrick
"""

import pytest
import pandas as pd
import numpy as np
from app import app
from app.web import getData
from app.DataPreprocessing import DataPreprocessing
from app.ML_Class_New import ML_Model, Active_ML_Model, load_model, tempload_model, generate_token, is_locked
from app.SamplingMethods import lowestPercentage
from app.forms import LabelForm
from sklearn.ensemble import RandomForestClassifier
import os




def test_ml_model_infoForProgress():
    # Given an ml_model with practical data
    # Assuming the labeling is either H for healthy or B for blighted 
    # Then itll return the current images listed as H and B
    ml_classifier = RandomForestClassifier() 
    preprocess = DataPreprocessing(True) 
    data = getData()
    token = generate_token()
    modeldir = 'Models/'
    tempdir = 'tempdata/'

    ml_model = ML_Model(ml_classifier, preprocess, data, token, modeldir, tempdir)

    ml_model = ml_model.infoForProgress()

    health_pic, blight_pic = ml_model
    assert len(health_pic) == 2  # Assuming two 'H' based on ml_model fixture
    assert len(blight_pic) == 1  # Assuming one 'B' based on ml_model fixture

def test_infoForResults(active_ml_model):
    # Set up test data (you might need to customize this based on your actual data)
    test_data = pd.DataFrame({
        'Feature1': [1, 2, 3, 2, 3, 2],
        'Feature2': [4, 5, 6, 2, 3, 2],
        'Feature3': [4, 5, 6, 2, 3, 2],
        'Feature4': [4, 5, 6, 2, 3, 2],
        'Feature5': [4, 5, 6, 2, 3, 2],
        'Feature6': [4, 5, 6, 2, 3, 2],
        'Feature7': [4, 5, 6, 2, 3, 2],
        'Feature8': [4, 5, 6, 2, 3, 2],
        'Feature9': [4, 5, 6, 2, 3, 2],
        'Feature10': [4, 5, 6, 2, 3, 2],
        'Feature11': [4, 5, 6, 2, 3, 2],
        'Feature12': [4, 5, 6, 2, 3, 2],
        'Feature13': [4, 5, 6, 2, 3, 2],
        'Feature14': [4, 5, 6, 2, 3, 2],
        'Feature15': [4, 5, 6, 2, 3, 2],
        'Feature16': [4, 5, 6, 2, 3, 2],
        'Feature17': [4, 5, 6, 2, 3, 2],
        'Feature18': [4, 5, 6, 2, 3, 2],
        'Feature19': [4, 5, 6, 2, 3, 2],
        'Feature20': [4, 5, 6, 2, 3, 2],

        'Label': ['H', 'B', 'H','H', 'B', 'H']
    })

    # Assuming ml_model.train_files, ml_model.ml_model, ml_model.GetUnknownPredictions, etc. are set up correctly

    # Call the function being tested
    health_pic_user, blight_pic_user, new_health_pic, new_blight_pic, new_health_pic_prob, new_blight_pic_prob = active_ml_model.infoForResults(test_data)

    # Make assertions based on your expectations
    assert isinstance(health_pic_user, list)
    assert isinstance(blight_pic_user, list)
    assert isinstance(new_health_pic, list)
    assert isinstance(new_blight_pic, list)
    assert isinstance(new_health_pic_prob, list)
    assert isinstance(new_blight_pic_prob, list)

