import pytest
from flask import Flask, send_from_directory
from flask import render_template, flash, redirect, url_for, session, request, jsonify
from app import app
from app.DataPreprocessing import DataPreprocessing
from app.ML_Class_New import ML_Model, Active_ML_Model, load_model, tempload_model, generate_token, is_locked
from app.SamplingMethods import lowestPercentage
from app.forms import LabelForm
from joblib import dump, load
from flask_bootstrap import Bootstrap
from sklearn.ensemble import RandomForestClassifier
from app.web import getData
import pandas as pd
import os
import numpy as np
import boto3
from io import StringIO
import json


@pytest.fixture
def client():
    # Runs a dummy web application client
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def X_test():
    # Create a sample DataFrame for X_test with 2 features
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [6, 7, 8, 9, 10],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_with_label():
    # Creates a sample dataframe with labels
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8],
        'label': ['A', 'B', 'A', 'B']
    })
    return data

@pytest.fixture
def ml_model():
    # Makes a dummy ML Model to use for testing using default model
    ml_classifier = RandomForestClassifier() 
    preprocess = DataPreprocessing(True) 
    data = pd.DataFrame({
        'Feature1': [1, 2, 3],
        'Feature2': [4, 5, 6],
        'Label': ['H', 'B', 'H']
    })
    token = generate_token()
    modeldir = 'Models/'
    tempdir = 'tempdata/'

    ml_model = ML_Model(ml_classifier, preprocess, data, token, modeldir, tempdir)

    return ml_model

@pytest.fixture
def ml_modelpractical():
    # Makes a dummy ML Model using existing csv and default model
    ml_classifier = RandomForestClassifier() 
    preprocess = DataPreprocessing(True) 
    data = getData()
    token = generate_token()
    modeldir = 'Models/'
    tempdir = 'tempdata/'

    ml_model = ML_Model(ml_classifier, preprocess, data, token, modeldir, tempdir)

    return ml_model

@pytest.fixture
def active_ml_model():
    #Create a Dummy Active Model to use for testing using default model
    ml_classifier = RandomForestClassifier()
    preprocess = DataPreprocessing(True)
    token = generate_token()
    modeldir = 'Models/'
    tempdir = 'tempdata/'
    data = pd.DataFrame({
        'Feature1': [1, 2, 3, 2, 3, 2, 3, 2, 3, 2],
        'Feature2': [4, 5, 6, 5, 6, 5, 6, 5, 6, 5],
        'Label': ['H', 'B', 'H','B','H', 'B', 'H','B','H','B']
    })

    return Active_ML_Model(ml_classifier, preprocess, data, token, modeldir, tempdir, n_samples=10)


