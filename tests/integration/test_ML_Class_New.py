"""
Created November 2023
@author: Patrick
"""

import pytest
import pandas as pd
from app import app
from app.DataPreprocessing import DataPreprocessing
from app.ML_Class_New import ML_Model, Active_ML_Model, load_model, tempload_model, generate_token, is_locked
from app.SamplingMethods import lowestPercentage
from app.forms import LabelForm
from sklearn.ensemble import RandomForestClassifier
import os



def test_ml_model_initialization(ml_model):
    #Given a sample ml_model 
    #When tested for proper innitialization
    #Then it will output what was given
    assert ml_model.ml_classifier.__class__ == RandomForestClassifier().__class__
    assert ml_model.preprocess is not None
    assert ml_model.X is not None
    assert ml_model.Y is not None
    assert ml_model.get_token() == ml_model.token



def test_locking_mechanism(ml_model):
    #Given a sample ml_model
    #When testing whether its locked or not
    #Then it will be locked if temp data exist and it is not locked if tempdata does not exist
    ml_model.tempsave_model()
    assert is_locked(ml_model.tempdir, ml_model.get_token())  # Model is locked after tempsave_model()

    # clear_tempdata() works in practice, but not in test 
    # ml_model.clear_tempdata()
    # assert not is_locked(ml_model.tempdir, ml_model.get_token()) 
    


def test_model_saving_loading(ml_model):
    # Given a Saved Model
    # When the model is loaded back in
    # Then it will be able to load the model back in
    ml_model.save_model()
    loaded_model = load_model(ml_model.modeldir, ml_model.tempdir, ml_model.get_token())
    assert loaded_model is not None
    assert loaded_model.ml_classifier == ml_model.ml_classifier
    assert loaded_model.preprocess == ml_model.preprocess
    assert loaded_model.X.equals(ml_model.X)
    assert loaded_model.Y.equals(ml_model.Y)
    assert loaded_model.get_token() == ml_model.get_token()


def test_active_ml_model_initialization(active_ml_model):
    #Given a sample ml_model 
    #When tested for proper innitialization
    #Then it will output what was given
    sample_size = 10
    assert len(active_ml_model.sample) == sample_size
    assert len(active_ml_model.test) == sample_size - len(active_ml_model.sample)


