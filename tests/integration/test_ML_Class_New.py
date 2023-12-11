"""
Created November 2023
@author: Patrick
"""

import pytest
import pandas as pd
import numpy as np
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



def test_ml_model_locking_mechanism(ml_model):
    #Given a sample ml_model
    #When testing whether its locked or not
    #Then it will be locked if temp data exist and it is not locked if tempdata does not exist

    ml_model.tempsave_model()
    assert is_locked(ml_model.tempdir, ml_model.get_token())  # Model is locked after tempsave_model()

    # the .dot and .png file and .joblib are attempted to be cleared by clear_tempdata 
    # but if one of them are not found then it would fail the clearing
    dot_file_path = os.path.join(ml_model.tempdir, ml_model.get_token() + '.dot')
    with open(dot_file_path, 'w') as f:
        f.write('Dummy DOT file content')
    png_file_path = os.path.join(ml_model.tempdir, ml_model.get_token() + '.png')
    with open(png_file_path, 'w') as f:
        f.write('Dummy PNG file content')

    ml_model.clear_tempdata()
    assert not is_locked(ml_model.tempdir, ml_model.get_token()) 
    
    
def test_ml_model_saving_loading(ml_model):
    # Given a Saved Model
    # When the model is loaded back in
    # Then it will be able to load the model back in

    ml_model.save_model()
    
    # the .dot and .png file are tied to visualize_model which the load_model will look for
    # for this test we will create a dummy .dot file and dummy .png file
    dot_file_path = os.path.join(ml_model.modeldir + ml_model.get_token(), ml_model.get_token() + '.dot')
    with open(dot_file_path, 'w') as f:
        f.write('Dummy DOT file content')
    png_file_path = os.path.join(ml_model.modeldir + ml_model.get_token(), ml_model.get_token() + '.png')
    with open(png_file_path, 'w') as f:
        f.write('Dummy PNG file content')

    
    loaded_model = load_model(ml_model.modeldir, ml_model.tempdir, ml_model.get_token())
    assert loaded_model is not None
    assert loaded_model.ml_classifier.get_params() == ml_model.ml_classifier.get_params()
    assert np.array_equal(loaded_model.X, ml_model.X)
    assert np.array_equal(loaded_model.X, ml_model.X)
    assert loaded_model.get_token() == ml_model.get_token()

def test_ml_model_temp_saving_loading(ml_model):
    # Given a  temporarily Saved Model
    # When the model is temporarily loaded back in
    # Then it will be able to load the model back in
    ml_model.tempsave_model()
    temploaded_model = tempload_model(ml_model.tempdir, ml_model.get_token())
    assert temploaded_model is not None
    assert temploaded_model.ml_classifier.get_params() == ml_model.ml_classifier.get_params()
    assert np.array_equal(temploaded_model.X, ml_model.X)
    assert np.array_equal(temploaded_model.X, ml_model.X)
    assert temploaded_model.get_token() == ml_model.get_token()

def test_ml_model_get_known_predictions(sample_with_label_healthblighted, ml_model):
    # Given a sample ml model and a sample with label 
    # When we call getknownprediction 
    #then it will return a prediction and a probability that is a float.
    data = sample_with_label_healthblighted
    ml_model = ml_model

    y_prediction, max_probability = ml_model.GetKnownPredictions(sample_with_label_healthblighted)

    assert len(y_prediction) == len(data)
    assert isinstance(max_probability, float) 

def test_ml_model_get_unknown_predictions(X_test, ml_model):
    # Given a sample ml model and a sample with label 
    # When we call getunknownprediction 
    #then it will return a prediction and the list of probability based on the data given

    data = X_test
    ml_model = ml_model

    y_prediction, max_probability = ml_model.GetUnknownPredictions(data)

    assert len(y_prediction) == len(data)
    assert isinstance(max_probability, list) 
 

def test_ml_model_Visualize_model(ml_modelpractical):
    #Given an ml_model with practical data
    #when we save the model and run visualize_model
    #then there will be a .dot and .png file at the expected directory

    #practical model is called due to visualize_model 
    #strictly only works with csv created by imageprepocessing

    ml_modelpractical.visualize_model(2)
    dot_file_path = os.path.join(ml_modelpractical.tempdir + ml_modelpractical.get_token() +'.dot')
    png_file_path = os.path.join(ml_modelpractical.tempdir + ml_modelpractical.get_token() + '.png')

    assert os.path.isfile(dot_file_path),("dot file does not exist")
    assert os.path.isfile(png_file_path),("png file does not exist")

def test_ml_model_Kfold(ml_modelpractical):
    #Given an ml_model and an active_ml_model
    #when tested for Kfold 
    #Then itll return the lenght of accuracy as 3 with each value ranging from 0 - 100%

    #practical model is called due to kfold requiring a split of 3, 
    #but ml_model only has 2 unique classes
    ml_model_accuracies = ml_modelpractical.K_fold()

    assert len(ml_model_accuracies) == 3

    valid_accuracy_range = (0.0, 1.0)
    assert all(valid_accuracy_range[0] <= acc <= valid_accuracy_range[1] for acc in ml_model_accuracies)

def test_active_ml_model_initialization(active_ml_model):
    #Given a sample ml_model 
    #When tested for proper innitialization
    #Then it will output what was given
    sample_size = 10
    assert len(active_ml_model.sample) == sample_size
    assert len(active_ml_model.test) == sample_size - len(active_ml_model.sample)

def test_active_ml_model_train_model_add(active_ml_model):
    #Given a sample active ml_model
    #When train_model_add is called with a sample to add
    #Then the current active ml model length becomes the original plus the length of the sample added
    sample_to_add = pd.DataFrame({
        'Feature1': [4, 5],
        'Feature2': [7, 8],
        'Label': ['B', 'H']
    })
    original_train_size = len(active_ml_model.train) if active_ml_model.train is not None else 0

    active_ml_model.train_model_add(sample_to_add)

    assert len(active_ml_model.train) == original_train_size + len(sample_to_add)
    
def test_active_ml_model_infoForProgress(active_ml_model,sample_with_label_healthblighted):
    # Given an active ml_model with dummy data
    # Assuming the labeling is either H for healthy or B for blighted 
    # Then it'll return the current images listed as H and B 

    sample = sample_with_label_healthblighted
    ml_model = active_ml_model

    # Train_files is set to empty in the beginning so running info for progress
    # Would result in index out of bounds, data need to be appended to train_files first

    train_set = sample
    train_img_names = train_set.index.tolist()  # Extracting the image names from the DataFrame index

    ml_model.train_model_add(train_set)

    for name in train_img_names:
        ml_model.train_files.append(name)

    health_pic, blight_pic = ml_model.infoForProgress()

    assert len(health_pic) > 1  # Assuming there's at least 1 health_pic
    assert len(blight_pic) > 1  # Assuming there's at least 1 blight pic


def test_active_ml_model_infoForResults(active_ml_model, sample_with_label_healthblighted):
    # Given an active ml_model with dummy data
    # When trained and appended data 
    # Then it'll return a list or a tuple based on the variable

    sample = sample_with_label_healthblighted
    ml_model = active_ml_model

    # Train_files is set to empty in the beginning so running info for progress
    # Would result in index out of bounds, data need to be appended to train_files first

    train_set = sample
    train_img_names = train_set.index.tolist()  # Extracting the image names from the DataFrame index

    ml_model.train_model_add(train_set)

    for name in train_img_names:
        ml_model.train_files.append(name)

    health_pic_user, blight_pic_user, new_health_pic, new_blight_pic, new_health_pic_prob, new_blight_pic_prob = ml_model.infoForResults(sample_with_label_healthblighted)

    #asserting the intended instance type for each variable
    assert isinstance(health_pic_user, list)
    assert isinstance(blight_pic_user, list)
    assert isinstance(new_health_pic, tuple)
    assert isinstance(new_blight_pic, tuple)
    assert isinstance(new_health_pic_prob, tuple)
    assert isinstance(new_blight_pic_prob, tuple)


