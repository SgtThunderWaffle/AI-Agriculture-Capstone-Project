# -*- coding:utf-8 -*-
"""@package web
This method is responsible for the inner workings of the different web pages in this application.
"""
from flask import Flask
from flask import render_template, flash, redirect, url_for, session, request, jsonify
from app import app
from app.DataPreprocessing import DataPreprocessing
from app.ML_Class_New import ML_Model, Active_ML_Model, load_model, tempload_model, generate_token
from app.SamplingMethods import lowestPercentage
from app.forms import LabelForm
from flask_bootstrap import Bootstrap
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import numpy as np
import boto3
from io import StringIO

bootstrap = Bootstrap(app)
al_model = None

def getData():
    """
    Gets and returns the csvOut.csv as a DataFrame.

    Returns
    -------
    data : Pandas DataFrame
        The data that contains the features for each image.
    """
    #s3 = boto3.client('s3')
    #path = 's3://cornimagesbucket/csvOut.csv'

    print("getData()ISCALLED#############################################")

    path = 'app/csvOut.csv'

    data = pd.read_csv(path, index_col = 0, header = None)
    data.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35']

    print("Original DataFrame:")
    print(data)

    #data_mod = data.astype({'8': 'int32','9': 'int32','10': 'int32','12': 'int32','14': 'int32'})
    data_mod = data

    print("\nDataFrame after type casting:")
    print(data_mod)

    return data_mod.iloc[1:, :]

def load_defaultMLmodel(data):
    """
    Loads the default machine learning model from it's special token folder
    
    Returns
    -------
    ml_model : ML_Model class object
        the model loaded
    """
    print("load_defaultMLmodel(data)ISCALLED#############################################")
    ml_model = load_model("../Models/", "default_model")
    return ml_model

def createMLModel(data,train):
    """
    Prepares the training set and creates a machine learning model using the training set.

    Parameters
    ----------
    data : Pandas DataFrame
        The data that contains the features for each image

    Returns
    -------
    ml_model : ML_Model class object
        ml_model created from the training set.
    train_img_names : String
        The names of the images.
    """
    print("createMLmodel(data)ISCALLED#############################################")
    train_img_names, train_img_label = list(zip(*session['train']))
    train_set = data.loc[train_img_names, :]
    train_set = train_set.iloc[:,:-1].assign(label=train_img_label)
    default_modelPath = 'Models/'
    default_tokenPath = session['token']
    default_tempPath = 'tempdata/'
    al_model = tempload_model(default_tempPath,default_tokenPath)
    if train:
        al_model.train_model_add(train_set)
    print("\n\n\n\n\nAL Model Y: ",end='')
    print(al_model.Y)
    print("\n\n\n\n")
    al_model.tempsave_model()
    al_model.visualize_model(5)
    return al_model, train_img_names

def renderLabel(form):
    """
    prepairs a render_template to show the label.html web page.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    print("renderLabel(form)ISCALLED#############################################")
    queue = session['queue']
    img = queue.pop()
    session['queue'] = queue
    return render_template(url_for('step4Labeling'), form = form, picture = img, confidence = session['confidence'])

def initializeAL(form, confidence_break = .7):
    """
    Initializes the active learning model and sets up the webpage with everything needed to run the application.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html
    confidence_break : number
        How confident the model is.

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    print("intializeAL(form, confidence_break =.7)ISCALLED#############################################")
    preprocess = DataPreprocessing(True)
    ml_classifier = RandomForestClassifier()
    default_modelPath = 'Models/'
    default_tokenPath = generate_token()
    default_tempPath = 'tempdata/'
    data = getData()
    al_model = Active_ML_Model(ml_classifier, preprocess,data,default_tokenPath,default_modelPath,default_tempPath)
    al_model.tempsave_model()
    print('DEBUGGING')
    print(al_model.__dict__) 
    session['confidence'] = 0
    session['confidence_break'] = confidence_break
    session['labels'] = []
    session['sample_idx'] = list(al_model.sample.index.values)
    session['test'] = list(al_model.test.index.values)
    session['train'] = al_model.train
    session['model'] = True
    session['token'] = default_tokenPath
    session['modeldir'] = default_modelPath
    session['tempdir'] = default_tempPath
    session['queue'] = list(al_model.sample.index.values)
    return renderLabel(form)

def getNextSetOfImages(form, sampling_method):
    """
    Uses a sampling method to get the next set of images needed to be labeled.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html
    sampling_method : SamplingMethods Function
        function that returns the queue and the new test set that does not contain the queue.

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    print("getNextSetOfImages(form, sampling_method)ISCALLED#############################################")
    data = getData()
    ml_model, train_img_names = createMLModel(data,False)
    test_set = data[data.index.isin(train_img_names) == False]

    session['sample_idx'], session['test'] = sampling_method(ml_model, test_set.iloc[:,:-1], 5)
    session['queue'] = session['sample_idx'].copy()

    return renderLabel(form)

def prepairResults(form):
    """
    Creates the new machine learning model and gets the confidence of the machine learning model.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html

    Returns
    -------
    render_template : flask function
        renders the appropriate webpage based on new confidence score.
    """
    print("prepairResults(form)ISCALLED#############################################")
    session['labels'].append(form.choice.data)
    session['sample'] = tuple(zip(session['sample_idx'], session['labels']))

    #if session['train'] != None:
        #session['train'] = session['train'] + session['sample']
    #else:
    session['train'] = session['sample']

    data = getData()
    ml_model, train_img_names = createMLModel(data,True)

    session['confidence'] = np.mean(ml_model.K_fold())
    session['labels'] = []

    if session['confidence'] < session['confidence_break']:
        health_pic, blight_pic = ml_model.infoForProgress(train_img_names)
        return render_template('step5Intermediate.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic, blight_user = blight_pic, healthNum_user = len(health_pic), blightNum_user = len(blight_pic))
    else:
        test_set = data.loc[session['test'], :]
        health_pic_user, blight_pic_user, health_pic, blight_pic, health_pic_prob, blight_pic_prob = ml_model.infoForResults(train_img_names, test_set)
        return render_template('step5Final.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic_user, blight_user = blight_pic_user, healthNum_user = len(health_pic_user), blightNum_user = len(blight_pic_user), health_test = health_pic, unhealth_test = blight_pic, healthyNum = len(health_pic), unhealthyNum = len(blight_pic), healthyPct = "{:.2%}".format(len(health_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), unhealthyPct = "{:.2%}".format(len(blight_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), h_prob = health_pic_prob, b_prob = blight_pic_prob)

@app.route("/", methods=['GET'])

@app.route("/index.html",methods=['GET'])
def home():
    print("RUNNING INDEX.HTML APP ROUTE")
    """
    Operates the root (/) and index(index.html) web pages.
    """
    session.pop('model', None)
    return render_template('index.html')

@app.route('/aiExplained.html')
def ai_explained():
    print("RUNNING AIEXPLAINED APP ROUTE")
    return render_template('aiExplained.html')

@app.route("/applicationExplained.html")
def applicationExplained():
    """
    Operates the applicationExplained.html web page.
    """
    return render_template('applicationExplained.html')

@app.route("/step1.html")
def step1():
    """
    Operates the step1.html web page.
    """
    return render_template('step1.html')

@app.route("/step2.html")
def step2():
    """
    Operates the step2.html web page.
    """
    return render_template('step2.html')

@app.route("/step3.html")
def step3():
    """
    Operates the step3.html web page.
    """
    return render_template('step3.html')

@app.route("/step4.html")
def step4():
    """
    Operates the step4.html web page.
    """
    return render_template('step4.html')

@app.route("/step5.html")
def step5():
    """
    Operates the step5.html web page.
    """
    return render_template('step5.html')

@app.route("/step6.html")
def step6():
    """
    Operates the step6.html web page.
    """
    return render_template('step6.html')

@app.route("/step4Labeling.html" ,methods=['GET', 'POST'])
def step4Labeling():
    """
    Operates the step4Labeling.html web page.
    Where Labeling of Leaf as Healthy/Unhealthy Happens.
    """
    print("RUNNING STEP4LABELING.HTML APP ROUTE")
    form = LabelForm()
    if 'model' not in session:#Start
        print("initialize al")
        return initializeAL(form, .7)

    elif session['queue'] == [] and session['labels'] == []: # Need more pictures
        return getNextSetOfImages(form, lowestPercentage)

    elif form.is_submitted() and session['queue'] == []:# Finished Labeling
        return prepairResults(form)

    elif form.is_submitted() and session['queue'] != []: #Still gathering labels
        session['labels'].append(form.choice.data)
        return renderLabel(form)

    return render_template('step4Labeling.html', form=form)

@app.route("/step5Intermediate.html")
def step5Intermediate():
    """
    Operates the step5.html web page.
    """
    return render_template('step5Intermediate.html')

@app.route("/step5Final.html")
def step5Final():
    """
    Operates the step5.html web page.
    """
    return render_template('step5Final.html')

@app.route("/feedback/<h_list>/<u_list>/<h_conf_list>/<u_conf_list>",methods=['GET'])
def step6Feedback(h_list,u_list,h_conf_list,u_conf_list):
    """
    Operates the step6Feedback.html web page.
    """
    print("RUNNING STEP6FEEDBACK.HTML APP ROUTE")
    h_feedback_result = list(h_list.split(","))
    u_feedback_result = list(u_list.split(","))
    h_conf_result = list(h_conf_list.split(","))
    u_conf_result = list(u_conf_list.split(","))
    h_length = len(h_feedback_result)
    u_length = len(u_feedback_result)
    
    return render_template('step6Feedback.html', healthy_list = h_feedback_result, unhealthy_list = u_feedback_result, healthy_conf_list = h_conf_result, unhealthy_conf_list = u_conf_result, h_list_length = h_length, u_list_length = u_length)

'''
@app.route("/label.html",methods=['GET', 'POST'])
def label():
    """
    Operates the label(label.html) web page.
    """
    print("RUNNING LABEL.HTML APP ROUTE")
    form = LabelForm()
    if 'model' not in session:#Start
        print("initialize al")
        return initializeAL(form, .7)

    elif session['queue'] == [] and session['labels'] == []: # Need more pictures
        return getNextSetOfImages(form, lowestPercentage)

    elif form.is_submitted() and session['queue'] == []:# Finished Labeling
        return prepairResults(form)

    elif form.is_submitted() and session['queue'] != []: #Still gathering labels
        session['labels'].append(form.choice.data)
        return renderLabel(form)

    return render_template('label.html', form = form)

@app.route("/intermediate.html",methods=['GET'])
def intermediate():
    """
    Operates the intermediate(intermediate.html) web page.
    """
    print("RUNNING INTERMEDIATE.HTML APP ROUTE")
    return render_template('intermediate.html')

@app.route("/final.html",methods=['GET'])
def final():
    """
    Operates the final(final.html) web page.
    """
    print("RUNNING FINAL.HTML APP ROUTE")
    return render_template('final.html')

@app.route("/feedback/<h_list>/<u_list>/<h_conf_list>/<u_conf_list>",methods=['GET'])
def feedback(h_list,u_list,h_conf_list,u_conf_list):
    """
    Operates the feedback(feedback.html) web page.
    """
    print("RUNNING FEEDBACK.HTML APP ROUTE")
    h_feedback_result = list(h_list.split(","))
    u_feedback_result = list(u_list.split(","))
    h_conf_result = list(h_conf_list.split(","))
    u_conf_result = list(u_conf_list.split(","))
    h_length = len(h_feedback_result)
    u_length = len(u_feedback_result)
    
    return render_template('feedback.html', healthy_list = h_feedback_result, unhealthy_list = u_feedback_result, healthy_conf_list = h_conf_result, unhealthy_conf_list = u_conf_result, h_list_length = h_length, u_list_length = u_length)

#app.run( host='127.0.0.1', port=5000, debug='True', use_reloader = False)
'''
