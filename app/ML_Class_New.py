import json
from json import JSONEncoder
from uuid import uuid4
from joblib import dump, load
import pickle
import numpy
import os
import shutil
from random import randint
import pydot
from sklearn.tree import export_graphviz
from sklearn.utils import shuffle
import pandas as pd

def load_model(modeldir, token):
    ml_model = load(modeldir+token+'/model.joblib')
    return ml_model
    
def tempload_model(tempdir, token):
    ml_model = load(tempdir+token+'.joblib')
    return ml_model
    
def generate_token():
    return str(uuid4())

class ML_Model:
    """
    This class creates a machine learning model based on the data sent,
    data preprocessing, and type of ml classifier.

    """

    def __init__(self, ml_classifier, preprocess, data, token, modeldir, tempdir):
        """
        This function controls the initial creation of the machine learning model.

        Parameters
        ----------
        train_data : pandas DataFrame
            The data the machine learning model will be built on.
        ml_classifier : classifier object
            The classifier to be used to create the machine learning model.
        preprocess : Python Function
            The function used to preprocess the data before model creation.

        Attributes
        -------
        ml_classifier : classifier object
            The classifier to be used to create the machine learning model.
        preprocess : Python Function
            The function used to preprocess the data before model creation.
        X : pandas DataFrame
            The features in the train set.
        y : pandas Series
            The responce variable.
        ml_model : fitted machine learning classifier
            The machine learning model created using the training data.
        """
        self.modeldir = modeldir
        self.tempdir = tempdir
        self.set_token(token)
        self.train_files = []

        # Debugging prints
        print(f"preprocess type: {type(preprocess)}") # added code
        print(f"preprocess attributes: {dir(preprocess)}") # added code

        #Updates needed to web.py to decide when to create new model and when to call load_model()
        self.ml_classifier = ml_classifier
        self.preprocess = preprocess
        self.train_model(data)
    
    def get_x(self,data):
        return self.preprocess.fit_transform(data.iloc[:,:-1].values)
        
    def get_y(self,data):
        return data.iloc[:,-1:].values.ravel()
    
    def train_model(self, data):
        self.X = self.get_x(data)
        self.Y = self.get_y(data)

        self.ml_model = self.ml_classifier.fit(self.X, self.Y) 
    
    def set_token(self, token):
        self.token = token
        
    def get_token(self):
        return self.token

    def GetKnownPredictions(self, new_data):
        """
        This function predicts the labels for a new set of data that contains labels.
        It returns these predictions and the probability.

        Parameters
        ----------
        new_data : pandas DataFrame
            The new data to be labeled.

        Returns
        -------
        y_prediction : list
            list of predicted labels.
        prob : list
            The probability that the label is correct.
        """
        new_data_X = self.preprocess.fit_transform(new_data.iloc[:, :-1].values)
        y_prediction = self.ml_model.predict(new_data_X)
        y_probabilities = self.ml_model.predict_proba(new_data_X)
        y_probabilities = [max(prob) for prob in y_probabilities]
        return y_prediction, max(y_probabilities)

    def GetUnknownPredictions(self, new_data_X):
        """
        This function predicts the labels for a new set of data that does not contains labels.
        It returns these predictions and the probability.

        Parameters
        ----------
        new_data : pandas DataFrame
            The new data to be labeled.

        Returns
        -------
        y_prediction : list
            list of predicted labels.
        prob : list
            The probability that the label is correct.
        """
        new_data_X = self.preprocess.transform(new_data_X)
        y_prediction = self.ml_model.predict(new_data_X)
        y_probabilities = self.ml_model.predict_proba(new_data_X)
        y_probabilities = [max(prob) for prob in y_probabilities]
        return y_prediction, y_probabilities

    def K_fold(self):
        """
        This function performs a 10-fold cross-validation and returns the accuracies of each fold.

        Returns
        -------
        accuracies : list
            The 10 accuracy values using 10-fold cross-validation.
        """
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(self.ml_classifier, self.X, self.Y, cv=3)
        return accuracies

    def infoForProgress(self):
        """
        This function returns the information nessessary to display the progress of the active learning model.

        Parameters
        ----------
        train_img_names : list
            list of image names in the training set.

        Returns
        -------
        health_pic : list
            List of images that were predicted as healthy.

        blight_pic : list
            List of images that were predicted as unhealthy.
        """
        y_actual = self.Y
        y_pic = self.train_files
        #y_pred = self.ml_model.predict(self.X)
        #y_pred = list(y_pred)
        health_pic = []
        blight_pic = []
        if len(y_pic) == 10:
            y_pic = y_pic[::-1]
            for y_idx, y in enumerate(y_actual):
                if y == 'H':
                    health_pic.append(y_pic[y_idx])
                elif y == 'B':
                    blight_pic.append(y_pic[y_idx])
        else:
            y_pic_head = y_pic
            y_pic_head_rev = y_pic_head[::-1]
            y_pic_result = y_pic_head_rev
            y_pic_tail = y_pic[10:]
            y_pic_tail_length = len(y_pic_tail)
            for i in range(5,y_pic_tail_length + 5,5):
                y_pic_tail_rev = y_pic_tail[:5]
                y_pic_tail_rev = y_pic_tail_rev[::-1]
                y_pic_result = y_pic_result + y_pic_tail_rev
                y_pic_tail = y_pic_tail[5:]
            for y_idx, y in enumerate(y_actual):
                if y == 'H':
                    health_pic.append(y_pic_result[y_idx])
                elif y == 'B':
                    blight_pic.append(y_pic_result[y_idx])
        return health_pic, blight_pic

    def infoForResults(self, test):
        """
        This function returns the information nessessary to display the final results of the active learning model.

        Parameters
        ----------
        train_img_names : list
            list of image names in the training set.
        test : pandas dataframe
            The test set of the machine learning model.

        Returns
        -------
        health_pic_user : list
            List of images that were predicted as healthy.

        blight_pic_user : list
            List of images that were predicted as blight.

        health_pic : list
            List of images in the test set that are predicted to being healthy.

        blight_pic : list
            List of images in the test set that are predicted to being blighted.
        """
        health_pic_user, blight_pic_user = self.infoForProgress()
        test_pic = list(test.index.values)
        test = test.iloc[:,:-1]
        y_pred, y_prob = self.GetUnknownPredictions(test)
        health_pic = []
        blight_pic = []
        health_pic_prob = []
        blight_pic_prob = []
        for y_idx, y in enumerate(y_pred):
            if y == 'H':
                health_pic.append(test_pic[y_idx])
                health_pic_prob.append(y_prob[y_idx])
            elif y == 'B':
                blight_pic.append(test_pic[y_idx])
                blight_pic_prob.append(y_prob[y_idx])
        health_list = list(zip(health_pic,health_pic_prob))
        blight_list = list(zip(blight_pic,blight_pic_prob))
        health_list_sorted = sorted(health_list, reverse=True, key = lambda x: x[1])
        blight_list_sorted = sorted(blight_list, reverse=True, key = lambda x: x[1])
        if health_pic and health_pic_prob:
            new_health_pic, new_health_pic_prob = list(zip(*health_list_sorted))
        else:
            new_health_pic = []
            new_health_pic_prob = []
        if blight_pic and blight_pic_prob:
            new_blight_pic, new_blight_pic_prob = list(zip(*blight_list_sorted))
        else:
            new_blight_pic = []
            new_blight_pic_prob = []

        return health_pic_user, blight_pic_user, new_health_pic, new_blight_pic, new_health_pic_prob, new_blight_pic_prob
        
    def save_model(self):
        if not os.path.exists(self.modeldir):
            os.mkdir(self.modeldir)
        if not os.path.exists(self.modeldir+self.token+'/'):
            os.mkdir(self.modeldir+self.token+'/')
        dump(self, self.modeldir+self.token+'/model.joblib')
        estimator_file = self.token+'.dot'
        estimator_image = self.token+'.png'
        if os.path.exists(self.tempdir+estimator_file):
            shutil.copy(self.tempdir+estimator_file, self.modeldir+self.token+'/'+estimator_file)
            shutil.copy(self.tempdir+estimator_image, self.modeldir+self.token+'/'+estimator_image)
            self.clear_tempdata()
            
    def tempsave_model(self):
        if not os.path.exists(self.tempdir):
            os.mkdir(self.tempdir)
        dump(self, self.tempdir+self.token+'.joblib')
          
    def clear_tempdata(self):
        try:
            os.remove(self.tempdir+self.token+'.dot')
            os.remove(self.tempdir+self.token+'.png')
            os.remove(self.tempdir+self.token+'.joblib')
        except:
            print("no temp data to clear")
        
    def visualize_model(self, maxdepth):
        if not os.path.exists(self.tempdir):
            os.mkdir(self.tempdir)
        estimator_file = self.tempdir+self.token+'.dot'
        estimator_image = self.tempdir+self.token+'.png'
        #self.clear_tempdata()
        #can change to random later maybe
        estimator = self.ml_model.estimators_[0]
        export_graphviz(estimator, out_file=estimator_file, max_depth=maxdepth, feature_names=['HU Moment 1','HU Moment 2','HU Moment 3','HU Moment 4','HU Moment 5','HU Moment 6','HU Moment 7','Haralick 1','Haralick 2','Haralick 3','Haralick 4','Haralick 5','Haralick 6','Haralick 7','Haralick 8','Haralick 9','Haralick 10','Haralick 11','Haralick 12','Haralick 13','Gray Mean','Green Mean','Red Mean','Blue Mean','Brown/Red','Brown/Green','Brown/Blue','Foreground Pixels','Blighted HSV Pixels','Blighted HSV Ratio','Blighted RGB Pixels','Blighted RGB Ratio','Blighted HSV/RGB Pixels','Blighted HSV/RGB Ratio'])
        (graph, ) = pydot.graph_from_dot_file(estimator_file)
        graph.write_png(estimator_image)
        


class Active_ML_Model(ML_Model):
    """
    This class creates an active learning model based on the data sent,
    data preprocessing, and type of ml classifier.

    """
    def __init__(self, ml_classifier, preprocess, data, token, modeldir, tempdir, n_samples = 10):
        data = shuffle(data)
        sample = data.iloc[:n_samples, :]
        test = data.iloc[n_samples:, :]
        super().__init__(ml_classifier, preprocess, sample, token, modeldir, tempdir)
        self.sample = sample
        self.test = test
        self.train = None
    
    def train_model_add(self, sample):
        if isinstance(self.train, pd.DataFrame):
            self.train = pd.concat([self.train, sample])
        else:
            self.train = sample
        self.train_model(self.train)
    
    def next_samples(self, sampling_method, n_samples=5):
        self.sample, self.test = sampling_method(self.ml_model, n_samples)
        

