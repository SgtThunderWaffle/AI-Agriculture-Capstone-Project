import json
from json import JSONEncoder
from uuid import uuid4
from joblib import dump, load
import numpy
import os
from random import randint
import pydot
from sklearn.tree import export_graphviz

def load_model(modeldir, token):
    ml_model = load(modeldir+token+'/model.joblib')
    return ml_model

class ML_Model:
    """
    This class creates a machine learning model based on the data sent,
    data preprocessing, and type of ml classifier.

    """

    def __init__(self, ml_classifier, preprocess, data, token, modeldir):
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
        self.set_token(token)
        if os.path.exists(modeldir+token+'/model.joblib'):
            load_model(token)
        else:
            self.ml_classifier = ml_classifier
            self.preprocess = preprocess
            self.set_token(token)
            self.visual_present = False

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
        accuracies = cross_val_score(self.ml_classifier, self.X, self.y, cv=3)
        return accuracies

    def infoForProgress(self, train_img_names):
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
        y_actual = self.y
        y_pic = train_img_names
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
            y_pic_head = y_pic[:10]
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

    def infoForResults(self, train_img_names, test):
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
        health_pic_user, blight_pic_user = self.infoForProgress(train_img_names)
        test_pic = list(test.index.values)
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
        
    def visualize_model(self):
        if not self.visual_present:
            models_dir = '../../Models/'
            token_dir = models_dir+self.token+'/'
            if not os.path.exists(models_dir):
                os.mkdir(models_dir)
            if not os.path.exists(token_dir):
                os.mkdir(token_dir)
            estimator_dir = token_dir+'random_estimator.dot'
            random_estimator = self.ml_model.estimators_[randint(0,len(self.ml_model.estimators_))]
            export_graphviz(random_estimator, out_file=estimator_dir, max_depth=3)
            (graph, ) = pydot.graph_from_dot_file(estimator_dir)
            graph.write_png(token_dir+'random_estimator.png')


class Active_ML_Model(ML_Model):
    
    def _init_(self, ml_classifier, preprocess, data, token):
        ML_Model._init_(self, ml_classifier, preprocess, data, token)
    
    def partial_train(data):
        self.X = numpy.add(self.X, self.get_x(data))
        self.Y = numpy.add(self.Y, self.get_y(data))
        
        self.ml_model = self.ml_classifier.fit(self.X, self.Y)