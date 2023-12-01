"""
Created November 2023
@author: Patrick
"""
import pytest
import pandas as pd
import numpy as np
from app.DataPreprocessing import DataPreprocessing
import warnings

# while in env use python -m pytest to run

def test_fit_transform():
    # Given a sample DataFrame
    data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
    X_train = pd.DataFrame(data)

    # When creating a Datapreprocessing object with standard scaling
    preprocessing = DataPreprocessing(standard_scaling=True)
    X_train_transformed = preprocessing.fit_transform(X_train)

    # Then the transformed data would have the shape of (3,2)
    assert X_train_transformed.shape == (3, 2)

def test_transform():
    # Given a sample DataFrame
    data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
    X_test = pd.DataFrame(data)

    # When creating a DataPreprocessing objecti with normalization
    preprocessing = DataPreprocessing(normalization=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Ignore warning of normalizer fitted without feature names
        X_test_transformed = preprocessing.transform(X_test)

    # Then the transformed data would have the shape of (3,2)
    assert X_test_transformed.shape == (3, 2)

def test_fit_transform_transform_consistency():
    # Given a sample DataFrame called test
    data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
    X_train = pd.DataFrame(data)

    #When creating a DataPreprocessing object with standard scaling and normalization
    preprocessing = DataPreprocessing(standard_scaling=True, normalization=True)
    X_train_transformed = preprocessing.fit_transform(X_train)

    # Given a sample DataFrame called test
    data_test = {'feature1': [7, 8, 9], 'feature2': [10, 11, 12]}
    X_test = pd.DataFrame(data_test)

    # When testing data is transformed
    X_test_transformed = preprocessing.transform(X_test)

    # Then the transformed testing data will have the same shape as the transformed training data
    assert X_test_transformed.shape == X_train_transformed.shape

def test_no_preprocessing():
    # Given a sample dataframe
    data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
    X_train = pd.DataFrame(data)

    # When creating a default DataPreprocessing object
    preprocessing = DataPreprocessing()

    # Then the preprocessing.fit_transform of X_train would equal X_train
    assert preprocessing.fit_transform(X_train).equals(X_train)

def test_pca_with_components():
    # Given a sample dataframe
    data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
    X_train = pd.DataFrame(data)

    # When creating a DataPreprocessing object with PCA and components of one
    preprocessing = DataPreprocessing(pca=True, components=1)
    X_train_transformed = preprocessing.fit_transform(X_train)

    # Then the tansformed shape would have a dimensionality of 1
    assert X_train_transformed.shape[1] == 1

def test_pca_with_many_components():
     # Given a sample dataframe
    data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
    X_train = pd.DataFrame(data)

    # When creating a DataPreprocessing object with PCA and components of 10
    preprocessing = DataPreprocessing(pca=True, components=10)

    # Then assert that a ValueError is raised because number of components cant 
    # be greater than the number of features, 2 in this case
    with pytest.raises(ValueError, match="n_components=10 must be between 0 and min"):
        X_train_transformed = preprocessing.fit_transform(X_train)