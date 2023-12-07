"""
Created November 2023
@author: Patrick
"""

import pytest
from app.SamplingMethods import lowestPercentage 
import pandas as pd


def test_lowestPercentage_correct_ouput(ml_model, X_test):
    #Given a sample ml_model and X_test
    #When tested for output
    #Then it will return the correct amount of sample

    n = 5  
    original_X_test = X_test.copy()  

    sample_indices, new_test_indices = lowestPercentage(ml_model, X_test, n)

    assert len(sample_indices) == n
    assert len(new_test_indices) == len(original_X_test) - n

    assert set(sample_indices).issubset(set(original_X_test.index))


def test_lowestPercentage_empty_X(ml_model):
    #Given a sample ml_model and X_test
    #When x is empty
    #Then it will be handled by lowestPercentage

    X_test = pd.DataFrame()
    n = 2

    with pytest.raises(ValueError, match="X_test is empty"):
        lowestPercentage(ml_model, X_test, n)


def test_lowestPercentage_large_n(ml_model, X_test):
    #Given a sample ml_model and X_test
    #When n is made too big
    #Then it will be handled by lowestPercentage 

    n = len(X_test) + 1

    with pytest.raises(ValueError, match=f"n exceeds the number of samples in X_test: {n} > {len(X_test)}"):
        lowestPercentage(ml_model, X_test, n)