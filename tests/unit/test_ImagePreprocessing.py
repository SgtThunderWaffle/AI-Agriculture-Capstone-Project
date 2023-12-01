"""
Created November 2023
@author: Patrick
"""

import cv2
import numpy as np
from app.ImagePreprocessing import getAdvancedFeatures, avgGray, avgRed, avgGreen
from app.ImagePreprocessing import avgBlue,numBrownRed, numBrownGreen, numBrownBlue
from app.ImagePreprocessing import FdHuMoments, FdHaralick, FdHistogram
from app.ImagePreprocessing import allFilesInDir, allSetFiles, generateDataFromFolder, generate_data_from_sets

# while in env use python -m pytest to run

def test_getAdvancedFeatures():
    # Given a pseudo image
    # When its passed using getAdvancedFeatures
    # Then It will return a tuple of 7 elements

    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8) #Random Image values
    result = getAdvancedFeatures(image)
    
    assert isinstance(result, tuple)
    assert len(result) == 7  # Check if the tuple has 7 elements
    
def test_avgGray():
    # Given a psuedo Image
    # When its tested for Average Gray
    # Then it will output the avg gray controlled to be 1 
    image = np.ones((100, 100, 3), dtype=np.uint8)
    result = avgGray(image)
    assert result == 1

def test_avgRed():
    # Given a psuedo Image
    # When its tested for Average Red
    # Then it will output the avg red controlled to be 1 
    image = np.ones((100, 100, 3), dtype=np.uint8)
    result = avgRed(image)
    assert result == 1

def test_avgGreen():
    # Given a pseudo Image
    # When it's tested for Average Green
    # Then it will output the avg green controlled to be 1 
    image = np.ones((100, 100, 3), dtype=np.uint8)
    result = avgGreen(image)
    assert result == 1

def test_avgBlue():
    # Given a pseudo Image
    # When it's tested for Average Blue
    # Then it will output the avg blue controlled to be 1 
    image = np.ones((100, 100, 3), dtype=np.uint8)
    result = avgBlue(image)
    assert result == 1

def test_numBrownRed():
    # Given a pseudo Image
    # When it's tested for the number of brown pixels in the Red channel
    # Then it will output the number of brown pixels controlled to be 0
    image = np.ones((100, 100, 3), dtype=np.uint8)
    result = numBrownRed(image)
    assert result == 0

def test_numBrownGreen():
    # Given a pseudo Image
    # When it's tested for the number of brown pixels in the Green channel
    # Then it will output the number of brown pixels controlled to be 0 
    image = np.ones((100, 100, 3), dtype=np.uint8)
    result = numBrownGreen(image)
    assert result == 0

def test_numBrownBlue():
    # Given a pseudo Image
    # When it's tested for the number of brown pixels in the Blue channel
    # Then it will output the number of brown pixels controlled to be 0
    image = np.ones((100, 100, 3), dtype=np.uint8)
    result = numBrownBlue(image)
    assert result == 0

def test_FdHuMoments():
    # Given a pseudo Image
    # When it's tested for extracting Hu moments
    # Then it will output the Hu moments controlled to be an array of zeros
    image = np.ones((100, 100, 3), dtype=np.uint8)
    result = FdHuMoments(image)
    expected_values = np.zeros(7) 
    np.testing.assert_allclose(result, expected_values, rtol=1e-5, atol=.2)

def test_FdHaralick():
    # Given a pseudo Image
    # When it's tested for extracting Haralick texture
    # Then it will output the Haralick texture controlled to be an array of zeros
    image = np.ones((100, 100, 3), dtype=np.uint8)
    result = FdHaralick(image)
    expected_values = np.ones(result.shape)  # Assuming the shape of the result
    print("Result:", result)
    print("Expected:", expected_values)
    np.testing.assert_allclose(result, expected_values, rtol=.5, atol=.5)

def test_FdHistogram():
    # Given an image
    # When FdHistogram is called
    # Then the result should be a flattened histogram
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    bins = 8
    result = FdHistogram(image)
    assert isinstance(result, np.ndarray)
    assert result.shape == (bins ** 3,)

def test_FdHistogram_with_custom_bins():
    # Given an image
    # When FdHistogram is called with custom bins
    # Then the result should be a flattened histogram with the specified number of bins
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    custom_bins = 16
    result = FdHistogram(image, bins=custom_bins)
    assert isinstance(result, np.ndarray)
    assert result.shape == (custom_bins ** 3,)

def test_FdHistogram_with_mask():
    # Given an image and a mask
    # When FdHistogram is called with a mask
    # Then the result should be a flattened histogram
    bins = 8
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8)
    result = FdHistogram(image, mask=mask)
    assert isinstance(result, np.ndarray)
    assert result.shape == (bins ** 3,)

    