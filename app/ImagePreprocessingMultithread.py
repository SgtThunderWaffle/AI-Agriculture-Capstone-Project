# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:20:09 2020
@author: Donovan
"""

import cv2
import os
import csv
import numpy
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
import numpy as np
from threading import Thread
import math

class ImagePreprocessing:
    """@package ImagePreprocessing
This class extracts a number of features from the images and saves them in a CSV
to be used by the machine learning class.

"""

    import cv2
    #conda install -c conda-forge opencv=3.4.1
    #3-Clause BSD License
    
    import os
    import csv
    import numpy
    from skimage.io import imread, imshow
    import matplotlib.pyplot as plt
    from skimage.color import rgb2hsv
    from threading import Thread
    import math
    #conda install -c anaconda scikit-image
    #BSD 3-Clause
    
    def __init__(self, folder_name, isTraining, threadCountPer):
        
        #Please update these column labels if you add features in order to help with feature selection.
        columnLabels = ('fileName','fvhu','fvhu2','fvhu3','fvhu4','fvhu5','fvhu6','fvhu7',
                        'fvha1','fvha2','fvha3','fvha4','fvha5','fvha6','fvha7','fvha7',
                        'fvha8','fvha9','fvha10','fvha11','fvha12',
                        'gray_mean', 'red_mean', 'green_mean', 'blue_mean', 'num_brown_red', 'num_brown_green', 
                        'num_brown_blue', 'numForegroundPxls', 'blightedHSV_pxls', 'blightedHSV_ratio', 
                        'numRGB_blightedPxls', 'blightedRGBRatio', 'RGB_and_HSV_blighted', 'RGB_and_HSV_both_ratio', 'label')
        if (isTraining):
            #halfway stop index in each txt file is 1410
            blighted_set = self.getFeaturesMultithread("unhealthySet", folder_name, 'B', 0, 1877, threadCountPer)
            healthy_set = self.getFeaturesMultithread("healthySet", folder_name, 'H', 0, 1877, threadCountPer)
            
            for td in blighted_set[0]:
                td.join()
            for td in healthy_set[0]:
                td.join()
                
            blighted_features = []
            for features in blighted_set[1]:
                for feature in features:
                    blighted_features.append(feature)
            healthy_features = []
            for features in healthy_set[1]:
                for feature in features:
                    healthy_features.append(feature)
        
            #blighted_features = self.allSetFiles("unhealthySet", folder_name, 'B', 1411)
            #healthy_features = self.allSetFiles("healthySet", folder_name, 'H', 1411)
            with open('csvOut_train.csv','w', newline = '') as csvfile:
                obj = csv.writer(csvfile)
                obj.writerow(columnLabels)
                obj.writerows(blighted_features)
            with open('csvOut_train.csv','a', newline = '') as csvfile:
                obj = csv.writer(csvfile)
                obj.writerows(healthy_features)
            print(blighted_features)
            print(healthy_features)
        else: #Fix this
            test_set_1 = self.getFeaturesMultithread("unhealthySet", folder_name, 'NA', 1878, 2821, threadCountPer)
            test_set_2 = self.getFeaturesMultithread("healthySet", folder_name, 'NA', 1878, 2817, threadCountPer)
            
            for td in test_set_1[0]:
                td.join()
            for td in test_set_2[0]:
                td.join()
            
            features = []
            for features in test_set_1[1]:
                for feature in features:
                    features.append(feature)
            for features in test_set_2[1]:
                for feature in features:
                    features.append(feature)
            
            with open('csvOut_test.csv','w',newline='') as csvfile:
                obj = csv.writer(csvfile)
                obj.writerow(columnLabels)
                obj.writerows(features)
            print(features)
            
    def getFeaturesMultithread(self, setListName, dir_name, label, startIndex, stopIndex, threadCount):
        features = []
        threads = []
        increment = math.floor((stopIndex+1)/threadCount)
        currentStart = startIndex
        currentStop = currentStart + increment - 1
        for x in range(0,threadCount):
            features.append([])
            if (stopIndex - currentStop < increment):
                currentStop = stopIndex
            threads.append(Thread(target=self.allSetFiles, args=(setListName, dir_name, label, currentStart, currentStop, features[x])))
            currentStart = currentStart + increment
            currentStop = currentStop + increment
            threads[x].start()
        return (threads,features)
            
    
    def getAdvancedFeatures(self,imageIn):
        """
        Returns a tuple of advanced features.

        Parameters
        ----------
        imageIn : Image
            The image to process.

        Returns
        -------
        returnValues : tuple
            numbers.

        """
        
        lowRed = 165
        highRed = 240
        lowGreen = 160
        highGreen = 200
        lowBlue = 135
        highBlue = 240
        
        rgb_img = imageIn
        red = rgb_img[:, :, 0]
        hsv_img = rgb2hsv(rgb_img)
        hue_img = hsv_img[:, :, 0]
        sat_img = hsv_img[:, :, 1]
        value_img = hsv_img[:, :, 2]
        
        #saturation mask to isolate foreground
        satMask = (sat_img > .11) | (value_img > .3)
        #hue and value mask to remove additional brown from background
        mask = (hue_img > .14) | (value_img > .48)
        #healthy corn mask to remove healthy corn, leaving only blighted pixels
        nonBlightMask = hue_img < .14
        #get foreground
        rawForeground = np.zeros_like(rgb_img)
        rawForeground[mask] = rgb_img[mask]
        #reduce brown in background
        foreground = np.zeros_like(rgb_img)
        foreground[satMask] = rawForeground[satMask]
        #get blighted pixels from foreground
        blightedPixels = np.zeros_like(rgb_img)
        blightedPixels[nonBlightMask] = foreground[nonBlightMask]
        #combine into one band
        blightedHSV = np.bitwise_or(blightedPixels[:,:,0], blightedPixels[:,:,1])
        blightedHSV = np.bitwise_or(blightedHSV, blightedPixels[:,:,2])
        
        red = rgb_img[:, :, 0]
        green = rgb_img[:, :, 1]
        blue = rgb_img [:, :, 2]
        binary_green = lowGreen < green
        binary_blue = lowBlue < blue
        binary_red = lowRed < red 
        RGB_Blights = np.bitwise_and(binary_red, binary_green)
        #'brown' pixels within each RGB threshold
        RGB_Blights = np.bitwise_and(RGB_Blights, binary_blue)
        HSV_and_RGB = np.bitwise_and(RGB_Blights, blightedHSV)
        #get features
        numForegroundPixels = np.count_nonzero(foreground)
        numBlightedHSVPixels = np.count_nonzero(blightedHSV)
        blightedHSVRatio = numBlightedHSVPixels / numForegroundPixels
        num_RGB_blightedPixels = np.count_nonzero(RGB_Blights)
        blightedRGBRatio = num_RGB_blightedPixels / numForegroundPixels 
        numBlightedBothPixels = np.count_nonzero(HSV_and_RGB)
        blightedBothRatio = numBlightedBothPixels / numForegroundPixels  
        returnValues = (numForegroundPixels, numBlightedHSVPixels, blightedHSVRatio, num_RGB_blightedPixels,
                        blightedRGBRatio, numBlightedBothPixels, blightedBothRatio)
       
        return returnValues
    
    
    def avgGray(self,image):
        grayscaleArray = numpy.reshape(image, -1)
        gray_mean = numpy.mean(grayscaleArray)
        return gray_mean
    
    def avgRed(self,image):
        red = image[0:4000, 0:6000, 0]
        red = numpy.reshape(red, -1)
        red_mean = numpy.mean(red)
        return red_mean
    
    def avgGreen(self,image):
        green = image[0:4000, 0:6000, 1]
        green = numpy.reshape(green, -1)
        green_mean = numpy.mean(green)
        return green_mean
    
    def avgBlue(self,image):
        blue = image [0:4000, 0:6000, 2]
        blue = numpy.reshape(blue, -1)
        blue_mean = numpy.mean(blue)
        return blue_mean
        
    def numBrownRed(self,image):
        red = image[0:4000, 0:6000, 0]
        red = numpy.reshape(red, -1)
        num_brown_red, bin_edges = numpy.histogram(red, bins=1, range=(180, 250))
        return num_brown_red[0]
    
    def numBrownGreen(self,image):
        green = image[0:4000, 0:6000, 1]
        green = numpy.reshape(green, -1)
        num_brown_green, bin_edges = numpy.histogram(green, bins=1, range=(160, 200))
        return num_brown_green[0]
    
    def numBrownBlue(self,image):
        blue = image [0:4000, 0:6000, 2]
        blue = numpy.reshape(blue, -1)
        num_brown_blue, bin_edges = numpy.histogram(blue, bins=1, range=(150, 240))
        return num_brown_blue[0]
    
    def FdHuMoments(self,image):
        """
        Extracts Hu moments feature from an image
        Parameters
        ----------
        
        image : imread
            The image used for feature extraction
        Returns
        -------
        Feature : Float Array
            The Hu moments in the image.
        Reference
        ---------
        https://gogul.dev/software/image-classification-python
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature
    
    def FdHaralick(self,image):
        import mahotas
        #
        #MIT License
        """
        Extracts Haralick texture feature from an image
        Parameters
        ----------
        
        image : imread
            The image used for feature extraction
        Returns
        -------
        Feature : Float Array
            The Haralick texture in the image.
        Reference
        ---------
        https://gogul.dev/software/image-classification-python
        """
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compute the haralick texture feature vector
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        # return the result
        return haralick
    
    def FdHistogram(self, image, mask=None, bins = 8):
        """
        Extracts color histogram feature from an image
        Parameters
        ----------
        
        image : imread
            The image used for feature extraction
        Returns
        -------
        Feature : Float Array
            The color histogram in the image.
        Reference
        ---------
        https://gogul.dev/software/image-classification-python
        """
        # convert the image to HSV color-space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # compute the color histogram
        hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        # normalize the histogram
        cv2.normalize(hist, hist)
        # return the histogram
        return hist.flatten()
    
    def allFilesInDir(self, dir_name, label):
            csvOut = []
            counter = 0
            for root, dirs, files in os.walk(os.path.abspath(dir_name)):
                for file in files:
    
                    image = imread(os.path.join(root, file), as_gray=True)
                    import matplotlib.pyplot as plt
                    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
                    #plt.show()
                    gray_mean = self.avgGray(image)
    
                    image = imread(os.path.join(root, file))
                    red_mean = self.avgRed(image)
                    green_mean = self.avgGreen(image)
                    blue_mean = self.avgBlue(image)
                    num_brown_red = self.numBrownRed(image)
                    num_brown_green = self.numBrownGreen(image)
                    num_brown_blue = self.numBrownBlue(image)
                    advanced_features = self.getAdvancedFeatures(image)
                    
                    image = cv2.imread(os.path.join(root, file))
                    fv_hu_moments = self.FdHuMoments(image)
                    fv_haralick = self.FdHaralick(image)
    #                fv_histrogram = FdHistogram(image)
    
                    feature_vector = np.hstack([file, fv_hu_moments, fv_haralick, gray_mean, red_mean, green_mean, blue_mean, 
                                                num_brown_red, num_brown_green, num_brown_blue, advanced_features[0], 
                                                advanced_features[1],  advanced_features[2],  advanced_features[3],
                                                 advanced_features[4],  advanced_features[5],  advanced_features[6], label])
                    
                    csvOut.append(feature_vector)
                    counter += 1
                    print(counter)
            return csvOut
            
    def allSetFiles(self, setListName, dir_name, label, startIndex, stopIndex, csvOut):
            #csvOut = []
            counter = 0
            with open("../"+setListName+".txt",'r') as set:
                lines = [line[:-1] for line in set]
                for x in range(startIndex,stopIndex+1):
                    
                    line = lines[x]
                    image = imread(dir_name+line, as_gray=True)
                    import matplotlib.pyplot as plt
                    #plt.imshow(image, cmap='gray', vmin=0, vmax=1)
                    #plt.show()
                    gray_mean = self.avgGray(image)
    
                    image = imread(dir_name+line)
                    red_mean = self.avgRed(image)
                    green_mean = self.avgGreen(image)
                    blue_mean = self.avgBlue(image)
                    num_brown_red = self.numBrownRed(image)
                    num_brown_green = self.numBrownGreen(image)
                    num_brown_blue = self.numBrownBlue(image)
                    advanced_features = self.getAdvancedFeatures(image)
                    
                    image = cv2.imread(dir_name+line)
                    fv_hu_moments = self.FdHuMoments(image)
                    fv_haralick = self.FdHaralick(image)
    #                fv_histrogram = FdHistogram(image)
    
                    feature_vector = np.hstack([line, fv_hu_moments, fv_haralick, gray_mean, red_mean, green_mean, blue_mean, 
                                                num_brown_red, num_brown_green, num_brown_blue, advanced_features[0], 
                                                advanced_features[1],  advanced_features[2],  advanced_features[3],
                                                 advanced_features[4],  advanced_features[5],  advanced_features[6], label])
                    
                    csvOut.append(feature_vector)
                    counter += 1
                    print(counter)
                    #if (counter > count): break
            return csvOut
    
#Main
#folder_name = '../TrainImages/'
#ImagePreprocessing(folder_name,True)
