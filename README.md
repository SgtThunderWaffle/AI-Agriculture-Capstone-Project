# AI-Agriculture-Capstone-Project
Repository for AI Capstone Project, teaching farmers how to effectively use Machine Learning and AI

Instructions:
Test data can be organized into the sub-directories of TrainImages/ by running PopulateImages3.ps1. Populating the data uses maxAreas.json, so the script will recreate it if no copy is found. Then, ImagePreprocessingTest.py is run from within the testing/ directory, followed by MLModelTest.py (from the same directory). These files will generate training data from the sample, train a model based on said sample, and then predict values using data derived from test images.

Current Status:
Code has been reconnected and successfully ran in flask, and the source of the images and their metadata has been found. In addition, code has been implemented to select training data from the set of total images based on blight area sizes. Then a test program extracts data from random test samples and creates/trains a model to predict image state.

Next Steps:
Creating our own image dataset and repository for the drone images, processing those images into a new CSV file, and disconnecting the old CSV file from the network (a more permanent CSV file/model needs to be created and refined). Refining methods used to manipulate data and models for our specific needs (more functionality needs to be added, as the current code is too rigid). Add live training functionality in conjunction with modified methods.
