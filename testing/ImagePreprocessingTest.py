import sys
sys.path.append('..')

import app
from app.ImagePreprocessing import ImagePreprocessing

ImagePreprocessing("../TrainImages/",True)