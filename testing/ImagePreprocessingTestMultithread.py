import sys
sys.path.append('..')

import app
from app.ImagePreprocessingMultithread import ImagePreprocessing

ImagePreprocessing("../images_drone/images_drone/",True,16)