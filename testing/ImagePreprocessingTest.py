import sys
sys.path.append('..')

import app
from app.ImagePreprocessing import ImagePreprocessing

ImagePreprocessing("../images_drone/images_drone/",True)