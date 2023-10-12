import os
import csv
import random
import sys
from os.path import isfile, join

def get_random_sample(imgPath,kVal):
        files = []
        for f in os.listdir(imgPath):
                if (isfile(join(imgPath,f))):
                        files.append(join(imgPath,f))
        sample_files = random.sample(files,k=kVal)
        return sample_files

sample_images = get_random_sample(str(sys.argv[1]),int(sys.argv[2]))