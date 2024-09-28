import cv2
import numpy as np
import os
from multiprocessing import Pool

def resize_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (500, 500), 
               interpolation = cv2.INTER_LINEAR)
    cv2.imwrite(image_path, image)
    
files = os.listdir('/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap/labels')

for file in files:
    path = '/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap/labels/' + file
    resize_image(path)
    
files = os.listdir('/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap/images')

for file in files:
    path = '/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap/images/' + file
    resize_image(path)