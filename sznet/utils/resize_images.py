import cv2
import numpy as np
import os
from multiprocessing import Pool

def resize_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print(image_path)
    image = cv2.resize(image, (500, 500), 
               interpolation = cv2.INTER_LINEAR)
    cv2.imwrite(image_path, image)
    
files = os.listdir('/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap_2_CLASS/labels')

for file in files:
    if file == ".DS_Store":
        continue
    path = '/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap_2_CLASS/labels/' + file
    resize_image(path)
    
files = os.listdir('/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap_2_CLASS/images')

for file in files:
    if file == ".DS_Store":
        continue
    path = '/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap_2_CLASS/images/' + file
    resize_image(path)