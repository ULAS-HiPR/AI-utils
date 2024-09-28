import cv2
import numpy as np
import os
from multiprocessing import Pool

def multi_to_single_channel(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHAGED)
    single_channel_image = np.zeros((500,500))
    for i in range(0, image.max()+1):
        single_channel_image = np.where(image == i, i, single_channel_image)
    cv2.imwrite(image_path, single_channel_image)
    
files = os.listdir('/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap_SINGLE_CH/labels/')

for file in files:
    path = '/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap_SINGLE_CH/labels/' + file
    multi_to_single_channel(path)
    

