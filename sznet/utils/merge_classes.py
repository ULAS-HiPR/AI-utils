import cv2
import numpy as np
import os
from multiprocessing import Pool

def merge_classes(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return
    new_image = np.zeros(image.shape)
    new_image = np.where(image == 1, 1, new_image)
    new_image = np.where(image == 2, 1, new_image)
    new_image = np.where(image == 4, 1, new_image)
    new_image = np.where(image == 7, 1, new_image)
    new_image = np.where(image == 5, 2, new_image)
    new_image = np.where(image == 6, 2, new_image)
    new_image = np.where(image == 3, 3, new_image)
    new_image = np.where(image == 8, 3, new_image)
    cv2.imwrite(image_path, new_image)
    
files = os.listdir('/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap/labels')

# with Pool(20) as p:
#     p.map(merge_classes, files)    
for file in files:
    path = '/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap/labels/' + file
    merge_classes(path)