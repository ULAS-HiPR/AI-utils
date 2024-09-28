import numpy as np
import cv2
import os
import shutil
from multiprocessing import Pool

IMAGE_DIMENSIONS = 500, 500
NUM_OF_CLASSES = 4

def convert_shape(shape, image_dimensions):
    width, height = image_dimensions
    points = np.zeros((len(shape), 2))
    for index, point in enumerate(shape):
        x, y = point[0]
        points[index] = [x/width,y/height]
    return points
    

def get_contours(image_path: str, num_of_classes: int):
    
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype('uint8')
    all_contours = []
    for i in range(0, num_of_classes):
        mask = np.where(image == i, 255, 0).astype('uint8')
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        all_contours.append(contours)
    return all_contours

def write_polygon_to_file(contours, output_path, image_dimensions):
    with open(output_path, "w") as file:
        for i in range(0, NUM_OF_CLASSES):
            for shape in contours[i]:
                file.write(f"{i} " + ' '.join([str(x) + ' ' + str(y) for [x, y] in convert_shape(shape, image_dimensions)]) + "\n")
    
def execute_operation(file_path):
    contours = get_contours(file_path, NUM_OF_CLASSES)
    write_polygon_to_file(contours, file_path.replace('.tif', '.txt'), IMAGE_DIMENSIONS)
    os.remove(file_path)

def process_folder(path):
    files = os.listdir(path)
    with Pool(20) as p:
        p.map(execute_operation, [path + '/' + file for file in files])
    
        

process_folder('/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap_YOLOV8/labels/train')
process_folder('/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap_YOLOV8/labels/val')