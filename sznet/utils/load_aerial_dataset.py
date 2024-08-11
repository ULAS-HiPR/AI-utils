import os
import shutil
import numpy as np
import cv2
from PIL import Image, ImageDraw


'''
Path should be a roboflowv8 format data set
Structure should be as follows:
train/
      /images
      /labels
valid/
      /images
      /labels
'''

            

def load_dataset(path: str, output_path: str, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_OF_CLASSES, generate_data=True, train_or_val="train"):
    label_files = os.listdir(f'{path}/{train_or_val}/labels')
    image_files = os.listdir(f'{path}/{train_or_val}/images')
    
    label_file_ids = [x.split('.txt')[0] for x in label_files]
    image_file_ids = [x.split('.jpg')[0] for x in image_files]
    overlap = set(label_file_ids).intersection(set(image_file_ids))
    
    if not generate_data:
        return list(overlap)
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    
    
    
    # Convert label polygons to n-channel image (n=no. of classes) and save as .npy
    for label_file in label_files:
        label_path = f'{path}/{train_or_val}/labels/{label_file}'
        polygons = open(label_path, "r").readlines()
        label_output_image = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH, NUM_OF_CLASSES))
        for polygon in polygons:
            split_polygon = polygon.strip().split(' ')
            class_id = split_polygon[0]
            points = [(float(split_polygon[i]) * IMAGE_WIDTH, float(split_polygon[i + 1]) * IMAGE_HEIGHT) for i in range(1, len(split_polygon), 2)]
            img = Image.new('L', (IMAGE_HEIGHT, IMAGE_WIDTH), 0)
            ImageDraw.Draw(img).polygon(points, outline=1, fill=1)
            label_output_image[:,:,int(class_id)] = np.array(img)
        
        np.save(output_path + '/' + label_file.replace('.txt', '') + "__LABEL__.npy", label_output_image)
        
        image_path = f'{path}/{train_or_val}/images/{label_file.replace('.txt', '.jpg')}'
        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation = cv2.INTER_LINEAR)
        np.save(output_path + '/' + label_file.replace('.txt', '') + "__IMAGE__.npy", image)