import os
import shutil
import random

path = "/Users/conor/Downloads/OpenEarthMap_wo_xBD/"
folders = os.listdir(path)
output_path = "/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap/"

def get_files(folder_path):
    if not os.path.exists(output_path + 'images'):
        os.mkdir(output_path + 'images')
    if not os.path.exists(output_path + 'labels'):
        os.mkdir(output_path + 'labels')
    
    if os.path.isdir(folder_path):
        subfolders = os.listdir(folder_path)
    if not ("images" in subfolders and "labels" in subfolders):
        return
    
    images_path = folder_path + "/images/"
    labels_path = folder_path + "/labels/"
    
    images = os.listdir(images_path)
    labels = os.listdir(labels_path)
    
    intersection = list(set(images).intersection(set(labels)))
    
    for frame in intersection:
        shutil.copy(images_path + frame, output_path + 'images/')
        shutil.copy(labels_path + frame, output_path + 'labels/')


for folder in folders:
    if os.path.isdir(path + folder):
        get_files(path + folder)