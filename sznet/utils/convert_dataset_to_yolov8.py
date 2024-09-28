import os
import random
import shutil

def make_dir_if_not_exists(path: str):
    if os.path.isdir(path):
        return
    os.mkdir(path)

SPLIT = 0.9
DATASET_PATH = '/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap_YOLOV8/'

frames = os.listdir(DATASET_PATH + '/images')

random.shuffle(frames)

train = frames[:int(len(frames)*.9)]
val = frames[int(len(frames)*.9):]

make_dir_if_not_exists(DATASET_PATH + 'images/train/')
make_dir_if_not_exists(DATASET_PATH + 'images/val/')
make_dir_if_not_exists(DATASET_PATH + 'labels/train/')
make_dir_if_not_exists(DATASET_PATH + 'labels/val/')

for frame in train:
    shutil.move(DATASET_PATH + '/images/' + frame, DATASET_PATH + '/images/train/' + frame)
    shutil.move(DATASET_PATH + '/labels/' + frame, DATASET_PATH + '/labels/train/' + frame)
    
for frame in val:
    shutil.move(DATASET_PATH + '/images/' + frame, DATASET_PATH + '/images/val/' + frame)
    shutil.move(DATASET_PATH + '/labels/' + frame, DATASET_PATH + '/labels/val/' + frame)