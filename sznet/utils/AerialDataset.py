import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class AerialDataset(Dataset):
    def __init__(self, path, class_dict, ids, IMAGE_WIDTH, IMAGE_HEIGHT):
        self.path = path
        self.ids = ids
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.class_dict = class_dict

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # image = np.load(f'{self.path}/{self.ids[idx]}.tif')
        # image = np.reshape(image,(3, self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
        # label = np.load(f'{self.path}/{self.ids[idx]}.tif')
        # label = np.reshape(label, (8, self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
        image = cv2.imread(f'{self.path}/images/{self.ids[idx]}.tif', cv2.IMREAD_UNCHANGED)
        # image = np.reshape(image, (3, 500, 500))
        image = np.transpose(image, (2,0,1))
        label = cv2.imread(f'{self.path}/labels/{self.ids[idx]}.tif', cv2.IMREAD_UNCHANGED)
        label = np.reshape(label, (500, 500))
        # label = np.where((label == 1) | (label == 2), 255, 0)
        multi_channel_label = np.zeros((len(self.class_dict), 500, 500))
        for i in range(0, len(self.class_dict)):
            multi_channel_label[i,:,:] = np.where(label == list(self.class_dict.keys())[i], 1, 0)
        multi_channel_label = np.reshape(multi_channel_label, (len(self.class_dict), 500,500))
        return {'image': torch.from_numpy(image), 'mask': torch.from_numpy(multi_channel_label)}
