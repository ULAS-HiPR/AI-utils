import numpy as np
import torch
from torch.utils.data import Dataset

class AerialDataset(Dataset):
    def __init__(self, path, ids, IMAGE_WIDTH, IMAGE_HEIGHT):
        self.path = path
        self.ids = ids
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = np.load(f'{self.path}/{self.ids[idx]}__IMAGE__.npy')
        image = np.reshape(image,(3, self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
        label = np.load(f'{self.path}/{self.ids[idx]}__LABEL__.npy')
        return {'image': torch.from_numpy(image), 'mask': torch.from_numpy(label)}
