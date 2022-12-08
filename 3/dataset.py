import os
import h5py

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """needs __init__, __len__ and __getitem__"""
    def __init__(self, data_dir, mode='train'):
        """the case of loading all data into memory"""
        with h5py.File(os.path.join(data_dir,f'{mode}.hdf5'), 'r') as f:
            self.X, self.Y = f['image'][...], f['label'][...]
        
        # preprocessing whole dataset
        self.X = torch.from_numpy(self.X/255.0).float()
        self.Y = torch.from_numpy(self.Y)

    def __len__(self):
        """returns the num of samples in dataset"""
        return len(self.X)

    def __getitem__(self, idx):
        """returns a tuple -> a pair of image and label"""
        image = self.X[idx]
        label = self.Y[idx]
        return image, label