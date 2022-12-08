import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

# source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__():
        pass

    def __getitem__():
        pass