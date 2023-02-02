import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

LABELS = {
    'airplane' : 0,
    'automobile' : 1,
    'bird' : 2,
    'cat' : 3,
    'deer' : 4,
    'dog' : 5,
    'frog' : 6,
    'horse' : 7,
    'ship' : 8,
    'truck' : 9,
}

def get_label(image_fp: str) -> torch.int64:
    """..cifar10/train/airplane/100.png"""
    label = os.path.split(os.path.split(image_fp)[0])[-1]
    enc_label = LABELS[label]
    enc_label = torch.as_tensor(enc_label, dtype=torch.int64)
    return enc_label

class CifarDS(Dataset):
    def __init__(self, cfg, image_paths, transforms=None):
        self.cfg = cfg
        self.image_paths = image_paths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_fp = self.image_paths[idx]
        image = Image.open(image_fp)
        label = get_label(image_fp)

        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, label
