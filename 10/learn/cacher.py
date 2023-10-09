from pathlib import Path

import torchvision
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CacheCifarDS(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)


class Cacher:
    def __init__(self, root='./data'):
        self.root = Path(root)
        self.transform = torchvision.transforms.ToTensor()
        
    def get_ds(self):
        """check if array exist, if not, download and create a cache"""

        if self.cache_exist('train'):
            train_ds = self.load_cache(mode='train')
        else:
            train_ds = torchvision.datasets.CIFAR10(
                root=root, train=True, download=True, transform=self.transform)
            train_ds = self.cache_ds(train_ds, mode='train')
        
        if self.cache_exist('test'):
            test_ds = self.load_cache(mode='test')
        else:
            test_ds = torchvision.datasets.CIFAR10(
                root=root, train=False, download=True, transform=self.transform)
            test_ds = self.cache_ds(test_ds, mode='test')
        
        return train_ds, test_ds
    
    def cache_exist(self, mode='train'):
        # check if cache exist in path
        image_fp = self.root / f'{mode}_images.pt'
        label_fp = self.root / f'{mode}_labels.pt'
        return image_fp.exists() and label_fp.exists()

    def load_cache(self, mode='train'):
        print('loading saved cache')
        images = torch.load(self.root / f'{mode}_images.pt')
        labels = torch.load(self.root / f'{mode}_labels.pt')
        cache_ds = CacheCifarDS(images, labels)
        return cache_ds

    def cache_ds(self, ds, mode='train'):
        print(f'caching {mode}')
        images = torch.zeros((len(ds),3,32,32), dtype=torch.float32)
        labels = torch.zeros((len(ds)), dtype=torch.int64)
        for i, (image, label) in tqdm(enumerate(ds)):
            images[i,:,:,:] = image
            labels[i] = label
        
        torch.save(images, self.root / f'{mode}_images.pt')
        torch.save(labels, self.root / f'{mode}_labels.pt')
        
        # create the cache dataset
        cache_ds = CacheCifarDS(images, labels)
        return cache_ds


if __name__=='__main__':
    root = '../../data/cifar10-dl'
    cacher = Cacher(root)
    train_ds, test_ds = cacher.get_ds()