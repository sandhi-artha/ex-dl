### DATASET ###
from pathlib import Path
import torchvision
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import torchvision.transforms as T

class CacheCifarDS(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.transforms = transforms
        self.images, self.labels = images, labels

    def __getitem__(self, idx):
        if self.transforms is None:
            return self.images[idx], self.labels[idx]
        else:
            return self.transforms(self.images[idx]), self.labels[idx]

    def __len__(self):
        return len(self.images)

class Cacher:
    def __init__(self, root='./data', transforms=None):
        self.root = Path(root)
        if transforms is None:
            self.transforms = {'train': None, 'test': None}
        else:
            self.transforms = transforms
        
    def get_ds(self):
        """check if array exist, if not, download and create a cache"""
        init_transform = torchvision.transforms.ToTensor()
        if self.cache_exist('train'):
            train_ds = self.load_cache(mode='train')
        else:
            train_ds = torchvision.datasets.CIFAR10(
                root=self.root, train=True, download=True, transform=init_transform)
            train_ds = self.cache_ds(train_ds, mode='train')
        
        if self.cache_exist('test'):
            test_ds = self.load_cache(mode='test')
        else:
            test_ds = torchvision.datasets.CIFAR10(
                root=self.root, train=False, download=True, transform=init_transform)
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
        cache_ds = CacheCifarDS(images, labels, self.transforms[mode])
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


### MODEL ###

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class SimpleAE(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        channels = input_shape[0]
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 7),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 7),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,
                               output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, channels, 3, stride=2, padding=1,
                               output_padding=1),
        )

        self.loss_fn = nn.MSELoss()

    
    def forward(self, input: Tensor) -> Tensor:
        z = self.encoder(input)
        x_hat = self.decoder(z)
        return x_hat

    def loss_function(self, *inputs) -> Tensor:
        return self.loss_fn(*inputs)


### TRAINER ###
from time import time

class Trainer():
    def __init__(self, cfg: dict, model: SimpleAE, train_dl, val_dl, device):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device

        # optimizer
        self.configure_optimizers(cfg['lr'])

        # logging
        self.train_ds_len = len(train_dl.dataset)
        self.metrics = {
            'epoch': [],
            'lr': [],
            't_train': [],
            'rec_los': [],
            't_total': [],
        }

    def configure_optimizers(self, lr):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)
        
    def on_epoch_train(self):
        run_loss = 0.0

        t_train = time()
        self.model.train()
        for i, (x, _) in enumerate(self.train_dl):
            x = x.to(self.device)
            x_hat = self.model(x)
            loss = self.model.loss_function(x, x_hat)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            run_loss += loss.item() * x.size(0) # will be divided by N later

        self.metrics['t_train'].append(time() - t_train)
        self.metrics['lr'].append(self.optimizer.param_groups[0]['lr'])
        self.metrics['rec_los'].append(run_loss / self.train_ds_len)

    def print_log(self, epoch):
        if epoch == 0:
            for key in self.metrics.keys():
                print(key, end='\t')    # print headers
            print('')

        for key, value in self.metrics.items():
            if key=='epoch':
                print(f'{value[epoch]:4d}', end='\t')
            elif key=='t_total':
                print(f'{value[epoch]:8.2f}', end='\t')
            else:
                print(f'{value[epoch]:.4f}', end='\t')
        print('')

    def one_epoch(self, n):

        self.on_epoch_train()

        self.metrics['epoch'].append(n)
        self.t_total += self.metrics['t_train'][n]
        self.metrics['t_total'].append(self.t_total)
        self.print_log(n)

    def fit(self):
        self.t_total = 0.0
        for epoch in range(self.cfg['epochs']):
            self.one_epoch(epoch)
        return self.metrics
    

def seed_torch(seed=42):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### CFG ###
CFG = {
    'data_path': '../data/cifar10-dl',
    'lr' : 1e-3,
    'max_lr': 0.1,
    'workers': 0,
    'train_bs': 128,
    'test_bs': 128,
    'epochs' : 10,
}



### MAIN ###

from torch.utils.data import DataLoader

def main(cfg: CFG):
    seed_torch(42)

    cacher = Cacher(cfg['data_path'], transforms=None)
    train_ds, test_ds = cacher.get_ds()
    print(f'train: {len(train_ds)} test: {len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=cfg['train_bs'], shuffle=True, num_workers=cfg["workers"])
    test_dl = DataLoader(test_ds, batch_size=cfg["test_bs"], num_workers=cfg["workers"])

    model = SimpleAE(input_shape=(3,32,32))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('training using:', device)
    trainer = Trainer(cfg, model, train_dl, test_dl, device)
    metrics = trainer.fit()

if __name__=='__main__':
    main(CFG)