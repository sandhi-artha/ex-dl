import os
from glob import glob
from time import time

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.transforms import ToTensor

from src.cfg import cfg
from src.dataset import CifarDS, CacheCifarDS
from src.model import LeNet
from src.module import MyModule


def main(cfg):
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    if cfg['download_ds']:
        train_ds = torchvision.datasets.CIFAR10(root='../data/cifar10-dl',
                                                train=True,
                                                download=True,
                                                transform=ToTensor())
        test_ds = torchvision.datasets.CIFAR10(root='../data/cifar10-dl',
                                               train=False,
                                               download=True,
                                               transform=ToTensor())
        print(f'train: {len(train_ds)} test: {len(test_ds)}')
    else:
        train_df = pd.read_csv(os.path.join(cfg['data_path'], 'train.csv'))
        test_df = pd.read_csv(os.path.join(cfg['data_path'], 'test.csv'))

        print(f'train: {train_df.shape[0]} test: {test_df.shape[0]}')
        if cfg['cache_ds']:
            start = time()
            train_ds = CacheCifarDS(cfg, train_df)
            test_ds = CacheCifarDS(cfg, test_df)
            cache_time = time() - start
            print(f'cache time: {cache_time:.2f}s')
        else:
            train_ds = CifarDS(cfg, train_df)
            test_ds = CifarDS(cfg, test_df)
    
    train_dl = DataLoader(train_ds, batch_size=cfg['train_bs'], shuffle=True, num_workers=cfg["workers"])
    test_dl = DataLoader(test_ds, batch_size=cfg["test_bs"], num_workers=cfg["workers"])
    # import pdb; pdb.set_trace()

    if cfg['is_wandb']:
        if not os.path.isdir(cfg['log_dir']):   # fix the wandb save_dir not writeable
            os.makedirs(cfg['log_dir'])
        logger = loggers.WandbLogger(save_dir=cfg['log_dir'], name=cfg['name'],
                                     project=cfg['project'], entity=cfg['entity'])
    else:
        logger = True

    model = LeNet(in_channels=3, num_classes=10)
    module = MyModule(cfg, model)
    trainer = pl.Trainer(
        logger=logger,
        **cfg['trainer']
    )

    start = time()
    trainer.fit(module, train_dl, test_dl)
    elapsed = (time() - start)
    with open('time.log', 'a') as f:
        f.write(f"{cfg['comment']}\t{elapsed:.2f}\n")

if __name__ == '__main__':
    main(cfg)
