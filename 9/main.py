import os
from glob import glob

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.cfg import cfg
from src.dataset import CifarDS
from src.model import LeNet
from src.module import MyModule


def main(cfg):
    pl.seed_everything(42, workers=True)
    train_df = pd.read_csv(os.path.join(cfg['data_path'], 'train.csv'))
    test_df = pd.read_csv(os.path.join(cfg['data_path'], 'test.csv'))

    print(f'train: {train_df.shape[0]} test: {test_df.shape[0]}')

    train_ds = CifarDS(cfg, train_df)
    test_ds = CifarDS(cfg, test_df)

    train_dl = DataLoader(train_ds, batch_size=cfg['train_bs'], shuffle=True, num_workers=cfg["workers"])
    test_dl = DataLoader(test_ds, batch_size=cfg["test_bs"], num_workers=cfg["workers"])
    # import pdb; pdb.set_trace()

    model = LeNet(in_channels=3, num_classes=10)
    module = MyModule(cfg, model)
    trainer = pl.Trainer(
        deterministic=True,
        gpus=1,
        logger=True,
        profiler='simple',
        max_epochs= 2,
        precision=16,
    )

    trainer.fit(module, train_dataloader=train_dl, val_dataloaders=test_dl)

if __name__ == '__main__':
    main(cfg)
