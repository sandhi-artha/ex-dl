import os
from glob import glob

import torch
from torch.utils.data import DataLoader
from src.cfg import cfg
from src.dataset import CifarDS
from src.transforms import get_transform
from src.model import load_model
from src.task import FineTuneCifar



def main(cfg):
    # load dataset
    train_fps = glob(os.path.join(cfg.data_dir, 'train', '*', '*.png'))
    train_ds = CifarDS(cfg, train_fps, get_transform())
    image, label = train_ds[0]
    print(image.dtype, image.shape)
    print(label.dtype, label.shape)

    test_fps = glob(os.path.join(cfg.data_dir, 'test', '*', '*.png'))
    test_ds = CifarDS(cfg, test_fps, get_transform())
    print(f'Samples, train: {len(train_ds)}, test: {len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=cfg.bs)

    # load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(cfg)

    ft_task = FineTuneCifar(cfg, model, train_dl, test_dl, device)
    ft_task.train(epochs=cfg.epochs)

if __name__=='__main__':
    main(cfg)