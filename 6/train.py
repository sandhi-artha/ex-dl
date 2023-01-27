from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader

from src.cfg import cfg
from src.dataset import CocoDS
from src.transforms import get_transform
from src.model import load_model
from src.viz import viz_pred
from src.evaluation import encode_pred, save_results
from src.task import FineTuneCoco


def main(cfg):
    # load dataset
    # label_fp = 'train.json'
    # val_label_fp = 'val.json'

    transforms = get_transform(cfg.image_resize)
    train_ds = CocoDS(cfg, cfg.label_fp, transforms)
    val_ds = CocoDS(cfg, cfg.val_label_fp, transforms)

    train_dl = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True,
        num_workers=2,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_dl = DataLoader(
        val_ds, batch_size=cfg.bs,
        num_workers=2,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model()

    ft_task = FineTuneCoco(cfg, model, train_dl, val_dl, device)
    ft_task.train(epochs=2)


if __name__=='__main__':
    main(cfg)