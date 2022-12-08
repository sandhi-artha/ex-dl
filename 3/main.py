import torch
from torch.utils.data import DataLoader

from cfg import cfg as dict_cfg
from viz import viz_ds_sample, viz_dl_batch
from dataset import MyDataset
from model import MyModel
from task import MyTask


class Dict2Obj(object):
    """Turns a dictionary into a class"""
    def __init__(self, dictionary):
        for key in dictionary: setattr(self, key, dictionary[key])

def run():
    cfg = Dict2Obj(dict_cfg)

    print('loading data..')
    train_ds = MyDataset(data_dir=cfg.ds_path, mode='train')
    test_ds = MyDataset(data_dir=cfg.ds_path, mode='test')

    viz_ds_sample(train_ds)

    train_dl = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=cfg.bs, shuffle=True)

    # viz_dl_batch(train_dl)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    print('creating model..')
    model = MyModel(cfg=cfg)
    print(model)

    mnist_task = MyTask(cfg=cfg, model=model, train_dl=train_dl, test_dl=test_dl)
    mnist_task.train(cfg.epochs)
    

if __name__=='__main__':
    run()