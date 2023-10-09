import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from cacher import Cacher
from time import time

class CFG:
    data_path = '../../data/cifar10-dl'
    bs = 128
    epochs = 15
    lr = 1e-3
    momentum = 0.9
    max_lr = 0.1

def main(cfg: CFG):
    cacher = Cacher(cfg.data_path)
    train_ds, _ = cacher.get_ds()
    train_dl = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, num_workers=1)
    print(f'epochs: {cfg.epochs}, batches: {len(train_dl)}')

    model = torch.nn.Linear(2,1)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.max_lr,
        steps_per_epoch=len(train_dl),
        epochs=cfg.epochs,
        anneal_strategy='linear'
    )

    lr_steps = []
    for epoch in range(cfg.epochs):
        t_ep_start = time()
        print(f'epoch: {epoch}/{cfg.epochs}', end=' ')
        t_opt = 0.0
        t_sch = 0.0
        t_load = 0.0
        t_load_start = time()
        for batch_idx, (x, y) in enumerate(train_dl):
            t_load += time() - t_load_start
            t_opt += timer(optimizer.step)
            t_sch += timer(scheduler.step)
            lr_steps.append(scheduler.get_last_lr()[0])
            t_load_start = time()

        t_ep = time() - t_ep_start
        print('t_ep: ', t_ep, 't_load: ', t_load, 't_opt: ', t_opt, 't_sch: ', t_sch)

    f, ax = plt.subplots(1,1)
    ax.plot(range(cfg.epochs*len(train_dl)), lr_steps)
    plt.show()


def timer(func):
    t_start = time()
    func()
    return time() - t_start

if __name__=='__main__':
    main(CFG)