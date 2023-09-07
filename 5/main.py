import os

from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from pytorch_lightning import loggers

from src.cfg import cfg
from src.model import LinearAE
from src.module import MyModule
from src.viz_callback import VisualisationCallback

def assert_ds(ds):
    image, label = ds[0]
    assert image.shape == (1,28,28)
    assert image.dtype == torch.float32
    assert type(label) == int

def plot_latent(module: MyModule, dl: DataLoader, max_batches=4):
    import matplotlib.pyplot as plt
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime('%Y%m%dT%H%M%S')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = module.model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            z = model.encoder(x.to(device))
            z = z.to('cpu').detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
            if i > max_batches:
                print(f'saving as {date_time}.png')
                plt.colorbar()
                plt.savefig(f'save/{date_time}.png')
                break

def init_logger(cfg):
    save_dir = cfg['wandb']['params']['save_dir']
    if not os.path.isdir(save_dir):   # fix the wandb save_dir not writeable
        os.makedirs(save_dir)

    if cfg['wandb']['is']:
        logger = loggers.WandbLogger(**cfg['wandb']['params'])
        logger.log_hyperparams(cfg)
    else:
        logger = True

    return logger, save_dir

def main(cfg):
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    train_ds = torchvision.datasets.MNIST(root='../data/mnist-dl', train=True, download=True, transform=ToTensor())
    test_ds = torchvision.datasets.MNIST(root='../data/mnist-dl', train=False, download=True, transform=ToTensor())
    assert_ds(train_ds)    
    print(f'train: {len(train_ds)} test: {len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=cfg['train_bs'], shuffle=True, num_workers=cfg["workers"])
    test_dl = DataLoader(test_ds, batch_size=cfg["test_bs"], num_workers=cfg["workers"])

    model = LinearAE(latent_dims=2, input_dim=(1,28,28))
    module = MyModule(cfg=cfg, model=model)

    logger, save_dir = init_logger(cfg)

    callbacks = [
        VisualisationCallback(log_dir=save_dir, max_batches=4)
    ]

    trainer = pl.Trainer(**cfg['trainer'], logger=logger, callbacks=callbacks)
    trainer.fit(module, train_dl)
    # plot_latent(module, train_dl)
    
if __name__ == '__main__':
    main(cfg)