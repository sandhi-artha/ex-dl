

### DATASET ###
from typing import Any
from torch import Tensor
import torch
import torchvision
from torchvision.datasets import STL10
import torchvision.transforms as T
import matplotlib.pyplot as plt
import kornia.augmentation as A

def view_images(ds):
    NUM_IMAGES = 8
    img_stack = [img for idx in range(NUM_IMAGES) for img in ds[idx][0]]
    imgs = torch.stack(img_stack, dim=0)    # [2*N, 3, 96, 96]
    img_grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(10,5))
    plt.title('Augmented image examples of the STL10 dataset')
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()


class ContrastiveTransformations(object):
    """sample multiple different DA for a given image"""
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x: Tensor) -> [Tensor, Tensor]:
        return [self.base_transforms(x) for i in range(self.n_views)]


def get_dataset(transform=None):
    transform_list = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=96, antialias=True),
        T.RandomApply([
            T.ColorJitter(brightness=0.5,
                          contrast=0.5,
                          saturation=0.5,
                          hue=0.1)
        ], p=0.8),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    if transform is None:
        transform = ContrastiveTransformations(transform_list, n_views=2)

    unlabeled_data = STL10(root='../data/stl10-dl', split='unlabeled', download=True, transform=transform)
    train_data = STL10(root='../data/stl10-dl', split='train', download=True, transform=transform)

    # view_images(unlabeled_data)
    print(f'unlabeled data: {len(unlabeled_data)}')
    print(f'train data: {len(train_data)}')
    return unlabeled_data, train_data

class ContrastiveDA(torch.nn.Module):
    def __init__(self, n_views=2):
        super().__init__()
        self.n_views = n_views
        self.transforms = torch.nn.Sequential(
            A.RandomHorizontalFlip(),
            A.RandomResizedCrop(size=(96,96)),
            A.ColorJitter(brightness=0.5,
                          contrast=0.5,
                          saturation=0.5,
                          hue=0.1, p=0.8),
            A.Normalize((0.5,), (0.5))
        )

    @torch.no_grad()    # disable gradients for efficiency
    def forward(self, x: Tensor) -> Tensor:
        return [self.transforms(x) for i in range(self.n_views)]
    

### MODULE ###
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from time import time

class SimCLR(pl.LightningModule):
    def __init__(self, cfg_module: dict, cfg, len_train_dl: int, transforms):
        super().__init__()
        # convert class attributes to key, value pairs
        hparams = {k:v for k,v in cfg.__dict__.items() if not k.startswith('__')}
        self.save_hyperparameters(hparams)

        self.cfg = cfg_module
        self.load_model(cfg_module['hidden_dim'])

        self.transforms = transforms

        self.t_batches = []
        self.len_train_dl = len_train_dl
        self.t_epoch = 0.0

    def load_model(self, hidden_dim: int):
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(num_classes=4*hidden_dim)  # Output of last linear layer
        
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.cfg['lr'],
                                weight_decay=self.cfg['weight_decay'])
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.cfg['max_epochs'],
                                                            eta_min=self.cfg['lr']/50)
        return [optimizer], [lr_scheduler]
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # just after batch.to(device)
        x, y = batch
        # if self.trainer.training:     # apply only during training
        x = self.transforms(x)  # => we perform GPU/Batched data augmentation
        return x, y

    def info_nce_loss(self, batch, mode='train'):
        imgs, _ = batch     # [[B,C,H,W], [B,C,H,W]]
        imgs = torch.cat(imgs, dim=0)   # [2B,C,H,W]

        # Encode all images
        feats = self.convnet(imgs)      # [2B, hidden_dim]

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1) # [2B, 2B], None just means add dim=1

        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device) # [2B, 2B] where diagonal is True, and rest is False
        cos_sim.masked_fill_(self_mask, -9e15)

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)   # [2B,2B] same as self_mask, but True in diagonal starts mid-way

        # InfoNCE loss
        cos_sim = cos_sim / self.cfg['temperature']
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode+'_loss', nll)

        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)    # [2B,2B+1]
        
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)  # [2B]
        
        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

        return nll
    
    def on_train_epoch_start(self):
        self.t_epoch = time()

    def training_step(self, batch: tuple, batch_idx: int):
        start = time()
        loss = self.info_nce_loss(batch, mode='train')
        self.t_batches.append(time() - start)
        return loss
    
    def on_train_epoch_end(self):
        print(f'average batch: {sum(self.t_batches) / self.len_train_dl} s')
        print(f'epoch time: {time() - self.t_epoch}')
        self.t_epoch = 0.0

    def validation_step(self, batch: tuple, batch_idx: int):
        self.info_nce_loss(batch, mode='val')


### CONFIG ###
class CFG:
    data_path = '../data/stl10-dl'
    out_dir = 'save'
    module = {
        'hidden_dim' : 128,
        'lr' : 5e-4,
        'weight_decay' : 1e-4,
        'max_epochs' : 20,
        'temperature' : 0.07,
    }
    batch_size = 256
    num_workers = 3
    trainer = {
        'deterministic': True,
        'profiler': None,           # 'simple', 'advanced'
        'max_epochs': 30,
        'precision': '16-mixed',    # '16-mixed', '32-true'
    }
    wandb = {
        'do': True,
        'wandb_args': {
            'name': '0_simclr',     # run name. SPECIFY THIS despite not using wandb
            'project': 'ex-dl-11',
            'entity': 's_wangiyana'
        }
    }



### TRAINER ###
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

def main(cfg: CFG):
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(42)

    # dataset
    pre_transform = T.ToTensor()
    unlabeled_ds, train_ds = get_dataset(pre_transform)

    train_dl = DataLoader(unlabeled_ds, batch_size=cfg.batch_size, shuffle=True,
                          drop_last=True, pin_memory=True, num_workers=cfg.num_workers,
                          persistent_workers=True)
    val_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False,
                        drop_last=False, pin_memory=True, num_workers=cfg.num_workers,
                        persistent_workers=True)

    # module
    con_transforms = ContrastiveDA(n_views=2)
    len_train_dl = len(train_dl)
    module = SimCLR(cfg.module, cfg, len_train_dl, con_transforms)

    # save output
    save_dir = Path(cfg.out_dir) / cfg.wandb['wandb_args']['name']
    save_dir.mkdir(parents=True, exist_ok=True)

    # logger
    if cfg.wandb['do']:
        cfg.wandb['wandb_args']['save_dir'] = save_dir
        logger = WandbLogger(**cfg.wandb['wandb_args'])
        hparams = {k:v for k,v in cfg.__dict__.items() if not k.startswith('__')}
        logger.log_hyperparams(hparams)
    else:
        logger = True   # use default TensorBoard

    # callback
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=save_dir,
            mode="min",
            filename=f'model-{{val_loss:.4f}}',
            save_top_k=1,
            save_last=True,
            verbose=1,
        )
    ]

    trainer = pl.Trainer(callbacks=callbacks, logger=logger, default_root_dir=save_dir,
                         **cfg.trainer)
    trainer.fit(module, train_dl, val_dl)


### EVALUATION ###
from collections import OrderedDict
import numpy as np
from sklearn.decomposition import PCA

def condition_state_dict(load_state_dict: dict, curr_state_dict: dict) -> dict:
    """saved ckpt from PL can have layer names: model.encoder.0.weight, ...
        but creating new model have layer names: encoder.0.weight, ...
        so keys are mismatched, see:
        https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    """
    cond_state_dict = OrderedDict()
    for k, v in load_state_dict.items():
        name = k[8:] # remove `convnet.`
        cond_state_dict[name] = v
    assert cond_state_dict.keys() == curr_state_dict.keys()
    return cond_state_dict

def labels_to_colors(arr: np.ndarray, num_classes: int) -> np.ndarray:
    c_arr = np.zeros(shape=(arr.shape[0], 3))
    cmap = plt.get_cmap('tab10')
    for c in range(num_classes):
        c_arr[np.argwhere(arr == c)] = cmap.colors[c]
    return c_arr

def save_pca(emb: Tensor, labels: Tensor, save_fp='pca.png') -> None:
    c_labels = labels_to_colors(labels.numpy(), num_classes=10)

    pca = PCA(n_components=2)
    pca.fit(emb)
    emb2d = pca.transform(emb)

    f, ax = plt.subplots(1,1, figsize=(6,6))
    ax.scatter(emb2d[:,0], emb2d[:,1], c=c_labels)
    ax.axis('equal')
    # ax.set(xlim=(-15,15), ylim=(-15,15))
    # ax.set_title(pca_stats(emb2d))
    f.savefig(save_fp)
    plt.close(f)

class EvalSimCLR(pl.LightningModule):
    def __init__(self, ckpt_path):
        super().__init__()
        checkpoint = torch.load(ckpt_path)
        self.load_model(checkpoint['state_dict'], hidden_dim=128)

    def load_model(self, state_dict, hidden_dim=128):
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(num_classes=4*hidden_dim)  # Output of last linear layer
        
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

        con_state_dict = condition_state_dict(state_dict, self.convnet.state_dict())
        self.convnet.load_state_dict(con_state_dict)
        self.convnet = self.convnet.cuda().eval()

    def predict(self, val_dl: DataLoader):
        print('predicting...')
        val_embs = []
        val_labels = []
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(val_dl):
                # Encode all images
                feats = self.convnet(imgs.cuda())      # [B, hidden_dim]
                val_embs.append(feats.detach().cpu())
                val_labels.append(labels)
                if i==4: break
        
        val_embs = torch.cat(val_embs, axis=0)
        val_labels = torch.cat(val_labels, axis=0)

        save_pca(val_embs, val_labels, 'first500.png')


def evaluate(cfg: CFG):
    ckpt_path = 'lightning_logs/version_0/checkpoints/epoch=4-step=1950.ckpt'
    module = EvalSimCLR(ckpt_path)

    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    unlabeled_data, train_data_contrast = get_dataset(transform)

    # train_dl = DataLoader(unlabeled_data, batch_size=cfg.batch_size, shuffle=True,
    #                           drop_last=True, pin_memory=True, num_workers=cfg.num_workers)
    val_dl = DataLoader(train_data_contrast, batch_size=100, shuffle=False,
                            drop_last=False, pin_memory=True, num_workers=cfg.num_workers)
    
    module.predict(val_dl)



if __name__=='__main__':
    main(CFG)
    # evaluate(CFG)