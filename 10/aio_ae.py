### DATASET ###
from pathlib import Path
import torchvision
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

import torchvision.transforms as T

class InvNormalize(T.Normalize):
    def __init__(self,mean,std,*args,**kwargs):
        new_mean = [-m/s for m,s in zip(mean,std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


def get_n_samples_per_class(ds, n=2) -> tuple[Tensor, Tensor]:
    """find n samples for each class and create a torch array"""
    num_classes = 10
    classes = list(range(num_classes))
    class_samples = [[] for _ in range(num_classes)]
    
    tot = 0
    for x, y in ds:             # loop through dataset
        for c in classes:       # loop through possible class index
            if y == c:          # if the label matches a class
                if len(class_samples[c]) < n:   # check if we already have n samples of that class
                    class_samples[c].append((x,y))
                    tot += 1
                break           # to reduce comp, if it matches an unfilled class, move to next sample
        
        if tot > n*num_classes: # check if all class samples are full
            break
    
    # check if we have n samples for each class
    for cls in class_samples:
        assert len(cls) == n
    
    # reformat to batch of images and batch of labels
    images = []
    labels = []
    for cls in class_samples:
        for x,y in cls:
            images.append(x)
            labels.append(y)
    
    # stack them by creating a batch dimension
    images, labels = torch.stack(images, dim=0), torch.stack(labels, dim=0)
    assert images.shape[0] == n*num_classes
    assert labels.shape[0] == n*num_classes

    return images, labels

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
            nn.Conv2d(channels, 64, 3, stride=2, padding=1),    # size/2
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),         # size/2
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 7),                             # size-6
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

    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input)
    
    def decode(self, input: Tensor) -> Tensor:
        return self.decoder(input)
    
    def forward(self, input: Tensor) -> Tensor:
        z = self.encode(input)
        x_hat = self.decode(z)
        return x_hat

    def loss_function(self, *inputs) -> Tensor:
        return self.loss_fn(*inputs)


class LinearAE(SimpleAE):
    def __init__(self, input_shape, latent_dim=128):
        # override basic functionalities of SimpleAE
        super().__init__(input_shape=input_shape)

        self.ch = input_shape[0]

        self.w = ((input_shape[1] // 2) // 2) -6
        lin_size = self.w * self.w * 256
        self.lin_enc = nn.Linear(in_features=lin_size, out_features=latent_dim)
        self.lin_dec = nn.Linear(in_features=latent_dim, out_features=lin_size)

    def encode(self, input: Tensor) -> Tensor:
        x = self.encoder(input)
        # flatten to 1D along batch size, out_shape: (B,1)
        x = x.view(x.shape[0], -1)
        return self.lin_enc(x)
    
    def decode(self, input: Tensor) -> Tensor:
        x = self.lin_dec(input)
        # expand back to 4D, out_shape: (B, C, H, W)
        x = x.view(x.shape[0], 256, self.w, self.w)
        return self.decoder(x)
    


### TRAINER ###
from time import time
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

class Trainer():
    def __init__(self, cfg: dict, model: SimpleAE, train_dl, val_dl, device, viz_data: dict):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.viz_data = viz_data

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
        self.save_dir = Path(cfg['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def configure_optimizers(self, lr: float):
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

    def print_log(self, epoch: int) -> None:
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

    def one_epoch(self, n: int) -> None:

        self.on_epoch_train()

        self.metrics['epoch'].append(n)
        self.t_total += self.metrics['t_train'][n]
        self.metrics['t_total'].append(self.t_total)
        self.print_log(n)

    def fit(self) -> dict:
        self.t_total = 0.0
        for epoch in range(self.cfg['epochs']):
            self.one_epoch(epoch)
            self.viz_reconstruction(self.cfg['n_viz'], **self.viz_data['train'], fn=f'train_ep{epoch}')
            self.viz_reconstruction(self.cfg['n_viz'], **self.viz_data['test'], fn=f'test_ep{epoch}')
        return self.metrics
    
    def viz_reconstruction(self, n: int, images: Tensor, labels: Tensor, inv_transform, fn: str) -> None:
        num_classes = 10

        f, ax = plt.subplots(num_classes, n*2, figsize=(n*2*2, num_classes*2))
        
        self.model.eval()
        with torch.no_grad():
            emb = self.model.encode(images.to(self.device))
            recs = self.model.decode(emb)
            emb = emb.detach().cpu()
            recs = recs.detach().cpu()

        c_labels = labels_to_colors(labels.numpy(), 10)
        save_pca(emb, c_labels, self.save_dir / f'pca_{fn}.png')

        i = 0
        for r in range(num_classes):
            for c in range(0, n*2, 2):
                image = inv_transform(images[i])
                stats = get_image_stats(image)
                ax[r, c].imshow(image.permute(1,2,0))
                ax[r, c].set_title(f'{labels[i].item()}\n{stats}')
                ax[r, c].axis('off')

                rec = inv_transform(recs[i])
                stats = get_image_stats(rec)
                rec_loss = torch.sqrt(self.model.loss_function(image, rec))
                ax[r, c+1].imshow(clip_image(rec.permute(1,2,0)))
                ax[r, c+1].set_title(f'rmse: {rec_loss.item():.4f}\n{stats}')
                ax[r, c+1].axis('off')
                i += 1

        plt.tight_layout()
        f.savefig(self.save_dir / f'{fn}.png')
        plt.close(f)

def labels_to_colors(arr: np.ndarray, num_classes: int) -> np.ndarray:
    c_arr = np.zeros(shape=(arr.shape[0], 3))
    cmap = plt.get_cmap('tab10')
    for c in range(num_classes):
        c_arr[np.argwhere(arr == c)] = cmap.colors[c]
    return c_arr

def pca_stats(arr_red: np.ndarray) -> str:
    """ xmax | xmin | xmean | xstd
        ymax | ymin | ymean | ystd
    """
    return '{:.3f} | {:.3f} | {:.3f} | {:.3f}\n{:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
        np.max(arr_red[:, 0]), np.min(arr_red[:, 0]),
        np.mean(arr_red[:, 0]), np.std(arr_red[:, 0]),
        np.max(arr_red[:, 1]), np.min(arr_red[:, 1]),
        np.mean(arr_red[:, 1]), np.std(arr_red[:, 1]),
    )

def save_pca(emb: Tensor, c_labels: np.ndarray, save_fp) -> None:
    pca = PCA(n_components=2)
    pca.fit(emb)
    emb2d = pca.transform(emb)

    f, ax = plt.subplots(1,1, figsize=(6,6))
    ax.scatter(emb2d[:,0], emb2d[:,1], c=c_labels)
    ax.axis('equal')
    ax.set(xlim=(-15,15), ylim=(-15,15))
    ax.set_title(pca_stats(emb2d))
    f.savefig(save_fp)
    plt.close(f)

def get_image_stats(image: Tensor) -> str:
    """returns:
        max  | min
        mean | std
    """
    return '{:.3f} | {:.3f} \n{:.3f} | {:.3f}'.format(
        image.max().item(),
        image.min().item(),
        image.mean().item(),
        image.std().item(),
    )

def clip_image(image: Tensor) -> Tensor:
    """clips to 0 and 1 to remove matplotlib warnings"""
    return torch.clip(image, min=0.0, max=1.0)

def seed_torch(seed=42) -> None:
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
    'n_viz': 2,
    'save_dir': 'save/ae6',
}



### MAIN ###

from torch.utils.data import DataLoader

def main(cfg: CFG):
    seed_torch(42)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = {'train': T.Normalize(mean, std), 'test': T.Normalize(mean, std)}
    denormalize = InvNormalize(mean, std)

    cacher = Cacher(cfg['data_path'], transforms=normalize)
    train_ds, test_ds = cacher.get_ds()
    print(f'train: {len(train_ds)} test: {len(test_ds)}')

    viz_data = {'train': {}, 'test': {}}
    viz_data['train']['images'], viz_data['train']['labels'] = get_n_samples_per_class(train_ds, cfg['n_viz'])
    viz_data['train']['inv_transform'] = denormalize
    viz_data['test']['images'], viz_data['test']['labels'] = get_n_samples_per_class(test_ds, cfg['n_viz'])
    viz_data['test']['inv_transform'] = denormalize

    train_dl = DataLoader(train_ds, batch_size=cfg['train_bs'], shuffle=True, num_workers=cfg["workers"])
    test_dl = DataLoader(test_ds, batch_size=cfg["test_bs"], num_workers=cfg["workers"])

    # model = SimpleAE(input_shape=(3,32,32))
    model = LinearAE(input_shape=(3,32,32), latent_dim=128)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('training using:', device)
    trainer = Trainer(cfg, model, train_dl, test_dl, device, viz_data)
    metrics = trainer.fit()

    # visualize 2 samples from each class in train and test set
    # trainer.viz_reconstruction(cfg['n_viz'], **viz_data['test'], fn='test')

if __name__=='__main__':
    main(CFG)