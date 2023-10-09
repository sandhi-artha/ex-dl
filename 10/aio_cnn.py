### DATASET ###
from pathlib import Path
import torchvision
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CacheCifarDS(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)


class Cacher:
    def __init__(self, root='./data'):
        self.root = Path(root)
        self.transform = torchvision.transforms.ToTensor()
        
    def get_ds(self):
        """check if array exist, if not, download and create a cache"""

        if self.cache_exist('train'):
            train_ds = self.load_cache(mode='train')
        else:
            train_ds = torchvision.datasets.CIFAR10(
                root=self.root, train=True, download=True, transform=self.transform)
            train_ds = self.cache_ds(train_ds, mode='train')
        
        if self.cache_exist('test'):
            test_ds = self.load_cache(mode='test')
        else:
            test_ds = torchvision.datasets.CIFAR10(
                root=self.root, train=False, download=True, transform=self.transform)
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
        cache_ds = CacheCifarDS(images, labels)
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

class ExampleCNN(nn.Module):
    # simple CNN from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    def __init__(self):
        super().__init__()
        # initialize parameters of each layers
        # since ReLU does not have parameters, we'll use it's functional version directly in forward function
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.loss_fn = nn.CrossEntropyLoss()

    def loss_function(self, *inputs) -> Tensor:
        return self.loss_fn(*inputs)

    def forward(self, x) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader
import copy
from time import time

### TRAINER ###
class Trainer():
    def __init__(self, cfg: dict, model: ExampleCNN, train_dl: DataLoader, val_dl: DataLoader, device):
        super().__init__()
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device

        # optimizer
        self.configure_optimizers(cfg['lr'])

        # other metrics
        self.acc_fn = MulticlassAccuracy(num_classes=10).to(device)

        # logging
        self.train_ds_len = len(train_dl.dataset)
        self.val_ds_len = len(val_dl.dataset)
        self.metrics = {
            'epoch': [],
            'lr': [],
            't_train': [],
            'loss' : [],
            'acc': [],
            't_val': [],
            'val_los': [],  # bcz >7 characters it will push the header 1 tab away
            'val_acc': [],
            't_total': []
        }

        # callback
        self.best_val_acc = 0.0
        self.best_model = copy.deepcopy(model.state_dict())

    def configure_optimizers(self, lr=1e-3):
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=lr)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=0.1, steps_per_epoch=len(self.train_dl),
            epochs=self.cfg['epochs'], pct_start=0.15, anneal_strategy='linear')
    
    def on_epoch_train(self):
        run_loss = 0.0
        run_acc = 0.0

        t_train = time()
        self.model.train()
        for i, (images, labels) in enumerate(self.train_dl):
            images = images.to(self.device)     # B,C,H,W
            labels = labels.to(self.device)     # B,1
            logits = self.model(images)         # B,10
            loss = self.model.loss_function(logits, labels)
            acc = self.acc_fn(logits, labels)   # preds, target -> averaged over classes
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()                       # schedule step on batch
            
            run_acc += acc.item() * images.size(0)      # avg_metric per batch * B
            run_loss += loss.item() * images.size(0)    # then divide by N later

        self.metrics['t_train'].append(time()-t_train)
        self.metrics['lr'].append(self.scheduler.get_last_lr()[0])  # returns a list with single element?
        self.metrics['loss'].append(run_loss / self.train_ds_len)
        self.metrics['acc'].append(run_acc / self.train_ds_len)

    def on_epoch_val(self):
        run_loss = 0.0
        run_acc = 0.0

        t_val = time()
        with torch.no_grad():
            self.model.eval()
            for i, (images, labels) in enumerate(self.val_dl):
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = self.model.loss_function(logits, labels)
                acc = self.acc_fn(logits, labels)

                run_acc += acc.item() * images.size(0)
                run_loss += loss.item() * images.size(0)

            val_acc = run_acc / self.train_ds_len

        self.metrics['t_val'].append(time()-t_val)
        self.metrics['val_los'].append(run_loss / self.train_ds_len)
        self.metrics['val_acc'].append(val_acc)
        
        self.model_checkpoint(val_acc)
    
    def model_checkpoint(self, val_acc):
        if val_acc < self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_model = copy.deepcopy(self.model.state_dict())
    
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
        self.metrics['epoch'].append(n)

        self.on_epoch_train()
        self.on_epoch_val()
        self.t_total += self.metrics['t_train'][n] + self.metrics['t_val'][n]
        self.metrics['t_total'].append(self.t_total)
        self.print_log(n)
            
    def fit(self):
        self.t_total = 0.0
        for epoch in range(self.cfg['epochs']):
            self.one_epoch(epoch)
        return self.metrics


### CFG ###
CFG = {
    'data_path': '../data/cifar10-dl',
    'lr' : 1e-3,
    'workers': 0,
    'train_bs': 128,
    'test_bs': 128,
    'epochs' : 20,
}


### MAIN ###

from torch.utils.data import DataLoader

def main(cfg: CFG):
    cacher = Cacher(cfg['data_path'])
    train_ds, test_ds = cacher.get_ds()
    print(f'train: {len(train_ds)} test: {len(test_ds)}')
    
    train_dl = DataLoader(train_ds, batch_size=cfg['train_bs'], shuffle=True, num_workers=cfg["workers"])
    test_dl = DataLoader(test_ds, batch_size=cfg["test_bs"], num_workers=cfg["workers"])

    model = ExampleCNN()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('training using:', device)
    trainer = Trainer(cfg, model, train_dl, test_dl, device)
    metrics = trainer.fit()



if __name__=='__main__':
    main(CFG)