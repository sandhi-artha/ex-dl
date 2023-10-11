### UTILS ###
# ref: https://github.com/pytorch/examples/blob/2c57b0011a096aef83da3b5265a14db2f80cb124/imagenet/main.py#L363
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



### DATASET ###
from pathlib import Path
import torchvision
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import torchvision.transforms as T

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
    
class CNN2(nn.Module):
    # https://www.kaggle.com/code/faizanurrahmann/cifar-10-object-classification-best-model
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),   # 32,30,30
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout(p=0.2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  # 64,28,28
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # 64,14,14

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # 128,14,14
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # 128,7,7

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3), # 128,5,5
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=4, stride=4),                      # 128,1,1
            nn.Dropout(p=0.2),

            nn.Flatten(),
            nn.Linear(in_features=64, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=256, out_features=10)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def loss_function(self, *inputs) -> Tensor:
        return self.loss_fn(*inputs)

    def forward(self, x) -> Tensor:
        return self.layers(x)

class PT_CNN(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.net = torchvision.models.resnet18(weights=weights, num_classes=10)
        # modifies the network to accept different input size (originally for 224x224)
        #   replace the first conv layer
        #   replace the maxpool with Identity layer (just passes input to output) so no pooling is performed
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.net.maxpool = nn.Identity()
        self.loss_fn = nn.CrossEntropyLoss()

    def loss_function(self, *inputs) -> Tensor:
        return self.loss_fn(*inputs)

    def forward(self, x) -> Tensor:
        return self.net(x)

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
            't_data': [],   # note t_data is in ms
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
            epochs=self.cfg['epochs'], pct_start=0.3, anneal_strategy='linear')
    
    def on_epoch_train(self):
        run_loss = 0.0
        run_acc = 0.0

        t_data = AverageMeter('t_data', ':6.3f')
        t_train = time()
        end = time()
        self.model.train()
        for i, (images, labels) in enumerate(self.train_dl):
            t_data.update(time() - end)      # measure data loading time
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
            end = time()

        self.metrics['t_train'].append(time()-t_train)
        self.metrics['t_data'].append(t_data.avg*1000)
        self.metrics['lr'].append(self.optimizer.param_groups[0]['lr'])  # returns a list with single element?
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

        val_acc = run_acc / self.val_ds_len
        self.metrics['t_val'].append(time()-t_val)
        self.metrics['val_los'].append(run_loss / self.val_ds_len)
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


def seed_torch(seed=42):
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
    'workers': 0,
    'train_bs': 128,
    'test_bs': 128,
    'epochs' : 20,
}


### MAIN ###

from torch.utils.data import DataLoader

def main(cfg: CFG):
    seed_torch(42)

    transforms = {
        'train': T.Compose([
            # T.RandomCrop(size=32, padding=4),
            T.RandomHorizontalFlip(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'test': T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    }

    cacher = Cacher(cfg['data_path'], transforms=transforms)
    train_ds, test_ds = cacher.get_ds()
    print(f'train: {len(train_ds)} test: {len(test_ds)}')
    
    train_dl = DataLoader(train_ds, batch_size=cfg['train_bs'], shuffle=True, num_workers=cfg["workers"])
    test_dl = DataLoader(test_ds, batch_size=cfg["test_bs"], num_workers=cfg["workers"])

    # model = ExampleCNN()
    model = CNN2()
    # model = PT_CNN(pretrained=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('training using:', device)
    trainer = Trainer(cfg, model, train_dl, test_dl, device)
    metrics = trainer.fit()



if __name__=='__main__':
    main(CFG)