### DATASET ###
import torchvision

def download_ds(root='./data', transform=torchvision.transforms.ToTensor()):
    train_ds = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform)
    return train_ds, test_ds


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
        self.metrics = {'loss' : [], 'val_loss': [], 'acc': [], 'val_acc': []}

        # callback
        self.best_val_acc = 0.0
        self.best_model = copy.deepcopy(model.state_dict())

    def configure_optimizers(self, lr=1e-3):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)
    
    def on_epoch_train(self):
        self.model.train()
        run_loss = 0.0
        run_acc = 0.0
        for i, (images, labels) in enumerate(self.train_dl):
            images = images.to(self.device)     # B,C,H,W
            labels = labels.to(self.device)     # B,1
            logits = self.model(images)         # B,10
            loss = self.model.loss_function(logits, labels)
            acc = self.acc_fn(logits, labels)   # preds, target -> averaged over classes
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            run_acc += acc.item() * images.size(0)      # avg_metric per batch * B
            run_loss += loss.item() * images.size(0)    # then divide by N later

        self.metrics['loss'].append(run_loss / self.train_ds_len)
        self.metrics['acc'].append(run_acc / self.train_ds_len)

    def on_epoch_val(self):
        with torch.no_grad():
            self.model.eval()
            run_loss = 0.0
            run_acc = 0.0
            for i, (images, labels) in enumerate(self.val_dl):
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = self.model.loss_function(logits, labels)
                acc = self.acc_fn(logits, labels)

                run_acc += acc.item() * images.size(0)
                run_loss += loss.item() * images.size(0)

            val_acc = run_acc / self.train_ds_len
            self.metrics['val_loss'].append(run_loss / self.train_ds_len)
            self.metrics['val_acc'].append(val_acc)
        
        self.model_checkpoint(val_acc)
    
    def model_checkpoint(self, val_acc):
        if val_acc < self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_model = copy.deepcopy(self.model.state_dict())
    
    def print_log(self, epoch):
        for key, value in self.metrics.items():
            print(f'{key}: {value[epoch]:.4f}', end=' ')
        print('\n')

    def one_epoch(self, n):
        print(f'Epoch: {n}/{self.cfg["epochs"]}')
        self.on_epoch_train()
        self.on_epoch_val()
        self.print_log(n)
            
    def fit(self):
        for epoch in range(self.cfg['epochs']):
            self.one_epoch(epoch)
        return self.metrics


### CFG ###
CFG = {
    'data_path': '../data/cifar10-dl',
    'lr' : 1e-3,
    'workers': 1,
    'train_bs': 128,
    'test_bs': 128,
    'epochs' : 10,
}


### MAIN ###

from torch.utils.data import DataLoader

def main(cfg: CFG):
    train_ds, test_ds = download_ds(cfg['data_path'])
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