from torchvision import models
import torch
from torch import nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # init layers
        # input is [B,3,32,32]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)  # [B,6,28,28]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # [B,6,14,14]
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  # [B,16,10,10]
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)  # after 2nd pool will be [B,16,5,5]
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_densenet(num_classes=10, train_only_task_head=True, use_pretrained=True):
    # # for new torchvision >0.13, use this
    # if use_pretrained:
    #     weights = models.DenseNet121_Weights.DEFAULT
    # else:
    #     weights = None
    # model = models.densenet121(weights=weights)
    
    # for old torchvision ver
    model = models.densenet121(pretrained=use_pretrained)

    if train_only_task_head:
        for param in model.parameters():    # freeze other layers
            param.requires_grad=False

    # replace the task head
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)  # replace the classifier
    return model


def load_model(cfg):
    if cfg.model=='cnn':
        model = SimpleNet(cfg.num_classes)
    elif cfg.model=='densenet':
        model = load_densenet(cfg.num_classes, cfg.train_only_task_head, cfg.use_pretrained)
    else:
        raise ValueError(f'unrecognized model string: {cfg.model}. use "cnn" or "densenet"')
    
    return model