import torch
from torch.nn import Module, Conv2d, ReLU, MaxPool2d, Linear, LogSoftmax


class LeNet(Module):
    # https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
    def __init__(self, in_channels: int, num_classes: int):
        # call the parent constructor
        super(LeNet, self).__init__() # 32
        
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=20, kernel_size=(5, 5)) # 28
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # 14
        
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)) # 10
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # 5,5,50
        
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=50*5*5, out_features=500)
        self.relu3 = ReLU()
        
        # initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=num_classes)
        # self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass through first block
        x = self.pool1(self.relu1(self.conv1(x)))
        # then 2nd block
        x = self.pool2(self.relu2(self.conv2(x)))
        # flatten all dimension except batch
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x