import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)

if __name__=='__main__':
    b_img = torch.zeros((16,3,32,32))
    print(b_img.shape)

    conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
    conv_pad = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
    bn = nn.BatchNorm2d(3)
    net = Model(conv_pad)
    print(net(b_img).shape)
