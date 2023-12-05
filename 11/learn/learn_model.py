import torchvision
import torch
import torch.nn as nn



dummy = torch.rand((8,3,96,96))

hidden_dim = 128
convnet = torchvision.models.resnet18(num_classes=4*hidden_dim)  # Output of last linear layer
        
# The MLP for g(.) consists of Linear->ReLU->Linear
convnet.fc = nn.Sequential(
    convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
    nn.ReLU(inplace=True),
    nn.Linear(4*hidden_dim, hidden_dim)
)

out = convnet(dummy)
print(out.shape)

convnet.fc = nn.Identity()  # Removing projection head g(.)
out = convnet(dummy)
print(out.shape)

# print(convnet)