import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dims: int, input_dim=(1,28,28)):
        super(Encoder, self).__init__()
        in_features = input_dim[1] * input_dim[2]
        self.linear1 = nn.Linear(in_features, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class Decoder(nn.Module):
    def __init__(self, latent_dims: int, input_dim=(1,28,28)):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        out_features = input_dim[1] * input_dim[2]
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, out_features)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, *self.input_dim))

class LinearAE(nn.Module):
    def __init__(self, latent_dims: int, input_dim: tuple = (1,28,28)):
        super(LinearAE, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)