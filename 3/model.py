from torch import nn

class MyModel(nn.Module):
    """requires __init__ and __forward__"""
    def __init__(self, cfg):
        """input is image of shape (bs,28,28,1)"""
        super(MyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.MLP = nn.Sequential(
            nn.Linear(cfg.image_dim[0]*cfg.image_dim[1], 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(512, cfg.n_class)
        )

    def forward(self, x):
        """a forward step of a batch x"""
        x = self.flatten(x)
        logits = self.MLP(x)
        return logits