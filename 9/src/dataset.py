import torch
import torchvision
import pandas as pd


class CifarDS(torch.utils.data.Dataset):
    def __init__(self, cfg: dict, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = torchvision.io.read_image(row['path'])
        image = image/255.0
        label = torch.tensor(row['label'])
        return image.float(), label

    def __len__(self):
        return len(self.df)

class CacheCifarDS(torch.utils.data.Dataset):
    def __init__(self, cfg: dict, df: pd.DataFrame):
        self.cfg = cfg
        self.images = torch.zeros((df.shape[0],3,32,32), dtype=torch.float32)
        self.labels = torch.zeros((df.shape[0]), dtype=torch.int64)

        for i in range(df.shape[0]):
            row = df.iloc[i]
            self.images[i,:,:,:] = torchvision.io.read_image(row['path'])/255.0
            self.labels[i] = row['label']

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)