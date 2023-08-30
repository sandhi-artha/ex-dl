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