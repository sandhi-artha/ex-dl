# Dataset

When you want to check if you have transformations or not, do:
```python
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
```

instead of general `if` checking, which takes .2 seconds longer each epoch
```python
    def __getitem__(self, idx):
        if self.transforms:
            return self.transforms(self.images[idx]), self.labels[idx]
        else:
            return self.images[idx], self.labels[idx]
```