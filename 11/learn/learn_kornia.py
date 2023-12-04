import torchvision.transforms as T
from torch import Tensor
import torch
import numpy as np
from PIL import Image

import kornia.augmentation as A

class KorniaDA(torch.nn.Module):
    """
        generic DA class, can take transforms as an input
            ToTensor is performed before batch transformation
            source: https://lightning.ai/docs/pytorch/2.1.0/notebooks/lightning_examples/augmentation_kornia.html
    """
    def __init__(self):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            A.RandomHorizontalFlip(),
            A.RandomResizedCrop(size=(96,96)),
            A.ColorJitter(brightness=0.5,
                          contrast=0.5,
                          saturation=0.5,
                          hue=0.1, p=0.8),
            A.Normalize((0.5,), (0.5))
        )

    @torch.no_grad()    # disable gradients for efficiency
    def forward(self, x: Tensor) -> Tensor:
        """output is [B,C,H,W]"""
        return self.transforms(x)


class ContrastiveDA(torch.nn.Module):
    def __init__(self, n_views=2):
        super().__init__()
        self.n_views = n_views
        self.transforms = torch.nn.Sequential(
            A.RandomHorizontalFlip(),
            A.RandomResizedCrop(size=(96,96)),
            A.ColorJitter(brightness=0.5,
                          contrast=0.5,
                          saturation=0.5,
                          hue=0.1, p=0.8),
            A.Normalize((0.5,), (0.5))
        )

    @torch.no_grad()    # disable gradients for efficiency
    def forward(self, x: Tensor) -> Tensor:
        return [self.transforms(x) for i in range(self.n_views)]


tv_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=96, antialias=True),
        T.RandomApply([
            T.ColorJitter(brightness=0.5,
                          contrast=0.5,
                          saturation=0.5,
                          hue=0.1)
        ], p=0.8),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
])


def main():
    # simulated PIL image
    im = np.random.randint(0, 255, size=(96,96,3), dtype=np.uint8)
    pil_im = Image.fromarray(im)

    # torchvision transformation
    tr_im = tv_transforms(pil_im)

    # kornia transformation
    ko_transforms = KorniaDA()
    con_ko_transforms = ContrastiveDA(n_views=2)

    # ToTensor() should be performed in the dataset and not during gpu batch transform
    tensor_transform = T.ToTensor()
    ko_tr_im = ko_transforms(tensor_transform(pil_im))
    list_ko_tr_im = con_ko_transforms(tensor_transform(pil_im))

if __name__=='__main__':
    main()