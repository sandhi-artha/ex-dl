import random
import torchvision.transforms as T


def get_transform(train=True):
    """
        append transformation to image and label using Compose
    """
    transforms = [T.ToTensor()]
    return T.Compose(transforms)