import random
import torchvision.transforms as T
import torchvision.transforms.functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Resize:
    def __init__(self, image_resize):
        self.image_resize = image_resize

    def __call__(self, image, target):
        assert len(image.shape)==3, f'image shape: {image.shape}'
        assert len(target['masks'].shape)==3, f'mask shape: {target["masks"].shape}'
        image = F.resize(image, self.image_resize, T.InterpolationMode.NEAREST)
        target['masks'] = F.resize(target['masks'], self.image_resize, T.InterpolationMode.NEAREST)
        return image, target


def get_transform(image_resize=None, train=True):
    """
        append transformation to image and label using Compose
        sources on pytorch transforms: https://pytorch.org/vision/main/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
        most of ready Classes in torchvision.transforms doesn't support label processing,
            so we create them manually using transforms.functional functions
    """
    transforms = [ToTensor()]
    if image_resize is not None:
        transforms.append(Resize(image_resize=image_resize))
    return Compose(transforms)