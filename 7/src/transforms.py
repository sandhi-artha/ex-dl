import torchvision.transforms as T

def get_transform():
    """
        append transformation to image and label using Compose
        sources on pytorch transforms: https://pytorch.org/vision/main/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])