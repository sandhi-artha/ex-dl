import matplotlib.pyplot as plt
import numpy as np

def viz_ds_sample(ds, idx=0):
    print(ds[idx][0].shape)  # img
    print(ds[idx][1].shape)  # label

def viz_dl_batch(dl):
    images, labels = next(iter(dl))
    print(f"Image batch shape: {images.size()}")
    print(f"Label batch shape: {labels.size()}")

    # viz a sample
    img = images[0].squeeze()  # removes start or end dim with size 1 (of single ch img)
    label = labels[0]
    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {label}")
    plt.show()