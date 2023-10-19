from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib.pyplot as plt

def labels_to_colors(arr: np.ndarray, num_classes: int) -> np.ndarray:
    """converts 1D arr to 2D with 3 col"""
    c_arr = np.zeros(shape=(arr.shape[0], 3))
    cmap = plt.get_cmap('tab10')
    for c in range(num_classes):
        c_arr[np.argwhere(arr == c)] = cmap.colors[c]
    return c_arr


def main():
    num_classes = 5     # classes: 0,1,2,3,4
    labels = np.random.randint(num_classes, size=20)  # [N]
    c_labels = labels_to_colors(labels, num_classes)     # [N,3]

    arr = torch.rand((20, 128))
    pca = PCA(n_components=2)   # vector is reduced to n_components
    pca.fit(arr)
    arr_red = pca.transform(arr)

    plt.scatter(arr_red[:, 0], arr_red[:, 1], c=c_labels)
    plt.axis('equal')
    plt.show()


if __name__=='__main__':
    main()