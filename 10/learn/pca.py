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

def pca_stats(arr_red: np.ndarray) -> str:
    """ xmax | xmin | xmean | xstd
        ymax | ymin | ymean | ystd
    """
    return '{:.3f} | {:.3f} | {:.3f} | {:.3f}\n{:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
        np.max(arr_red[:, 0]), np.min(arr_red[:, 0]),
        np.mean(arr_red[:, 0]), np.std(arr_red[:, 0]),
        np.max(arr_red[:, 1]), np.min(arr_red[:, 1]),
        np.mean(arr_red[:, 1]), np.std(arr_red[:, 1]),
    )

def main():
    num_classes = 5     # classes: 0,1,2,3,4
    labels = np.random.randint(num_classes, size=20)  # [N]
    c_labels = labels_to_colors(labels, num_classes)     # [N,3]

    arr = torch.rand((20, 128))
    pca = PCA(n_components=2)   # vector is reduced to n_components
    pca.fit(arr)
    arr_red = pca.transform(arr)

    f, ax = plt.subplots(1,1,figsize=(6,6))
    ax.scatter(arr_red[:, 0], arr_red[:, 1], c=c_labels)
    ax.axis('equal')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    # ax.set_aspect('equal', 'box')
    ax.set_title(pca_stats(arr_red))
    plt.show()


if __name__=='__main__':
    main()