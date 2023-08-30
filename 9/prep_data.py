import os
from glob import glob

import pandas as pd

from src.cfg import cfg


def main(cfg):
    classes = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck',
    ]

    train_fps = glob(f"{cfg['data_path']}/train/*/*.png")
    train_labels = [classes.index(fp.split('\\')[-2]) for fp in train_fps]
    print(train_fps[0], train_labels[0])
    # show: '../data/cifar10/test\\airplane\\0001.png' 0

    df = pd.DataFrame({'path': train_fps, 'label': train_labels})
    df.to_csv(os.path.join(cfg['data_path'], 'train.csv'), index=False)

    test_fps = glob(f"{cfg['data_path']}/test/*/*.png")
    test_labels = [classes.index(fp.split('\\')[-2]) for fp in test_fps]
    df = pd.DataFrame({'path': test_fps, 'label': test_labels})
    df.to_csv(os.path.join(cfg['data_path'], 'test.csv'), index=False)


if __name__ == '__main__':
    main(cfg)