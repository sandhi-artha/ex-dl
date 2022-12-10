import os
import h5py
import tensorflow as tf
import numpy as np


AUTO = tf.data.experimental.AUTOTUNE

class DataLoader:
    def __init__(self, cfg, AE_type):
        self.cfg = cfg
        self.AE_type = AE_type

    def _read_data(self):
        """
            self.cfg.ds_path : str
                where 'train.hdf5' and 'test.hdf5' exist
            train_x = [60000,28,28] np.array, np.uint8
        """
        with h5py.File(os.path.join(self.cfg.ds_path,'train.hdf5'), 'r') as f:
            train_x, train_y = f['image'][...], f['label'][...]

        with h5py.File(os.path.join(self.cfg.ds_path,'test.hdf5'), 'r') as f:
            test_x, test_y = f['image'][...], f['label'][...]
        print(train_x.shape)
        print(train_x.dtype)
        return train_x, test_x    

    def _preprocess_dataset(self, dataset):
        """normalize and reshape"""
        assert len(dataset.shape) == 3
        dataset = dataset.astype('float32') / 255.
        if self.AE_type == 'cnn':
            dataset = np.expand_dims(dataset, axis=-1)  # (60000,28,28,1)
        else:
            dataset = dataset.reshape(
                (len(dataset), np.prod(dataset.shape[1:]))  # (60000,784)
            )
        return dataset

    def load_data(self):
        train_x, test_x = self._read_data()
        train_x = self._preprocess_dataset(train_x)
        test_x = self._preprocess_dataset(test_x)
        return train_x, test_x

