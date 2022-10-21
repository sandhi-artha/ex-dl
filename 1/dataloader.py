import os
import h5py
import tensorflow as tf


AUTO = tf.data.experimental.AUTOTUNE

class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def _read_data(self):
        """
            self.cfg.ds_path : str
                where 'train.hdf5' and 'test.hdf5' exist
        """
        with h5py.File(os.path.join(self.cfg.ds_path,'train.hdf5'), 'r') as f:
            train_x, train_y = f['image'][...], f['label'][...]

        with h5py.File(os.path.join(self.cfg.ds_path,'test.hdf5'), 'r') as f:
            test_x, test_y = f['image'][...], f['label'][...]
        
        # convert to tf.data
        self.ds_train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        self.ds_test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        
    def _preprocess_data(self, image, label):
        """reshape image to have channel dimension and normalize"""
        image = tf.expand_dims(image, axis=-1)
        image = tf.cast(image, tf.float32) / 255.
        # label = tf.one_hot(indices=label, depth=10)
        return image, label
    
    def _get_train_ds(self):
        self.ds_train = self.ds_train  \
            .map(self._preprocess_data, num_parallel_calls=AUTO)  \
            .cache()  \
            .shuffle(self.cfg.buffer_size)  \
            .batch(self.cfg.bs)  \
            .prefetch(AUTO)

    def _get_test_ds(self):
        self.ds_test = self.ds_test  \
            .map(self._preprocess_data, num_parallel_calls=AUTO)  \
            .batch(self.cfg.bs)  \
            .cache()  \
            .prefetch(AUTO)

    def load_data(self):
        self._read_data()
        self._get_train_ds()
        self._get_test_ds()
        return self.ds_train, self.ds_test

