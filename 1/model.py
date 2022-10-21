from dataloader import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class Dict2Obj(object):
    """Turns a dictionary into a class"""
    def __init__(self, dictionary):
        for key in dictionary: setattr(self, key, dictionary[key])

class MyDNN():
    def __init__(self, cfg):
        self.dict_cfg = cfg     # used for wandb mostly
        self.cfg = Dict2Obj(cfg)

    def load_data(self):
        dataloader = DataLoader(self.cfg)
        self.ds_train, self.ds_test = dataloader.load_data()
    
    def check_data(self):
        for img, label in self.ds_train.take(1):
            print(img.shape)
            print(label.shape)
            print(label[0])

    def load_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self.cfg.image_shape),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])

        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        self.model.compile(metrics=metrics, optimizer=optimizer, loss=loss)
        print(f'Total params: {self.model.count_params():,}')

    def train(self):
        self.history = self.model.fit(
            x=self.ds_train,
            epochs=self.cfg.epochs,
            validation_data=self.ds_test
        )

    def evaluate(self):
        for image, label in self.ds_test.take(1):
            predictions = self.model.predict(image)

        # plot 16 test images and their prediction
        f = plt.figure(figsize=(12,12))
        for i in range(16):
            ax = f.add_subplot(4,4,i+1)
            ax.imshow(image[i])
            pred = np.argmax(predictions[i])
            conf = np.max(predictions[i])
            ax.set_title(f'pred: {pred} (conf:{conf:.2f}), true: {label[i]}')
            
        plt.show()