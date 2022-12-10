from dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from model import shallow_AE, deep_AE, deep_convolution_AE

class Dict2Obj(object):
    """Turns a dictionary into a class"""
    def __init__(self, dictionary):
        for key in dictionary: setattr(self, key, dictionary[key])

class MyDNN():
    def __init__(self, cfg, AE_type='base'):
        self.dict_cfg = cfg     # used for wandb mostly
        self.cfg = Dict2Obj(cfg)
        self.AE_type = AE_type

    def load_data(self):
        dataloader = DataLoader(self.cfg, self.AE_type)
        self.train_x, self.test_x = dataloader.load_data()
    
    def check_data(self):
        img = self.train_x[0]
        print(img.shape)
        print(img.dtype)

    def load_model(self):
        if self.AE_type == 'base':
            self.autoencoder, self.encoder, self.decoder = shallow_AE(encoding_dim=32)
        elif self.AE_type == 'sparse':
            self.autoencoder, self.encoder, self.decoder = shallow_AE(
                sparsity=False, encoding_dim=32
            )
        elif self.AE_type == 'deep':
            self.autoencoder, self.encoder, self.decoder = deep_AE(encoding_dim=32)

        elif self.AE_type == 'cnn':
            self.autoencoder, self.encoder, self.decoder = deep_convolution_AE()

        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        print(self.autoencoder.summary())

    def train(self):
        self.autoencoder.fit(
            x=self.train_x,
            y=self.train_x,
            epochs=self.cfg.epochs,
            batch_size=self.cfg.bs,
            shuffle=True,
            validation_data=(self.test_x, self.test_x)
        )

    def visualize(self):
        encoded_imgs = self.encoder.predict(self.test_x)
        if self.decoder is None:
            decoded_imgs = self.autoencoder.predict(self.test_x)
        else:
            decoded_imgs = self.decoder.predict(encoded_imgs)

        n = 10  # How many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.test_x[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()