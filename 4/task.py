from dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from model import shallow_AE, deep_AE, deep_convolution_AE, VAE

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
        self.train_x, self.test_x, self.test_y = dataloader.load_data()
    
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
        
        elif self.AE_type == 'vae':
            self.autoencoder, self.encoder, self.decoder = VAE(
                in_dim=784, hidden_dim=64, latent_dim=2
            )

    def check_model(self):
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
            if self.AE_type == 'vae':
                _z_mean, _z_log_sigma, _z = encoded_imgs
                encoded_imgs = _z
                # plot_latent_space(self.encoder, self.test_x, self.test_y)
                plot_latent_scan(self.decoder)

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


def plot_latent_space(encoder, x, y):
    _z_mean, _z_log_sigma, _z = encoder.predict(x)
    plt.figure(figsize=(6,6))
    plt.scatter(_z_mean[:,0], _z_mean[:,1], c=y)
    plt.colorbar()
    plt.show()

def plot_latent_scan(decoder, n=15):
    """Display a 2D manifold of the digits"""
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # We will sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-15, 15, n)
    grid_y = np.linspace(-15, 15, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()