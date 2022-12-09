from dataloader import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def shallow_AE(sparsity=False, encoding_dim=32):
    """
        uses functional API. creates 3 different models but the encoder and
            decoder are just different endpoints of autoencoder to ease dataflow
            during inference
    """
    regularizer = None
    if sparsity:
        regularizer = tf.keras.regularizers.l1(10e-5)

    _input = tf.keras.Input(shape=784)
    _encoder = tf.keras.layers.Dense(
        encoding_dim, activation='relu',
        activity_regularizer=regularizer
    )(_input)
    _decoder = tf.keras.layers.Dense(784, activation='sigmoid')(_encoder)

    # model mapping input to reconstruction
    autoencoder = tf.keras.Model(inputs=_input,outputs=_decoder)

    # model mapping input to encoded representation
    encoder = tf.keras.Model(inputs=_input,outputs=_encoder)

    # model from encoded repr. to output
    _encoder_input = tf.keras.Input(shape=(encoding_dim,))
    _decoder_layer = autoencoder.layers[-1]  # last layer of autoencoder model
    decoder = tf.keras.Model(_encoder_input, _decoder_layer(_encoder_input))

    return autoencoder, encoder, decoder



class Dict2Obj(object):
    """Turns a dictionary into a class"""
    def __init__(self, dictionary):
        for key in dictionary: setattr(self, key, dictionary[key])

class MyDNN():
    def __init__(self, cfg, type='base'):
        self.dict_cfg = cfg     # used for wandb mostly
        self.cfg = Dict2Obj(cfg)
        self.type = type

    def load_data(self):
        dataloader = DataLoader(self.cfg)
        self.train_x, self.test_x = dataloader.load_data()
    
    def check_data(self):
        img = self.train_x[0]
        print(img.shape)
        print(img.dtype)

    def load_model(self):
        if self.type == 'base':
            self.autoencoder, self.encoder, self.decoder = shallow_AE(encoding_dim=32)
        elif self.type == 'sparse':
            self.autoencoder, self.encoder, self.decoder = shallow_AE(
                sparsity=False, encoding_dim=32
            )

        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

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