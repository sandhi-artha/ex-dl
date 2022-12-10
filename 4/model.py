from dataloader import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def shallow_AE(sparsity=False, encoding_dim=32):
    """
        uses functional API. creates 3 different models but the encoder and
            decoder are just different endpoints of autoencoder to ease dataflow
            during inference
        regularization allows for more epochs (less likely to overfit)
    """
    regularizer = None
    if sparsity:
        regularizer = tf.keras.regularizers.l1(10e-5)

    _input = tf.keras.Input(shape=(784,))
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

def deep_AE(encoding_dim=32):
    """better representation"""
    _input = tf.keras.Input(shape=(784,))
    _encoder = tf.keras.layers.Dense(128, activation='relu')(_input)
    _encoder = tf.keras.layers.Dense(64, activation='relu')(_encoder)
    _encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(_encoder)

    _decoder = tf.keras.layers.Dense(64, activation='relu')(_encoder)
    _decoder = tf.keras.layers.Dense(128, activation='relu')(_decoder)
    _decoder = tf.keras.layers.Dense(768, activation='sigmoid')(_decoder)

    autoencoder = tf.keras.Model(inputs=_input,outputs=_decoder)

    # model mapping input to encoded representation
    encoder = tf.keras.Model(inputs=_input,outputs=_encoder)

    # model from encoded repr. to output
    _encoder_input = tf.keras.Input(shape=(encoding_dim,))
    _decoder_layer = autoencoder.layers[-1]  # last layer of autoencoder model
    decoder = tf.keras.Model(_encoder_input, _decoder_layer(_encoder_input))

    return autoencoder, encoder, decoder

def deep_convolution_AE():
    """suitable for finding features in images"""
    _input = tf.keras.Input(shape=(28,28,1))
    x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(_input)
    x = tf.keras.layers.MaxPooling2D((2,2), padding='same')(x)  # (14,14,8)
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2,2), padding='same')(x)  # (7,7,8)
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
    _encoder = tf.keras.layers.MaxPooling2D((2,2), padding='same')(x)  # (4,4,8)
    
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')(_encoder)
    x = tf.keras.layers.UpSampling2D((2,2))(x)  # (8,8,8)
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2,2))(x)  # (16,16,8)
    x = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(x)  # (14,14,16)
    x = tf.keras.layers.UpSampling2D((2,2))(x)  # (28,28,16)
    _decoder = tf.keras.layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)  # (28,28,1)

    autoencoder = tf.keras.Model(inputs=_input,outputs=_decoder)

    # model mapping input to encoded representation
    encoder = tf.keras.Model(inputs=_input,outputs=_encoder)

    # # model from encoded repr. to output
    # _encoder_input = tf.keras.Input(shape=(4,4,8))
    # _decoder_layer = autoencoder.layers[-1]  # last layer of autoencoder model
    # decoder = tf.keras.Model(_encoder_input, _decoder_layer(_encoder_input))
    decoder=None  # _encoder_input returns an error, this is a simple fix
    
    return autoencoder, encoder, decoder


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