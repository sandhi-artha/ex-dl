import tensorflow as tf
from tensorflow.keras import backend as K

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

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder, encoder, decoder

def deep_AE(encoding_dim=32):
    """better representation"""
    _input = tf.keras.Input(shape=(784,))
    _encoder = tf.keras.layers.Dense(128, activation='relu')(_input)
    _encoder = tf.keras.layers.Dense(64, activation='relu')(_encoder)
    _encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(_encoder)

    _decoder = tf.keras.layers.Dense(64, activation='relu')(_encoder)
    _decoder = tf.keras.layers.Dense(128, activation='relu')(_decoder)
    _decoder = tf.keras.layers.Dense(784, activation='sigmoid')(_decoder)

    autoencoder = tf.keras.Model(inputs=_input,outputs=_decoder)

    # model mapping input to encoded representation
    encoder = tf.keras.Model(inputs=_input,outputs=_encoder)

    # model from encoded repr. to output
    _encoder_input = tf.keras.Input(shape=(encoding_dim,))
    _decoder_layer = autoencoder.layers[-1]  # last layer of autoencoder model
    decoder = tf.keras.Model(_encoder_input, _decoder_layer(_encoder_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

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
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder, encoder, decoder

def VAE(in_dim=784, hidden_dim=64, latent_dim=2):
    def sampling(args):
        """expects a single variable"""
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(  # random tensor sampled from norm. dist
            shape=(K.shape(z_mean)[0], latent_dim),
            mean=0.0,
            stddev=0.1
        )

        return z_mean + K.exp(z_log_sigma) * epsilon

    _input = tf.keras.Input(shape=(in_dim,))
    _hidden = tf.keras.layers.Dense(hidden_dim, activation='relu')(_input)

    # parameters that define the latent space
    z_mean = tf.keras.layers.Dense(latent_dim)(_hidden)
    z_log_sigma = tf.keras.layers.Dense(latent_dim)(_hidden)

    # sample a point z from the latent space
    z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

    # create encoder
    encoder = tf.keras.Model(
        inputs=_input,
        outputs=[z_mean, z_log_sigma, z],
        name='encoder')

    # map sample point back to original input space
    _latent_input = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(_latent_input)
    _output = tf.keras.layers.Dense(in_dim, activation='sigmoid')(x)
    
    # create decoder
    decoder = tf.keras.Model(inputs=_latent_input, outputs=_output, name='decoder')

    # the VAE model - needs to connect each layer using functional API
    _output = decoder(encoder(_input)[-1])  # a bit strange bcz encoder already has input of _input
    vae = tf.keras.Model(inputs=_input, outputs=_output, name='vae')

    # train using reconstruction loss:
    reconstruction_loss = tf.keras.losses.binary_crossentropy(_input, _output)
    reconstruction_loss *= in_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return vae, encoder, decoder