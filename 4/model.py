import tensorflow as tf

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


