from tensorflow.keras import layers, models
import tensorflow as tf

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(latent_dim):
    encoder_inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = models.Model(encoder_inputs, [z_mean, z_log_var], name='encoder')

    return encoder

def build_decoder(latent_dim):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)

    decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

    return models.Model(latent_inputs, decoder_outputs, name='decoder')

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, snr_db, K, m):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.snr_db = snr_db
        self.K = K
        self.m = m

    def call(self, inputs, training=False):
        z_mean, z_log_var = self.encoder(inputs)
        z = sampling([z_mean, z_log_var])

        # Channel effects to be added here as needed
        reconstructed_rayleigh = self.decoder(rayleigh_fading_channel(z, self.snr_db))
        reconstructed_rician = self.decoder(rician_fading_channel(z, self.snr_db, self.K))
        reconstructed_nakagami = self.decoder(nakagami_fading_channel(z, self.snr_db, self.m))

        # Calculate losses
        reconstruction_loss_rayleigh = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(inputs, reconstructed_rayleigh), axis=(1, 2)
            )
        )
        reconstruction_loss_rician = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(inputs, reconstructed_rician), axis=(1, 2)
            )
        )
        reconstruction_loss_nakagami = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(inputs, reconstructed_nakagami), axis=(1, 2)
            )
        )

        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)

        # Total loss
        total_loss = reconstruction_loss_rayleigh + reconstruction_loss_rician + reconstruction_loss_nakagami + kl_loss

        self.add_loss(total_loss)

        return reconstructed_rayleigh, reconstructed_rician, reconstructed_nakagami
