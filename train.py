import os
import tensorflow as tf
from model import VAE, build_encoder, build_decoder
from channels import rayleigh_fading_channel, rician_fading_channel, nakagami_fading_channel
from metrics import MetricsCallback

def train_vae(channel_type, latent_dim, snr_db, x_train, x_test, epochs, batch_size, K, m):
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)

    if channel_type == 'rayleigh':
        vae = VAE(encoder, decoder, snr_db=snr_db, K=0, m=0)
    elif channel_type == 'rician':
        vae = VAE(encoder, decoder, snr_db=snr_db, K=K, m=0)
    elif channel_type == 'nakagami':
        vae = VAE(encoder, decoder, snr_db=snr_db, K=0, m=m)
    else:
        raise ValueError("Invalid channel type specified")

    vae.compile(optimizer='adam')
    
    checkpoint_dir = './VAE_checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    initial_epoch = 0
    if latest_checkpoint:
        vae.load_weights(latest_checkpoint)
        initial_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])  # Extract epoch number

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./VAE_checkpoints/vae_{epoch:02d}.weights.h5',
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )

    metrics_callback = MetricsCallback(x_test)

    vae.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[checkpoint_callback, metrics_callback]
    )

    return vae
