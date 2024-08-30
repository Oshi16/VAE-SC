from tensorflow.keras.datasets import mnist
from train import train_vae
from plotting import plot_metrics, plot_loss

# Load and preprocess the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

latent_dims = [16, 32, 64, 128]
snr_range = range(0, 61, 5)
epochs = 10
batch_size = 128
K = 5  # Rician factor
m = 2  # Nakagami-m parameter

training_details = {'rayleigh': {}, 'rician': {}, 'nakagami': {}}

for channel_type in ['rayleigh', 'rician', 'nakagami']:
    for latent_dim in latent_dims:
        training_details[channel_type][latent_dim] = [train_vae(channel_type, latent_dim, snr_db, x_train, x_test, epochs, batch_size, K, m) for snr_db in snr_range]

plot_metrics('PSNR', training_details, snr_range)
plot_metrics('SSIM', training_details, snr_range)
plot_metrics('MSE', training_details, snr_range)

plot_loss(training_details, 'rayleigh', latent_dims)
plot_loss(training_details, 'rician', latent_dims)
plot_loss(training_details, 'nakagami', latent_dims)
