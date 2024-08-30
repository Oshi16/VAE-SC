import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as comp_ssim
from skimage.metrics import mean_squared_error as comp_mse

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test):
        self.x_test = x_test
        self.psnr_list = []
        self.ssim_list = []
        self.mse_list = []

    def on_epoch_end(self, epoch, logs=None):
        reconstructed_images = self.model.decoder(self.model.encoder(self.x_test)[0])

        psnr_value = comp_psnr(self.x_test, reconstructed_images)
        ssim_value = comp_ssim(self.x_test, reconstructed_images)
        mse_value = comp_mse(self.x_test, reconstructed_images)

        self.psnr_list.append(psnr_value)
        self.ssim_list.append(ssim_value)
        self.mse_list.append(mse_value)

        print(f"Epoch {epoch + 1}: PSNR = {psnr_value}, SSIM = {ssim_value}, MSE = {mse_value}")
