import matplotlib.pyplot as plt

def plot_metrics(metric_name, results, snr_range):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channels = ['rayleigh', 'rician', 'nakagami']

    for i, channel in enumerate(channels):
        axes[i].plot(snr_range, results[metric_name][channel], label=f'{channel.capitalize()} {metric_name}')
        axes[i].set_xlabel('SNR (dB)')
        axes[i].set_ylabel(metric_name)
        axes[i].set_title(f'{channel.capitalize()} Channel')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_loss(training_details, channel_type, latent_dims):
    fig, ax = plt.subplots(figsize=(10, 6))

    for latent_dim in latent_dims:
        snrs = [item['snr_db'] for item in training_details[channel_type][latent_dim]]
        losses = [item['val_loss'][-1] for item in training_details[channel_type][latent_dim]]

        ax.plot(snrs, losses, marker='o', linestyle='-', label=f'Latent Dim {latent_dim}')
    ax.set_title(f'{channel_type.capitalize()} Channel - Loss vs SNR')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.legend()

    plt.show()
