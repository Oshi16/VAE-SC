import tensorflow as tf

@tf.function(jit_compile=False)
def rayleigh_fading_channel(x, snr_db):
    noise_real = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0, dtype=x.dtype)
    noise_imag = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0, dtype=x.dtype)
    fading = tf.sqrt(tf.square(noise_real) + tf.square(noise_imag))
    faded_x = x * fading
    signal_power = tf.reduce_mean(tf.square(faded_x))
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=tf.sqrt(noise_power), dtype=x.dtype)
    noisy_x = faded_x + noise
    return noisy_x

@tf.function(jit_compile=False)
def rician_fading_channel(x, snr_db, K):
    K = tf.cast(K, dtype=tf.float32)
    noise_real = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0, dtype=x.dtype)
    noise_imag = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0, dtype=x.dtype)
    scattering = tf.sqrt(tf.square(noise_real) + tf.square(noise_imag))
    los_component = tf.sqrt(K / (K + 1))
    scattering_component = scattering / tf.sqrt(K + 1)
    fading = los_component + scattering_component
    faded_x = x * fading
    signal_power = tf.reduce_mean(tf.square(faded_x))
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=tf.sqrt(noise_power), dtype=x.dtype)
    noisy_x = faded_x + noise
    return noisy_x

@tf.function(jit_compile=False)
def nakagami_fading_channel(x, snr_db, m):
    m = tf.cast(m, dtype=tf.float32)
    fading = tf.random.gamma(shape=tf.shape(x), alpha=m, beta=1.0/m, dtype=x.dtype)
    faded_x = x * fading
    signal_power = tf.reduce_mean(tf.square(faded_x))
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=tf.sqrt(noise_power), dtype=x.dtype)
    noisy_x = faded_x + noise
    return noisy_x
