import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fft
from scipy.io import wavfile


def spectrogram(samples, sample_rate, stride_ms=10.0,
                window_ms=20.0, max_freq=50000, eps=1e-14):
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples,
                                              shape=nshape, strides=nstrides)

    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]

    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft ** 2

    scale = np.sum(weighting ** 2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale

    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return specgram


def get_Hz_scale_vec(ks, sample_rate, Npoints):
    freq_Hz = ks * sample_rate / Npoints
    freq_Hz = [int(i) for i in freq_Hz]
    return (freq_Hz)


def plot_spectrogram(spec, data, sample_rate, data_length, starts, mappable=None):
    plt.figure(figsize=(20, 8))
    plt_spec = plt.imshow(spec, origin='lower')

    # create y lim
    nyticks = 10
    ks = np.linspace(0, spec.shape[0], nyticks)
    ksHz = get_Hz_scale_vec(ks, sample_rate, spec.shape[0] * 4)
    plt.yticks(ks, ksHz)
    plt.ylabel("Frequency (Hz)")

    # create x lim
    nxticks = 10
    total_ts_sec = data_length / sample_rate
    ts_spec = np.linspace(0, spec.shape[1], nxticks)
    ts_spec_sec = ["{:4.2f}".format(i) for i in np.linspace(0, total_ts_sec * starts[-1] / len(data), nxticks)]
    plt.xticks(ts_spec, ts_spec_sec)
    plt.xlabel("Time (sec)")

    plt.title("Spectrogram L={} Spectrogram.shape={}".format(data_length, spec.shape))
    plt.colorbar(mappable, use_gridspec=True)
    plt.show()
    return plt_spec


# file_name = input('Path to wav file: ')
# window_length = int(input('Window length(ms): '))
window_ms = 20
stride_ms = 10
fs, data = wavfile.read('test.wav')
channel_data = data.T[0]
data_length = data.shape[0]
seconds_count = data_length // fs
time = np.arange(0, seconds_count, 1 / fs)
freqs = np.array(scipy.fftpack.fftfreq(data.size, time[1] - time[0]))
fft_freqs = np.array(freqs[range(data_length // 2)])
max_freq = max(fft_freqs)
spec_data = spectrogram(channel_data, fs, stride_ms, window_ms, max_freq)

plot_spectrogram(spec_data, channel_data, fs, data_length, np.arange(0, data_length, window_ms - stride_ms, dtype=int))
