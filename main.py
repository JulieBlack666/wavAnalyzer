import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal


def spectrogram(samples, sample_rate, stride_size,
                window_size, max_freq=50000, window_func=np.hanning):
    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples,
                                              shape=nshape, strides=nstrides)

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = window_func(window_size)[:, None]

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
    specgram = np.log(fft[:ind, :])
    return specgram


def plot_spectrogram(spec, data, sample_rate, data_length, starts, mappable=None):
    plt.figure(figsize=(20, 8))
    plt_spec = plt.imshow(spec, aspect='auto', origin='lower')

    def get_Hz_scale_vec(ks, sample_rate, Npoints):
        freq_Hz = ks * sample_rate / Npoints
        freq_Hz = [int(i) for i in freq_Hz]
        return (freq_Hz)

    # create y lim
    nyticks = 10
    ks = np.linspace(0, spec.shape[0], nyticks)
    ksHz = get_Hz_scale_vec(ks, sample_rate, spec.shape[0] * 4)
    plt.yticks(ks, ksHz)
    plt.ylabel("Частота (Гц)")

    # create x lim
    nxticks = 10
    total_ts_sec = data_length / sample_rate
    ts_spec = np.linspace(0, spec.shape[1], nxticks)
    ts_spec_sec = ["{:4.2f}".format(i) for i in np.linspace(0, total_ts_sec * starts[-1] / len(data), nxticks)]
    plt.xticks(ts_spec, ts_spec_sec)
    plt.xlabel("Время (с)")

    plt.colorbar(mappable, use_gridspec=True)
    plt.show()
    return plt_spec


def plot_signal(signal):
    time = np.linspace(0, len(signal) / fs, num=len(signal))
    plt.plot(time, channel_data)
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда (Дб)')
    plt.show()


def plot_fft(signal, fs, N):
    fft_res = np.abs(fft(signal))
    t = np.linspace(0, fs, len(fft_res))
    t = t[0:len(fft_res) // 2]
    plt.plot(t, fft_res[0: len(fft_res) // 2])
    plt.xlabel('Частота (Гц)');
    plt.ylabel('Амплитуда (Дб)')
    plt.show()


def calc_window_size(signal_data, fs, window_func=np.hanning):
    main_lobe = {
        np.hamming: 4,
        np.hanning: 4,
        np.blackman: 6
    }
    f, Pxx_den = signal.periodogram(signal_data, fs)
    normalized = Pxx_den / sum(Pxx_den)
    mu = 0
    for i in range(len(f)):
        mu += f[i] * normalized[i]
    window_size = 3 * main_lobe[window_func] * fs / mu
    return int(window_size)


fs, data = wavfile.read('test_files/test2.wav')
channel_data = data.T[0] if len(data.shape) > 1 else data
data_length = data.shape[0]
seconds_count = data_length // fs
time = np.arange(0, seconds_count, 1 / fs)
freqs = np.array(scipy.fftpack.fftfreq(data.size, time[1] - time[0]))
fft_freqs = np.array(freqs[range(data_length // 2)])
max_freq = max(fft_freqs)
ws = calc_window_size(channel_data, fs)
ws_sec = ws / fs
window_ms = ws_sec * 1000
stride_ms = window_ms / 2
print(ws_sec)
print(fs / ws)
spec_data = spectrogram(channel_data, fs, ws//2, ws, max_freq)
plot_spectrogram(spec_data, channel_data, fs, data_length, np.arange(0, data_length, window_ms - stride_ms, dtype=int))
#plot_signal(channel_data)
#plot_fft(channel_data, fs, data_length)
