import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal

def get_xn(Xs,n):
    '''
    calculate the Fourier coefficient X_n of
    Discrete Fourier Transform (DFT)
    '''
    L  = len(Xs)
    ks = np.arange(0,L,1)
    xn = np.sum(Xs*np.exp((1j*2*np.pi*ks*n)/L))/L
    return(xn)

def get_xns(ts):
    '''
    Compute Fourier coefficients only up to the Nyquest Limit Xn, n=1,...,L/2
    and multiply the absolute value of the Fourier coefficients by 2,
    to account for the symetry of the Fourier coefficients above the Nyquest Limit.
    '''
    mag = []
    L = len(ts)
    for n in range(int(L/2)): # Nyquest Limit
        mag.append(np.abs(get_xn(ts,n))*2)
    return(mag)

def get_Hz_scale_vec(ks,sample_rate,Npoints):
    freq_Hz = ks*sample_rate/Npoints
    freq_Hz  = [int(i) for i in freq_Hz ]
    return(freq_Hz )

def create_spectrogram(ts,NFFT,noverlap = None):
    if noverlap is None:
        noverlap = NFFT/2
    noverlap = int(noverlap)
    starts  = np.arange(0,len(ts),NFFT-noverlap,dtype=int)
    # remove any window with less than NFFT sample size
    starts  = starts[starts + NFFT < len(ts)]
    xns = []
    for start in starts:
        # short term discrete fourier transform
        ts_window = get_xns(ts[start:start + NFFT])
        res = ts_window[:len(ts_window) // 2]
        xns.append(ts_window)
    specX = np.array(xns).T
    # rescale the absolute value of the spectrogram as rescaling is standard
    spec = 10*np.log10(specX)
    #assert spec.shape[1] == len(starts)
    return(starts,spec)

def plot_spectrogram(spec, ts, sample_rate, L, starts, mappable = None):
    plt.figure(figsize=(20,8))
    plt_spec = plt.imshow(spec,origin='lower')

    ## create ylim
    Nyticks = 10
    ks = np.linspace(0,spec.shape[0],Nyticks)
    ksHz = get_Hz_scale_vec(ks,sample_rate,len(ts))
    plt.yticks(ks,ksHz)
    plt.ylabel("Frequency (Hz)")

    ## create xlim
    Nxticks = 10
    total_ts_sec = L / sample_rate
    ts_spec = np.linspace(0,spec.shape[1],Nxticks)
    ts_spec_sec  = ["{:4.2f}".format(i) for i in np.linspace(0,total_ts_sec*starts[-1]/len(ts),Nxticks)]
    plt.xticks(ts_spec,ts_spec_sec)
    plt.xlabel("Time (sec)")

    plt.title("Spectrogram L={} Spectrogram.shape={}".format(L,spec.shape))
    plt.colorbar(mappable,use_gridspec=True)
    plt.show()
    return(plt_spec)


# file_name = input('Path to wav file: ')
# window_length = int(input('Window length(ms): '))
fs, data = wavfile.read('test.wav')
channel_data = data.T[0]
data_length = data.shape[0]
seconds_count = data_length // fs
starts, spec = create_spectrogram(channel_data, int(fs))
plot_spectrogram(spec,channel_data,fs, data_length, starts)

# f, t, Sxx = signal.spectrogram(channel_data, fs)
# plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()


# Ts = 1 / fs
# time = np.arange(0, seconds_count, window_length/100)
# freqs = np.array(scipy.fftpack.fftfreq(data.size, time[1]-time[0]))
# fft_freqs = np.array(freqs[range(data_length//2)])
# max_freq = max(fft_freqs)
# spec_data = spectrogram(channel_data, fs, window_length, max_freq=max_freq)
# f, t, Sxx = signal.spectrogram(channel_data, fs)
# plt.pcolormesh(time, fft_freqs, spec_data)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
# plt.subplot(311)
# p1 = plt.plot(time, data, 'g')
# plt.ylabel("Amplitude")
# plt.xlabel('Time')
# plt.subplot(312)
# p2 = plt.plot(fft_freqs_side, data_f[range(data_length//2)],'r')
# plt.ylabel('Amplitude')
# plt.xlabel('Frequency [Hz]')
# plt.show()