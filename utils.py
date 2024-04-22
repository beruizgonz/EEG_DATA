import numpy as np
import matplotlib.pyplot as plt
import pyedflib
from scipy.signal import resample, butter, lfilter
from sklearn.preprocessing import OneHotEncoder
import h5py

def plot_masked_data(signals, signal_masked, signal_mask, seq_length=1280, n_channels=8):
    fig, axs = plt.subplots(n_channels, 1, figsize=(15, 10))
    for i in range(n_channels):
        axs[i].plot(signals[i, :seq_length], label='Raw Signal')
        axs[i].plot(signal_masked[i, :seq_length], label='Masked Signal')
        axs[i].plot(signal_mask[i, :seq_length], label='Mask')
        
        axs[i].legend() 
        axs[i].set_title('Channel {}'.format(i+1)) 
        axs[i].grid()
    plt.tight_layout()
    plt.show()

def save_csv(signals, file_name='reshaped_signals.csv'):
    reshaped_signals = signals.reshape(signals.shape[0], -1)
    print(reshaped_signals.shape)
    np.savetxt(file_name, reshaped_signals, delimiter=',')

def normalize_min_max(signal, apply_signal, n_channels=8):
    # Normalize the signal using Min-Max normalization
    for j in range(n_channels): 
        mins = signal[:, j, :].min(axis=1, keepdims=True)
        maxs = signal[:, j, :].max(axis=1, keepdims=True)
        apply_signal[:, j, :] = (apply_signal[:, j, :] - mins) / (maxs - mins)
    return apply_signal

def normalization(signal, apply_signal, n_channels=8):
    eps = 1e-8
    for j in range(n_channels): 
        means = signal[:, j, :].mean(axis=1, keepdims=True)
        stds = signal[:, j, :].std(axis=1, keepdims=True)
        apply_signal[:, j, :] = (apply_signal[:, j, :] - means) / (stds + eps)
    return apply_signal

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def divide_signal(signal, seg_length):
    n_segments = signal.shape[1] // seg_length
    new_signal = signal[:, :n_segments * seg_length].reshape(signal.shape[0], n_segments, seg_length)
    #plt.plot(signal[0, :seg_length], label='Original Signal')
    return new_signal

def downsample(signal, freq, new_freq):
    new_n_samples = int((signal.shape[1] / freq) * new_freq)
    new_signal = resample(signal, new_n_samples, axis=1)
    return new_signal

def  read_edf(data_path, channels):
    f = pyedflib.EdfReader(data_path)
    n = len(channels)
    sigbufs = [f.readSignal(i) for i in range(n)]
    sigbufs = np.array(sigbufs)
    sigbufs = bandpass_filter(sigbufs, 0.5, 50, 256)
    d_sigbufs = divide_signal(sigbufs, 2560)
    return d_sigbufs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())

def one_hot_labels(categorical_labels):
    enc = OneHotEncoder(handle_unknown='ignore')
    on_hot_labels = enc.fit_transform(
        categorical_labels.reshape(-1, 1)).toarray()
    return on_hot_labels


def convert_csv_h5(path_file):
    data = np.loadtxt(path_file, delimiter=',')
    hf = h5py.File(path_file.replace('.csv', '.h5'), 'w')
    hf.create_dataset('data', data=data)
    hf.close()