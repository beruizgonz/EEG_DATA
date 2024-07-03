
import numpy as np
import torch
import os
import pandas as pd
import h5py
from torch.utils.data import Dataset
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.pardir))

from utils import plot_masked_data, normalization, padding_mask

def normalization1(signal, apply_signal, n_channels=8):
    eps = 1e-8
    means = signal.mean(axis=1, keepdims=True)
    stds = signal.std(axis=1, keepdims=True)
    apply_signal = (apply_signal - means) / (stds + eps)
    return apply_signal


def normalize_min_max(data,apply_signal, n_channels=1):
    # Normalize the signal using Min-Max normalization
    for j in range(n_channels):
        mins = data[:,j].min(axis=1, keepdims=True)
        maxs = data[:,j].max(axis=1, keepdims=True)
        apply_signal[:,j] = (apply_signal[:,j] - mins) / (maxs - mins)
    return apply_signal

def normalize_min_max1(data,apply_signal, n_channels=8):
    # Normalize the signal using Min-Max normalization
    for j in range(n_channels):
        mins = data[j, :].min(axis=0, keepdims=True)
        maxs = data[j, :].max(axis=0, keepdims=True)
        apply_signal[j,:] = (apply_signal[j,:] - mins) / (maxs - mins)
    return apply_signal

class MaskedDataset(Dataset):
    def __init__(self, hdf5_file, normalize = None):
        self.hdf5_file = hdf5_file
        self.normalize = normalize
        with h5py.File(hdf5_file, 'r') as hf:
            self.data_shape = hf['eeg'].shape

    def __len__(self):
        return self.data_shape[0]

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as hf:
            raw_eeg = np.array(hf['eeg'][idx])
            masked_eeg = np.array(hf['masked'][idx])
            mask = np.array(hf['mask'][idx])
            raw_eeg_n = raw_eeg.copy()
        if self.normalize == 'normalization':
            raw_eeg_n = normalization1(raw_eeg, raw_eeg_n)
            masked_eeg = normalization1(raw_eeg, masked_eeg)
        elif self.normalize == 'min_max':
   
            raw_eeg_n = normalize_min_max1(raw_eeg, raw_eeg_n)
            masked_eeg = normalize_min_max1(raw_eeg, masked_eeg)

        #Transpose if necessary
       
        raw_eeg_n= raw_eeg_n.transpose(1, 0)
        masked_eeg = masked_eeg.transpose(1, 0)
        mask = mask.transpose(1,0)
        # Invert the mask
        mask = 1 - mask

        return torch.from_numpy(masked_eeg).float(), torch.from_numpy(raw_eeg_n).float(), torch.from_numpy(mask).bool()


class ERPDataset(Dataset):
    # must be numpy array of shape (n_samples, ts_steps) and (n_samples, 1)
    def __init__(self, data_erp_path, normalize = None):
        self.hf = h5py.File(data_erp_path, 'r')
        self.features = np.array(self.hf.get("features"))
        self.erp_labels = np.array(self.hf.get("erp_labels"))

        # Get only the 10000 first samples
        self.features = self.features[:10000]
        self.erp_labels = self.erp_labels[:10000]
        self.normalize = normalize
        self.hf.close()
        eps = 1e-8
        # for j in range(8):  # Assuming 8 channels
        #     means = self.features[:,j,:].mean(axis=0, keepdims=True)
        #     stds = self.features[:,j,:].std(axis=0, keepdims=True)
        #     self.features[:, j,:] = (self.features[:,j,:] - means) / (stds + eps)
        # #self.erp_labels = one_hot_labels(self.erp_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        X = self.features[idx].copy()
        if self.normalize == 'normalization':
            X = normalization1(self.features[idx], X)
        elif self.normalize == 'min_max':
            X = normalize_min_max(self.features[idx], X)
        else:
            X = self.features[idx].copy()
        Y = self.erp_labels[idx]
        return torch.from_numpy(X).float(), Y

class EmotionDataset(Dataset):
    # must be numpy array of shape (n_samples, ts_steps) and (n_samples, 1)
    def __init__(self, data_path, normalize = None):

        # Load the seed dataset
        with h5py.File(data_path, 'r') as hf:
           self.signals = np.array(hf.get("signals"))
           self.labels = np.array(hf.get("labels"))
        hf.close()

        if normalize == 'normalization':
            self.signals_n = normalization(self.signals, self.signals)
        elif normalize == 'min_max':
            self.signals_n = normalize_min_max(self.signals, self.signals)
        else:
            self.signals_n = self.signals.copy()

        #self.signals_n = self.signals_n.reshape(-1, 128, 8)
        self.signals_n = self.signals_n.reshape(-1, 8, 128)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.signals_n[idx]
        Y = self.labels[idx]
        Y = int(Y)
        return torch.from_numpy(X).float(), Y

class StressDataset(Dataset):
    def __init__(self, data_path, normalize = None):

        self.hdf5_file = data_path
        self.normalize = normalize
        with h5py.File(data_path, 'r') as hf:
            self.subjects = np.array(hf.get("subject"))

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as hf:
            raw_eeg = np.array(hf['signals'][idx])
            labels = np.array(hf['labels'][idx])
            raw_eeg_n = raw_eeg.copy()
        if self.normalize == 'normalization':
            raw_eeg_n = normalization1(raw_eeg, raw_eeg_n)
        elif self.normalize == 'min_max':
            raw_eeg_n = normalize_min_max1(raw_eeg, raw_eeg_n)

        X = raw_eeg_n.transpose(1, 0)
        Y = labels
        return torch.from_numpy(X).float(), np.float32(Y)
    

class Valence_Arousal(Dataset):
    def __init__(self, data_path, normalize = None):
        with h5py.File(data_path, 'r') as hf:
           self.signals = np.array(hf.get("signals"))
           self.valence = np.array(hf.get("valence"))
           self.arousal = np.array(hf.get("arousal"))
        hf.close()
        self.signals_n = self.signals.copy()
        if normalize == 'normalization':
            self.signals_n = normalization1(self.signals, self.signals_n)
        elif normalize == 'min_max':
            self.signals_n = normalize_min_max(self.signals, self.signals_n)
        else:
            self.signals_n = self.signals.copy()
    
    def __len__(self):
        return len(self.valence)
    
    def __getitem__(self, idx):
        X = self.signals_n[idx].transpose(1,0)
        Y = self.valence[idx]
        return torch.from_numpy(X).float(), Y.astype(np.float32)

if __name__ == "__main__":
    # Load an ecg of masked dataset and plot mask and the ecg
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path_mask = os.path.join(parent_dir, 'preprocess_data/deap_mask.hdf5')
    path_lemon = os.path.join(parent_dir, 'data/LEMON-mask-4s-8channel.h5')
    path_stress = os.path.join(parent_dir, 'data/deap_stress_4s_8channel.h5')
    
    path_uva_mask = os.path.join(parent_dir, 'data/UVA-mask.h5')
    dataset1 = MaskedDataset(path_lemon, normalize=None)
    print(dataset1[0][0].shape)
    sample1 = dataset1[0]

    dataset2 = StressDataset(path_stress, normalize = 'normalization')
    sample2 = dataset2[0]

    dataset3 = MaskedDataset(path_lemon, normalize='normalization')
    sample3 = dataset3[0]
    
    dataset4 = ERPDataset('./UVA-DATASET/archive/GIB-UVA ERP-BCI.hdf5', normalize='normalization')
    sample4 = dataset4[18]

    # dataset5 = ERPDataset('./UVA-DATASET/archive/GIB-UVA ERP-BCI.hdf5', normalize='min_max')
    # sample5 = dataset5[18]

    dataset6 = StressDataset(path_stress)
    sample6 = dataset6[0]

    print(sample3[0].shape)
    # Print the type of the values of the sample
    print(sample1[0].dtype, sample1[1].dtype, sample1[2].dtype)
    
    # Plot the data
    fig, ax1 = plt.subplots()

    # Plot the first signal on the primary y-axis
    ax1.plot(sample2[0][:,6], 'b-', label='Signal 1')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Signal 1 (blue)', color='b')
    ax1.tick_params('y', colors='b')

    # Create a secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    # Plot the second signal on the secondary y-axis
    ax2.plot(sample6[0][:,6], 'r-', label='Signal 2')
    ax2.set_ylabel('Signal 2 (red)', color='r')
    ax2.tick_params('y', colors='r')

    # Adding legends
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

    # Adding a title
    plt.title('Two signals with different scales in the same plot')

    # Show the plot
    plt.show()

    print(len(dataset1))