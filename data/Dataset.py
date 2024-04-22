
import numpy as np
import torch
import os 
import pandas as pd 
import h5py
from torch.utils.data import Dataset
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.pardir))

from utils import plot_masked_data, normalization, normalize_min_max

class MaskedDataset(Dataset):
    # must be numpy array of shape (n_samples, ts_steps) and (n_samples, 1)
    def __init__(self, raw_csv : str = "./reshaped_signals.csv", masked_csv : str = "./masked_data.csv", mask : str = "./mask_data.csv", nrows = 1000, seq_length=1280, normalize = None):
        raw_eeg = pd.read_csv(raw_csv,nrows = nrows, header = None) # use pyarrow engine to read csv file as it is faster than pandas default engine
        self.masked_eeg = pd.read_csv(masked_csv, nrows=nrows, header=None)  # no column names in csv file
        self.mask = pd.read_csv(mask, nrows = nrows, header=None)
        # remove first column of each dataframe
        self.raw_eeg = raw_eeg.to_numpy()
        self.masked_eeg = self.masked_eeg.to_numpy()
        self.mask = self.mask.to_numpy()

        self.raw_eeg = self.raw_eeg.reshape(-1, 8, seq_length) #(n_samples, n_channels, n_timesteps)
        self.masked_eeg = self.masked_eeg.reshape(-1, 8, seq_length) 
        self.mask = self.mask.reshape(-1, 8, seq_length)

        self.raw_eeg_n = self.raw_eeg.copy()
        if normalize == 'normalization':
            self.raw_eeg_n = normalization(self.raw_eeg, self.raw_eeg_n)
            self.masked_eeg = normalization(self.raw_eeg, self.masked_eeg)
        elif normalize == 'min_max':
            self.raw_eeg = normalize_min_max(self.raw_eeg, self.raw_eeg_n)
            self.masked_eeg = normalize_min_max(self.raw_eeg, self.masked_eeg)
            
        self.raw_eeg_n = self.raw_eeg_n.reshape(-1, seq_length, 8)
        self.masked_eeg = self.masked_eeg.reshape(-1,seq_length, 8)
        self.mask = self.mask.reshape(-1, seq_length, 8)

        #self.mask = 1 - self.mask  # Invert the mask
        
    def __len__(self):
        return len(self.raw_eeg)
    
    def __getitem__(self, idx):
        Y = self.raw_eeg_n[idx]
        X = self.masked_eeg[idx]
        M = self.mask[idx]
        return torch.from_numpy(X).float(), torch.from_numpy(Y).float(), torch.from_numpy(M).bool()
    

class MaskedDataset1(Dataset):
    def __init__(self, hdf5_file, normalize = None):
        hf = h5py.File(hdf5_file, 'r')
        self.raw_eeg = np.array(hf.get("eeg"))
        self.masked_eeg = np.array(hf.get("masked"))
        self.mask = np.array(hf.get("mask"))
        hf.close()
        print(self.raw_eeg.shape)
        self.raw_eeg_n = self.raw_eeg.copy()
        self.masked_eeg_n = self.masked_eeg.copy()
        if normalize == 'normalization':
            self.raw_eeg_n = normalization(self.raw_eeg, self.raw_eeg_n)
            self.masked_eeg_n = normalization(self.raw_eeg, self.masked_eeg_n)
        elif normalize == 'min_max':
            self.raw_eeg_n = normalize_min_max(self.raw_eeg, self.raw_eeg)
            self.masked_eeg_n = normalize_min_max(self.raw_eeg, self.masked_eeg_n)
        
        self.raw_eeg_n = self.raw_eeg_n.transpose(0,2,1)
        self.masked_eeg_n = self.masked_eeg_n.transpose(0,2,1)
        self.mask = self.mask.transpose(0,2,1)
    
        self.mask = 1 - self.mask  # Invert the mask
            
    def __len__(self):
        return len(self.raw_eeg)
    
    def __getitem__(self, idx):
        Y = self.raw_eeg_n[idx]
        X = self.masked_eeg_n[idx]
        M = self.mask[idx]
        return torch.from_numpy(X).float(), torch.from_numpy(Y).float(), torch.from_numpy(M).bool()
    
def normalization1(signal, apply_signal, n_channels=8):
    eps = 1e-8
    means = signal.mean(axis=1, keepdims=True)
    stds = signal.std(axis=1, keepdims=True)
    apply_signal = (apply_signal - means) / (stds + eps)
    return apply_signal


# def normalize_min_max(data):
#     # Example min-max normalization
#     return (data - np.min(data)) / (np.max(data) - np.min(data))    
    
class MaskedDataset2(Dataset):
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
            raw_eeg = normalize_min_max(raw_eeg)
            masked_eeg = normalize_min_max(raw_eeg)

        # Transpose if necessary
        raw_eeg_n= raw_eeg_n.transpose(1, 0)
        masked_eeg = masked_eeg.transpose(1, 0)
        mask = mask.transpose(1, 0)

        # Invert the mask
        mask = 1 - mask

        return torch.from_numpy(masked_eeg).float(), torch.from_numpy(raw_eeg_n).float(), torch.from_numpy(mask).bool()
    

class ERPDataset(Dataset):
    # must be numpy array of shape (n_samples, ts_steps) and (n_samples, 1)
    def __init__(self, data_erp_path):
        hf = h5py.File(data_erp_path, 'r')
        self.features = np.array(hf.get("features"))
        self.erp_labels = np.array(hf.get("erp_labels"))

        # Get only the 10000 first samples
        # self.features = self.features[:100000]
        # self.erp_labels = self.erp_labels[:100000]

        hf.close()
        eps = 1e-8 
        for j in range(8):  # Assuming 8 channels
            means = self.features[:, :,j].mean(axis=1, keepdims=True)
            stds = self.features[:, :,j].std(axis=1, keepdims=True)
            self.features[:, :, j] = (self.features[:,:,j] - means) / (stds + eps)
        #self.erp_labels = one_hot_labels(self.erp_labels)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        X = self.features[idx]
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

        self.signals_n = self.signals_n.reshape(-1, 1280, 8)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        X = self.signals_n[idx]
        Y = self.labels[idx]
        Y = int(Y)
        return torch.from_numpy(X).float(), Y

class StressDataset(Dataset):
    def __init__(self, data_path, normalize = None):

        with h5py.File(data_path, 'r') as hf:
           self.signals = np.array(hf.get("signals"))
           self.labels = np.array(hf.get("labels"))
        hf.close()
        self.signals_n = self.signals.copy()

        if normalize == 'normalization':
            self.signals_n = normalization(self.signals, self.signals_n)
        elif normalize == 'min_max':
            self.signals_n = normalize_min_max(self.signals, self.signals_n)
        else:
            self.signals_n = self.signals.copy()

        self.signals_n = self.signals_n.transpose(0,2,1)
        self.signals = self.signals.transpose(0,2,1)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        X = self.signals_n[idx]
        Y = self.labels[idx]
        return torch.from_numpy(X).float(), Y
    
    
if __name__ == "__main__": 
    # Load an ecg of masked dataset and plot mask and the ecg 
    # parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    # raw_csv = os.path.join(parent_dir, 'preprocess_data/tuh_signals_1s.csv')
    # masked_csv = os.path.join(parent_dir, 'preprocess_data/tuh_masked_signals_1s.csv')
    # mask_csv = os.path.join(parent_dir, 'preprocess_data/tuh_mask_signals_1s.csv')
    # dataset = MaskedDataset(raw_csv=raw_csv, masked_csv=masked_csv, mask=mask_csv, nrows= 1000)
    # print(len(dataset))
    # print(dataset[1][0].shape)
    # print(dataset[1][1].shape)
    # print(dataset[1][2].shape)
    # # Plot the 1280 time points of the first channel of the first sample
    # plot_masked_data(dataset[40])
    dataset = StressDataset('deap_stress.h5')
    print(len(dataset))
    sample = dataset[0]
    new_sample = sample[0].transpose(1,0)
    print(sample[0].shape)
    plt.plot(sample[0][:,0])
    plt.plot(new_sample[0,:])
    plt.show()