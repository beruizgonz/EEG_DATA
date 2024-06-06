import os 
import numpy as np
import mne
from scipy.signal import resample, butter, lfilter
import pandas as pd 
import matplotlib.pyplot as plt
import h5py
import sys 

parent_dir = os.path.abspath(os.pardir)
sys.path.append(parent_dir)

from utils import bandpass_filter, downsampled_signals, divide_signal

seed_path = os.path.join(os.getcwd(), 'SEEDV')
labels_file =  os.path.join(seed_path, 'information/Scores.xlsx')
use_channels = ['FZ', 'CZ', 'PZ', 'P3', 'P4','PO7','PO8','OZ']
ignore_file = '7_1_20180411.cnt'

start_second = [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]
end_second = [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]

trial_1 = [4,1,3,2,0]
trial_2_3 = [2,1,3,0,4,4,0,3,2,1,3,4,1,2,0]

def info_samples(start_second, end_second):
    samples = []
    for i in range(len(start_second)):
        samples.append((end_second[i] - start_second[i]) / 10)
    return sum(samples)

def info_eeg_file(file_path):
    eeg_raw = mne.io.read_raw_cnt(file_path)
    info = eeg_raw.info
    # frequency
    sfreq = info['sfreq']
    return sfreq

def time_seconds(file_path,freq):
    eeg_raw = mne.io.read_raw_cnt(file_path)
    data = eeg_raw.get_data()
    time = data.shape[1]/freq
    return time

def drop_channels(file_path,use_channels):
    eeg_raw = mne.io.read_raw_cnt(file_path)
    eeg_raw = eeg_raw.drop_channels([ch for ch in eeg_raw.ch_names if ch not in use_channels])
    return eeg_raw

def get_score(score_file, index, patient, trial):
    scores = pd.read_excel(score_file)
    row = (patient - 1) * 3 +trial - 1
    column = index + 2
    score = scores.iloc[row, column]
    return score

def get_label(index, trial):
    number = index % 5
    if trial == 1:
        label = trial_1[number]
        #print('index:', index, 'label:', label)
    else:
        label = trial_2_3[index]
    return label

def label_stress(score):
    if score >= 2:
        return 1
    else:
        return 0

def discard_seconds(number_seconds, freq, signals):
    return signals[:, int(number_seconds * freq):]

def get_eeg_data(file_path, start_second, end_second, use_channels, trial):
    sfreq= info_eeg_file(file_path)  # Assuming this initializes and loads the EEG file
    eeg_raw = drop_channels(file_path, use_channels)  # Assuming drop_channels now accepts eeg_raw object
    eeg_raw = eeg_raw.reorder_channels(use_channels)
    data = eeg_raw.get_data()
    all_divided_signals = []
    labels = []
    for i in range(len(start_second)):
        score = get_score(labels_file, i, patient, trial)
        if score >=3 and trial == 1:    
            label = get_label(i, trial)
            label = label_stress(label)
            index_start = int(start_second[i] * sfreq)
            index_end = int(end_second[i] * sfreq)
            data_trial = data[:, index_start:index_end]
            if data_trial.shape[1] == 0:
                print('Error: Empty segment')
                continue
            data_trial = downsampled_signals(data_trial, sfreq, 128)
            data_trial = bandpass_filter(data_trial, 0.5, 45, 128)
            data_trial = discard_seconds(60, 128, data_trial)
            divided_signals = divide_signal(data_trial, 128)
            divided_signals = divided_signals.reshape(-1, 8, 128)
            for divided_signal in divided_signals:
                all_divided_signals.append(divided_signal)
                labels.append(label)
    return np.array(labels), np.array(all_divided_signals)

def shuffle_singals(signals, labels):
    # Generate shuffled indices
    paired_data = list(zip(signals, labels))
    np.random.shuffle(paired_data)
    shuffled_signals, shuffled_labels = zip(*paired_data)

    # Convert the shuffled features and labels back to numpy arrays if needed
    shuffled_signals = np.array(shuffled_signals)
    shuffled_labels = np.array(shuffled_labels)

    return shuffled_signals, shuffled_labels

def save_h5py(signals, labels, filename):
    with h5py.File (filename, 'w') as f:
        f.create_dataset('signals', data=signals)
        f.create_dataset('labels', data=labels)
        print(f"Shuffled data written to {filename}")

if __name__ == "__main__":
    joined_signals_list = []
    labels_list = []  
    for root, dirs, files in os.walk(seed_path):
        for file in files:
            print(file)
            if file.endswith('.cnt') and file != ignore_file:
                file_path = os.path.join(root, file)
                divide = file.split('_')
                patient = int(divide[0])    
                trial = int(divide[1])
                labels, divided_signals = get_eeg_data(file_path, start_second, end_second, use_channels, trial)
                joined_signals_list.extend(divided_signals)
                labels_list.extend(labels)
    
    print('Total segments:', len(joined_signals_list))
  
    save_h5py(joined_signals_list, labels_list, 'seed_stress_1s_b1.h5')