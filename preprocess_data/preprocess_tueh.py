import pyedflib
import os
from matplotlib import pyplot as plt
import numpy as np
import sys 

sys.path.append(os.path.abspath(os.pardir))

from utils import downsample, divide_signal, save_csv, bandpass_filter

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_folder = os.path.join(parent_dir, 'data/TUEH/001/001')

channels = [ 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG T5-REF', 'EEG T6-REF','EEG O1-REF']

def save_csv(signals, path_name='signals_001_128_4s.csv'):
    reshaped_signals = signals.transpose(1, 0, 2).reshape(signals.shape[1], -1)
    print(reshaped_signals.shape)
    path_save = os.path.join(os.getcwd(),path_name)
    np.savetxt(path_save, reshaped_signals, delimiter=',')

def info_eeg_signal(file, channels_list):
    with pyedflib.EdfReader(file) as f:  # Use context manager to ensure proper file closure
        signal_labels_set = set(f.getSignalLabels())
        channels_set = set(channels_list)

        if not channels_set.issubset(signal_labels_set):
            return False

        freqs = []
        signals = []
        for channel in channels_list:  
            index = f.getSignalLabels().index(channel)  
            freqs.append(f.getSampleFrequency(index))  
            signals.append(f.readSignal(index))
            print(f.getSignalHeaders())

        if len(set(freqs)) == 1:
            signals = np.array(signals)
            downsampled_list = [downsample(signal.reshape(1,-1), freq, 128) for signal, freq in zip(signals, freqs)]
            downsampled_signals = np.array(downsampled_list)
            downsampled_signals = downsampled_signals.reshape(8, -1)
            return downsampled_signals
        else:
            return False

def main(): 
    count = 0
    count2 = 0
    count3 = 0
    total_segments = 0

    segments_10_seconds = np.array([])
    for subject in os.listdir(data_folder):
        for session in os.listdir(os.path.join(data_folder, subject)):
            for eeg in os.listdir(os.path.join(data_folder, subject, session)):
                for file in os.listdir(os.path.join(data_folder, subject, session, eeg)):
                    #print(file)
                    if file.endswith('.edf'):
                        boolean = info_eeg_signal(os.path.join(data_folder, subject, session, eeg, file), channels)
                        if boolean is not False:

                            count2 += 1
                            signals = boolean
                            signals = np.array(signals)

                            freq = 128
                            seg_length = 4* freq
                            divided_signals = divide_signal(signals, seg_length)
                            divided_signals = bandpass_filter(divided_signals, 0.5, 45, 128)
    
                            total_segments += divided_signals.shape[1]
                            segments_10_seconds = np.concatenate((segments_10_seconds, divided_signals), axis=1) if segments_10_seconds.size else divided_signals

                    count += 1

    # print(segments_10_seconds.shape)
    # save_csv(segments_10_seconds)


if __name__ == '__main__':
    main()