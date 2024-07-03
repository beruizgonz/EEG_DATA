import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import mne
import re
import sys
import h5py

sys.path.append(os.path.abspath(os.pardir))
from utils import bandpass_filter, downsample, divide_signal

# Path to the folder containing the EEG and VMRK files
data_path = os.path.join(os.getcwd(), 'LEMON')
subject_path = os.path.join(data_path, 'sub-032306/RSEEG')
vhdr_file = os.path.join(subject_path, 'sub-032306.vhdr')

interest_channels = ['Fz', 'Cz', 'Pz', 'P3', 'P4', 'PO7', 'PO8', 'Oz']
interest_channels1 = ['Fz', 'Cz', 'Pz', 'Oz']
interest_channels2 = ['Cz']

def change_vmrkeeg_marker_file(vhdr_file_path, new_marker_file_name):
    name_file = os.path.basename(vhdr_file_path).split('.')[0]

    # Load the content of the VHDR file
    with open(vhdr_file_path, 'r') as vhdr_file:
        vhdr_content = vhdr_file.read()

    # Change the EEG and marker file names in the VHDR content
    vhdr_content = re.sub(r'DataFile=.*?\n', f'DataFile={name_file}.eeg\n', vhdr_content)
    vhdr_content = re.sub(r'MarkerFile=.*?\n', f'MarkerFile={name_file}.vmrk\n', vhdr_content)

    # Write the modified content to a new VHDR file
    new_vhdr_file_path = os.path.join(os.path.dirname(vhdr_file_path), new_marker_file_name)
    with open(new_vhdr_file_path, 'w') as new_vhdr_file:
        new_vhdr_file.write(vhdr_content)

    # Load the EEG data and events
    raw = mne.io.read_raw_brainvision(new_vhdr_file_path, preload=True)
    #return raw.get_data()


def lemon_eeg_folder(folder_path):
    # Do it for all the folders in the MAX folder
    for folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, folder)):
            subject_path = os.path.join(data_path, folder, 'RSEEG')
            vhdr_file = os.path.join(subject_path, f'{folder}.vhdr')
            change_vmrkeeg_marker_file(vhdr_file, f'{folder}_new.vhdr')

def get_data_from_vhdr(vhdr_file):
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
    return raw

def filter_eeg_channels(raw, interest_channels):
    # Get the indices of the channels of interest
    picks = mne.pick_channels(raw.info['ch_names'], interest_channels)

    # Filter the raw data to keep only the channels of interest
    raw.pick_channels([raw.ch_names[pick] for pick in picks])

    return raw

def downsample_eeg_data(raw, fs_new):
    raw.resample(fs_new)

def prepare_data(raw):
    # Get the time points of the raw data
    raw_np1 = raw.get_data()
    raw_np = downsample_eeg_data(raw, 128)
    raw_np = raw.get_data()
    raw_filtered = bandpass_filter(raw_np, 4, 45, 128)
    divided_signals = divide_signal(raw_filtered, 512)
    return divided_signals

def main(folder):
    all_divided_signals = []
    for sub_folder in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, sub_folder)):
            subject_path = os.path.join(folder, sub_folder, 'RSEEG')
            vhdr_file = os.path.join(subject_path, f'{sub_folder}.vhdr')
            change_vmrkeeg_marker_file(vhdr_file, f'{sub_folder}_new.vhdr')
            vhdr_new = os.path.join(subject_path, f'{sub_folder}_new.vhdr')
            raw = get_data_from_vhdr(vhdr_new)
            raw = filter_eeg_channels(raw, interest_channels)
            divided_signals = prepare_data(raw)
            # return to the original folder
            os.chdir(folder)
            print(divided_signals.shape)
            all_divided_signals.append(divided_signals)
    
    return np.concatenate(all_divided_signals, axis=1)
            

if __name__ == '__main__':
    # change_vmrkeeg_marker_file(vhdr_file, 'sub-032306_new.vhdr')
    # lemon_eeg_folder(data_path)
    # raw = get_data_from_vhdr(vhdr_file)
    # raw = filter_eeg_channels(raw, interest_channels)
    # divided_signals = prepare_data(raw)
    # # plot the 0th channel
    # print(divided_signals.shape)
    # # plt.plot(divided_signals[0, 0, :])
    # # plt.show()
    all_lemon = main(data_path)
    print(all_lemon.shape)
    with h5py.File('all_lemon_4s_8channel.h5', 'w') as hf:
        hf.create_dataset("data", data=all_lemon)
