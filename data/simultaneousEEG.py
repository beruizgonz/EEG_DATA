import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# Directory paths
data_path = os.path.join(os.getcwd(), 'Simultaneous EEG-fMRI dataset/Simultaneous-EEG-fMRI/BIDS_dataset_EEG')
subject_path = os.path.join(data_path, 'sub-001')

# Load the data
file_path = os.path.join(subject_path, 'eeg', 'sub-001_task-fmrieoec_eeg.set')
fdt_file = os.path.join(subject_path, 'eeg', 'sub-001_task-fmrieoec_eeg.fdt')
# # Load EEG data
# eeg_data = mne.io.read_raw_eeglab(file_path, preload=True)

# # Load markers (events) if available
# events = mne.find_events(eeg_data)


EEG = sio.loadmat(file_name=file_path, mat_dtype=True)

print(EEG)
# # plot the data
# plt.figure()
# plt.plot(eeg_data)
# plt.show()


# Define the data type based on your EEG data
# For example, if your data is float32, you can use np.float32
data_type = np.float32

# Load the data from the .fdt file
eeg_data = np.fromfile(fdt_file, dtype=data_type)
print(eeg_data)

# If you know the shape of the data (e.g., number of channels and time points),
# reshape the array accordingly
# For example, if you have 64 channels and 10000 time points:
# num_channels = 64
# num_samples = 10000
# eeg_data = eeg_data.reshape(num_channels, num_samples)