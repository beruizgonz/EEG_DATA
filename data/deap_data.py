import os 
import numpy as np
import h5py 
import pickle
import matplotlib.pyplot as plt
import sys 

sys.path.append(os.path.abspath(os.pardir))
from utils import bandpass_filter, downsample, downsampled_signals

# Load the data
deap_folder = os.path.join(os.getcwd(), 'DEAP')

# Parameters
channels = [19,24,16,11,29,14,32,15]
channels1 = [19,24,16,15]
channels2 =[24]

def get_data_subject(subject): 
    """
    Get the data for a single subject
    """
    subject_data = os.path.join(deap_folder, f's{subject:02d}.dat')
    with open(subject_data, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

def filter_data_channels(data, channels):
    """
    Filter the data to only include the specified channels
    """
    return data[:, channels, :]


def divide_signal(signal, seg_length):
    """
    Divide the signal in segments of seg_length. I put the number of segments in the first dimension
    """
    trials, channels, time_points = signal.shape
    n_segments = time_points // seg_length
    segmented_signal = signal.reshape(trials* n_segments, channels, seg_length)
    return segmented_signal

def valence_arousal(subject):
    """
    Get valence and arousal labels for a single subject
    """
    subject_data = os.path.join(deap_folder, f's{subject:02d}.dat')
    with open(subject_data, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    valence = data['labels'][:, 0]
    arousal = data['labels'][:, 1]
    return valence, arousal

def define_stress_labels(subject):
    """
    Define the stress labels for a single subject. Get the indices of the stress and calm labels
    """
    valence, arousal = valence_arousal(subject)
    stress = np.zeros_like(valence)
    stress[(valence < 3) & (arousal > 5)] = 1

    calm = np.zeros_like(valence)
    calm[ (arousal <4) & (valence < 6) & (valence > 4)]= 1
    # Get the indices of the stress and calm labels
    stress_indices = np.where(stress == 1)[0]
    calm_indices = np.where(calm == 1)[0]
    print(stress_indices+1, calm_indices+1)
    return stress_indices, calm_indices

def get_stress_data1(base_folder, num_subjects=32):
    stress_total = {}
    calm_total = {}
    for i in range(1, num_subjects + 1):
        try:
            data = get_data_subject(i)  # Assuming function adapted for path
            data = data['data'][:,:, 384:]
            #signals = data
            signals = filter_data_channels(data, channels1)
            #signals = downsampled_signals(signals, 256, 128)
            #signals = bandpass_filter(signals, 4, 45, 128)
            stress_indices, calm_indices = define_stress_labels(i)
            if len(stress_indices) == 0 or len(calm_indices) == 0:
                print(f"Skipping subject {i} due to lack of indices")
                continue
            stress_data = signals[stress_indices]
            calm_data = signals[calm_indices]
            stress_divided = divide_signal(stress_data, 512)
            calm_divided = divide_signal(calm_data, 512)
            stress_total[i] = stress_divided
            calm_total[i] = calm_divided
        except Exception as e:
            print(f"Failed processing subject {i}: {e}")
    return stress_total, calm_total

def join_data_and_labels(dict_calm, dict_stress):
    """
    Join the data from multiple subjects, each with calm and stress data, into a single dataset
    with corresponding labels.
    """
    combined_data = []
    combined_labels = []
    subject_ids = []
    print(dict_calm.keys(), dict_stress.keys())
    for subject_id in dict_calm.keys():

        calm_data = dict_calm[subject_id]
        print(calm_data.shape, len(calm_data))
        stress_data = dict_stress[subject_id]

        combined_data.append(calm_data)
        combined_data.append(stress_data)

        calm_labels = np.zeros(calm_data.shape[0])
        stress_labels = np.ones(stress_data.shape[0])

        combined_labels.append(calm_labels)
        combined_labels.append(stress_labels)

        #subject_ids.append(np.full(len(calm_data), subject_id))
        subject_ids.append(np.full(len(stress_data), subject_id))

    # Concatenate all subject data, labels, and subject IDs into single arrays
    final_data = np.concatenate(combined_data, axis=0)
    final_labels = np.concatenate(combined_labels, axis=0)
    final_subject_ids = np.concatenate(subject_ids, axis=0)

    return final_data, final_labels, final_subject_ids

def valence_arousal_data(folder):
    """
    Get the valence and arousal labels for all the subjects
    """
    valence_total = []
    arousal_total = []
    combined_data = []
    for i in range(1, 33):
        data = get_data_subject(i)
        data = data['data'][:,:,384:]
        signals = filter_data_channels(data, channels1)
        valence, arousal = valence_arousal(i)
        divided = divide_signal(signals, 512)
        
        valence_total.append(np.repeat(valence, divided.shape[0]//valence.shape[0]))
        arousal_total.append(np.repeat(arousal, divided.shape[0]//arousal.shape[0]))
        combined_data.append(divided)

    valence_total = np.concatenate(valence_total, axis=0)
    arousal_total = np.concatenate(arousal_total, axis=0)
    divided = np.concatenate(combined_data, axis=0)
    print(valence_total.shape)
    print(arousal_total.shape)
    print(divided.shape)
    with h5py.File('deap_valence_arousal1.h5', 'w') as hf:
        hf.create_dataset("valence", data=valence_total)
        hf.create_dataset("arousal", data=arousal_total)
        hf.create_dataset("signals", data=divided)

if __name__ == '__main__':
    # data = filter_data_channels(get_data_subject(1)['data'][:,:,:7680], channels)
    # new_data = divide_signal(data, 512) 
    # print('data', data[0])
    # print('new',new_data.shape)
    # data = join_data()
    # print(data.shape)
    # with h5py.File('deap_data.h5', 'w') as hf:
    #     hf.create_dataset("signals", data=data)
    # print('Data saved')
    stress, calm = get_stress_data1(deap_folder)
    data, labels, subjects = join_data_and_labels(calm, stress)
    with h5py.File('deap_stress_4s_4channel.h5', 'w') as hf:
        hf.create_dataset("signals", data=data)
        hf.create_dataset("labels", data=labels)
        hf.create_dataset("subject", data=subjects)
    # # subject = 's01'
    # data = get_data_subject(1)
    # # Get the data
    # data = data['data'][:,:,:7680]
    # signals = filter_data_channels(data, channels)
    
    # # Plot the first channel of the first trial
    # plt.plot(data[0, 32, :128])
    # plt.plot(signals[0, 6, :128])
    # plt.show()
    #valence_arousal_data(deap_folder)