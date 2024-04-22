import os 
import numpy as np
import h5py 
import pickle
import matplotlib.pyplot as plt

# Load the data
deap_folder = os.path.join(os.getcwd(), 'DEAP')

# Parameters
channels = [19,24,16,11,29,14,32,15]

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

def join_data(): 
    """
    Join the data for all the subjects
    """
    data = []
    for i in range(1, 33):
        subject_data = get_data_subject(i)
        s_data = subject_data['data'][:,:,:7680]
        signals = filter_data_channels(s_data, channels)
        divided = divide_signal(signals, 512)
        data.append(divided)
    return np.concatenate(data, axis=0)

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
    return stress_indices, calm_indices

def get_stress_data(folder):
    """
    Get the data for the stress and calm labels
    """
    stress_total = []
    calm_total = []
    for i in range(1, 33):
        data = get_data_subject(i)
        data = data['data'][:,:,:7680]
        signals = filter_data_channels(data, channels)
        stress_indices, calm_indices = define_stress_labels(i)
        stress_data = signals[stress_indices]
        calm_data = signals[calm_indices]
        stress_divided = divide_signal(stress_data, 512)
        calm_divided = divide_signal(calm_data, 512)
        stress_total.append(stress_divided)
        calm_total.append(calm_divided)
    return np.concatenate(stress_total, axis=0), np.concatenate(calm_total, axis=0)

def join_labels(calm, stress):
    """
    Join the labels for the calm and stress data
    """
    calm_labels = np.zeros(calm.shape[0])
    stress_labels = np.ones(stress.shape[0])
    return np.concatenate([calm_labels, stress_labels], axis=0)



if __name__ == '__main__':
    data = filter_data_channels(get_data_subject(1)['data'][:,:,:7680], channels)
    new_data = divide_signal(data, 512) 
    print('data', data[0])
    print('new',new_data.shape)
    data = join_data()
    print(data.shape)
    with h5py.File('deap_data.h5', 'w') as hf:
        hf.create_dataset("signals", data=data)
    print('Data saved')
    stress, calm = get_stress_data(deap_folder)
    labels = join_labels(calm, stress)
    with h5py.File('deap_stress.h5', 'w') as hf:
        hf.create_dataset("signals", data=np.concatenate([calm,stress], axis=0))
        hf.create_dataset("labels", data=labels)
    