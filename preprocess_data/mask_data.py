import numpy as np
import os 
import csv 
from matplotlib import pyplot as plt
import h5py
from argparse import ArgumentParser

def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask

def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[0]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[m,:] = geom_noise_mask_single(X.shape[1], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[1], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask

def french_mask(sequence_length, mask_probability, mask_length, num_sequences=1):
    """
    Applies a mask with a given probability and length to a sequence or multiple sequences.

    Parameters:
    - sequence_length: Length of each sequence.
    - mask_probability: Probability of applying the mask at each position.
    - mask_length: Length of the mask to be applied.
    - num_sequences: Number of sequences. Defaults to 1 for a single sequence.

    Returns:
    - A numpy array with the masked sequences. Shape is (num_sequences, sequence_length) for multiple sequences,
      or (sequence_length,) for a single sequence.
    """

    sequences = np.ones((num_sequences, sequence_length))
    for seq_index in range(num_sequences):
        for i in range(sequence_length):
            if np.random.rand() < mask_probability:
                mask_end = min(i + mask_length, sequence_length)
                sequences[seq_index, i:mask_end] = 0

    if num_sequences == 1:
        return sequences[0]
    else:
        return sequences

def apply_mask(sequence, mask):
    """
    Applies a mask to a sequence.

    Parameters:
    - sequence: A numpy array with the sequence to be masked.
    - mask: A numpy array with the mask to be applied. Must have the same length as the sequence.

    Returns:
    - A numpy array with the masked sequence.
    """
    return sequence * mask

if __name__ == '__main__':
    # define de arguments
    data_hf5 = os.path.join('data', 'UVA-DATASET', 'archive', 'GIB-UVA ERP-BCI.hdf5')
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=os.path.join(parent_dir, 'data/tuh_signals_4s.h5'), help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default=os.path.join(parent_dir,'data/TUEH-mask-4s-8channel.h5'), help='Path to save the masked data')
    parser.add_argument('--type_mask', type=str, default='noise', help='Type of mask to apply (noise, french)')

    parser.add_argument('--masking_ratio', type=float, default=0.15, help='Masking ratio')

    parser.add_argument('--mask_probability', type=float, default=0.0325, help='Probability of applying the mask at each position')
    parser.add_argument('--mask_length', type=int, default=20, help='Length of the mask to be applied')
    parser.add_argument('--sequence_length', type=int, default=128, help='Length of each sequence')
    args = parser.parse_args()
 
    dataset_path = args.dataset_path

    # LOAD DATASET
    hf = h5py.File(dataset_path, 'r')
    # Get the names of the groups in the hdf5 file
    print(list(hf.keys()))
    features = np.array(hf.get("data"))
    hf.close()

    print(features.shape)
    features = features.reshape(-1, 8, 512)
    print(features.shape)
    # features = features.transpose(1,0,2)
    # # features = features.transpose(0,2,1)
    # print(features.shape)
   
    if args.type_mask == 'noise':
        # Apply the mask to all the samples
        mask = np.array([noise_mask(sample,args.masking_ratio) for sample in features])
        masked_data = np.array([apply_mask(sample, mask) for sample, mask in zip(features, mask)])
    elif args.type_mask == 'french':
        # features = features.transpose(0,2,1)
        # print(features.shape)
        mask= np.array([french_mask(args.sequence_length, args.mask_probability, args.mask_length, num_sequences=features.shape[1]) for sample in features])
        masked_data = np.array([apply_mask(sample, mask) for sample, mask in zip(features, mask)])

        percentages = []
        for i in mask:
            percentage_masked = np.sum(i == 0) / len(i) * 100
            percentages.append(percentage_masked)
            print("Percentage of masked points: {:.2f}%".format(percentage_masked))

    # Print mean percentage of masked points
        print("Mean percentage of masked points: {:.2f}%".format(np.mean(percentages)))

    # data = features.transpose(0,2,1)
    # masked_data = masked_data.transpose(0,2,1)
    # mask = mask.transpose(0,2,1)
    # masked_data = masked_data.transpose(0,2,1)
    # Save the mask, masked data and the reshaped data in hdf5 format
    data = features
    hf = h5py.File(args.output_path, 'w')
    hf.create_dataset('masked', data=masked_data)
    hf.create_dataset('mask', data=mask)
    hf.create_dataset('eeg', data=data)
    hf.close()




    # # read the csv file
    # parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    # csv_file = os.path.join(parent_dir, 'tuh_signals_1s.csv')
    # data = np.loadtxt(csv_file, delimiter=',')
    # # reshape the data to the original shape
    # data = data.reshape(-1, 8, 128)

    # # Apply the mask to all the samples
    # mask = np.array([noise_mask(sample.transpose(1,0), masking_ratio) for sample in data])
    # masked_data = np.array([apply_mask(sample.transpose(1,0), mask) for sample, mask in zip(data, mask)])
    # print(masked_data.shape)

    # mask_data = apply_mask(data[0].transpose(1,0), mask)

    # mask = mask.transpose(0,2,1)
    # masked_data = masked_data.transpose(0,2,1)



    # # To the first sample, apply the noise mask
    # mask = noise_mask(data[0].transpose(1,0), masking_ratio)

    # save_csv(masked_data, 'tuh_masked_signals_1s.csv')
    # save_csv(mask, 'tuh_mask_signals_1s.csv')
