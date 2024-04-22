
import numpy as np
import h5py, os
from EEGInception import EEGInception
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder

"""
Usage example of EEG-Inception with an ERP-based BCI dataset from:


Download the dataset from:
https://www.kaggle.com/esantamaria/gibuva-erpbci-dataset

"""

parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
dataset_path = os.path.join(parent_path, 'data/UVA-DATASET/archive/GIB-UVA ERP-BCI.hdf5')
seed_path = os.path.join(parent_path, 'data/seed_128.h5')

input_time = 10000
fs = 128
n_cha = 8
filters_per_branch = 8
scales_time = (500, 250, 125)
dropout_rate = 0.25
activation = 'elu'
n_classes = 5
learning_rate = 0.001

# hf = h5py.File(dataset_path, 'r')
# features = np.array(hf.get("features"))
# erp_labels = np.array(hf.get("erp_labels"))
# codes = np.array(hf.get("codes"))
# trials = np.array(hf.get("trials"))
# sequences = np.array(hf.get("sequences"))
# matrix_indexes = np.array(hf.get("matrix_indexes"))
# run_indexes = np.array(hf.get("run_indexes"))
# subjects = np.array(hf.get("subjects"))
# database_ids = np.array(hf.get("database_ids"))
# target = np.array(hf.get("target"))
# matrix_dims = np.array(hf.get("matrix_dims"))
# hf.close()

# LOAD DATASET
hf = h5py.File(seed_path, 'r')
features = np.array(hf.get("signals"))
erp_labels = np.array(hf.get("labels"))


features = features.reshape(
    (features.shape[0], features.shape[2],
     features.shape[1], 1)
)


# # Generate shuffled indices
# indices = np.arange(features.shape[0])
# np.random.shuffle(indices)

# # Use the shuffled indices to reorder both features and labels
# shuffled_features = features[indices]
# shuffled_labels = erp_labels[indices]
# One hot encoding of labels
def one_hot_labels(caategorical_labels):
    enc = OneHotEncoder(handle_unknown='ignore')
    on_hot_labels = enc.fit_transform(
        caategorical_labels.reshape(-1, 1)).toarray()
    return on_hot_labels


train_erp_labels = one_hot_labels(erp_labels)

# TRAINING
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Create model
model = EEGInception(
    input_time=10000, fs=128, ncha=8, filters_per_branch=8,
    scales_time=(500, 250, 125), dropout_rate=0.25,
    activation='elu', n_classes=5, learning_rate=0.001)

# Print model summary
model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0.0001,
    mode='min', patience=10, verbose=1,
    restore_best_weights=True)

# Fit model
fit_hist = model.fit(features,
                     train_erp_labels,
                     epochs=100,
                     batch_size=64,
                     validation_split=0.2,
                     #callbacks=[early_stopping])
)      
# Save
model.save('model')