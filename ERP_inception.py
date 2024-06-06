import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, GradientAccumulationScheduler, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt

from models import EEGInceptionPL

"""
This script is used to train the ERP detection model. The model is trained using the EEG-Inception architecture.
This architecture is the one used by the UVA university to detect ERPs in EEG signals.
"""

dataset_path = os.path.join(os.getcwd(), 'data/UVA-DATASET/archive/GIB-UVA ERP-BCI.hdf5')

# LOAD DATASET
hf = h5py.File(dataset_path, 'r')
features = np.array(hf.get("features"))
erp_labels = np.array(hf.get("erp_labels"))
codes = np.array(hf.get("codes"))
trials = np.array(hf.get("trials"))
sequences = np.array(hf.get("sequences"))
matrix_indexes = np.array(hf.get("matrix_indexes"))
run_indexes = np.array(hf.get("run_indexes"))
subjects = np.array(hf.get("subjects"))
database_ids = np.array(hf.get("database_ids"))
target = np.array(hf.get("target"))
matrix_dims = np.array(hf.get("matrix_dims"))
hf.close()

# # Normalize each channel independently with z schore normization 
def min_max_normalization(data):
    # Copy the data to avoid modifying the original array in place
    normalized_data = np.copy(data)
    
    # Normalize each channel in each sample
    for sample in range(data.shape[0]):
        for channel in range(data.shape[2]):
            # Select the data for the current sample and channel
            channel_data = data[sample, :, channel]
            
            # Compute the min and max of this channel's data
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)

            # Normalize the data
            normalized_data[sample, :, channel] = (channel_data - min_val) / (max_val - min_val)
    return normalized_data

def z_score_normalization(data):
    # Normalize the data per channel
    normalize_data = np.copy(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            normalize_data[i, :, j] = (data[i, :, j] - np.mean(data[i, :, j])) / np.std(data[i, :, j])
    return normalize_data


features = z_score_normalization(features)
# plt.plot(features[0, :,0])
# plt.show()

# PREPARE FEATURES AND LABELS
features = features.reshape(
    (features.shape[0],1, features.shape[1],
     features.shape[2])
    )   

print(features.shape)
# Prepare the data
labels_tensor = torch.tensor(erp_labels, dtype=torch.float32)
features_tensor = torch.tensor(features, dtype=torch.float32)

total_samples = len(features_tensor)
split_index = int(total_samples * 0.8)

features_train = features_tensor[:split_index]
labels_train = labels_tensor[:split_index]
features_val = features_tensor[split_index:]
labels_val = labels_tensor[split_index:]

train_dataset = TensorDataset(features_train, labels_train)
val_dataset = TensorDataset(features_val, labels_val)

# Prepare the LightningDataModule to handle data loading
class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=1024):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# Initialize your LightningDataModule
data_module = MyDataModule(train_dataset, val_dataset)

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Validation dataset: {len(val_dataset)} samples")

# Prepare to train the model
model = EEGInceptionPL(pretrained_path=None, learning_rate=0.001, n_classes=1)
# model.train_dataset = train_dataset
# model.val_dataset = val_dataset

wandb_logger = WandbLogger(
        name="Classifier",
        project="EEG Inception",
        log_model='all'
    )

checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath='./erp_checkpoints/',
        filename='EEGInceptionPL',
        mode='max',
    )

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=10,
    verbose=True,
    mode='min'
)

trainer = Trainer(
        max_epochs=100,  # You can adjust this as needed
        accelerator='gpu',
        devices=1,
        #precision="16-mixed",
        accumulate_grad_batches=8,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger
    )

trainer.fit(model, data_module)