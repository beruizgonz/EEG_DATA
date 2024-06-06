import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger  
import matplotlib.pyplot as plt
from models import EEGInceptionPL

"""
This script is used to train a model to predict an emotion. The model is trained using the EEG-Inception architecture.
This architecture is the one used by the UVA university to detect ERPs in EEG signals.
"""

dataset_path = os.path.join(os.getcwd(), 'data/combined_data.h5')

# LOAD DATASET
hf = h5py.File(dataset_path, 'r')
features = np.array(hf.get("signals"))
labels = np.array(hf.get("labels"))
hf.close()

print(features.shape)
# PREPARE FEATURES AND LABELS
features = features.transpose(0, 2, 1)

print(features.shape)

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
plt.plot(features[0, :,0])
plt.show()

features = features.reshape(
    (features.shape[0],1, features.shape[1],
     features.shape[2])
)

print(features.shape)
# Prepare the data
print(labels.dtype)
labels_tensor = torch.tensor(labels, dtype=torch.float32)
features_tensor = torch.tensor(features, dtype=torch.float32)

total_samples = len(features_tensor)
split_index = int(total_samples * 0.9)

features_train = features_tensor[:split_index]
labels_train = labels_tensor[:split_index]
features_val = features_tensor[split_index:]
labels_val = labels_tensor[split_index:]

train_dataset = TensorDataset(features_train, labels_train)
val_dataset = TensorDataset(features_val, labels_val)

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Validation dataset: {len(val_dataset)} samples")

pretrained_weights_path = 'erp_checkpoints/EEGInceptionPL-v8.ckpt'

# Initialize your model
model = EEGInceptionPL(pretrained_path=pretrained_weights_path, learning_rate=0.001, n_classes=1)

# Load pre-trained weights
#pretrained_dict = torch.load(pretrained_weights_path)
#model_pre = EEGInceptionPL.load_from_checkpoint(pretrained_weights_path)
# Prepare to train the model
batch_size = 32  # You can adjust this value as needed

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

wandb_logger = WandbLogger(
        name="Classifier",
        project="EEG Inception",
        log_model='all'
    )

checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath='./emotion_checkpoints/',
        filename='EEGInceptionPL',
        mode='max',
    )

early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0.0001,
    mode='min', patience=10, verbose=1,
    )

trainer = Trainer(
        max_epochs=500,  # You can adjust this as needed
        accelerator='gpu',
        devices=1,
        #precision="16-mixed",
        accumulate_grad_batches=16,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )

trainer.fit(model, train_loader, val_loader)