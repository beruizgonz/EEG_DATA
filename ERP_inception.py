import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, GradientAccumulationScheduler
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

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

# PREPARE FEATURES AND LABELS
features = features.reshape(
    (features.shape[0],1, features.shape[1],
     features.shape[2])
    )   

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

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Validation dataset: {len(val_dataset)} samples")

# Prepare to train the model
model = EEGInceptionPL()
model.train_dataset = train_dataset
model.val_dataset = val_dataset

wandb_logger = WandbLogger(
        name="Classifier",
        project="EEG Inception",
        log_model='all'
    )

checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath='./checkpoints/',
        filename='EEGInceptionPL',
        mode='max',
    )

trainer = Trainer(
        max_epochs=100,  # You can adjust this as needed
        accelerator='gpu',
        devices=1,
        #precision="16-mixed",
        accumulate_grad_batches=8,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )

trainer.fit(model)