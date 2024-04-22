import os
import h5py
import numpy as np

import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger  # Import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, GradientAccumulationScheduler
from pytorch_lightning.tuner import Tuner

from data.Dataset import ERPDataset
from data.DataModules import ERPDataModule
from models import SSL_EEG, EEG_ERP
from modules.decoders import MaskedDecoder, ERP_decoder
from modules.loss import MaskedMSELoss
from modules.encoders import ConvNet, TSTransformerEncoder

"""
This script is used to train the ERP detection model. 
It loads the encoder from the pre-trained model and freezes it. It trains the classifier.
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

dataset = ERPDataset(dataset_path)
dm = ERPDataModule(dataset)

ckpt = os.path.join('checkpoints', 'SSL-1s-v1.ckpt')

# Load the model from the checkpoint
model = SSL_EEG.load_from_checkpoint(ckpt, decoder = MaskedDecoder, loss_fn= MaskedMSELoss)
model.eval()

conv = model.covnet
# Load the encoder from the model
encoder = model.encoder
# Freeze the encoder
for param in encoder.parameters():
    param.requires_grad = False

for param in conv.parameters():
    param.requires_grad = False

encoder_args = {'feat_dim': 8, 'seq_len':128, 'd_model':64, 'n_heads': 8, 'num_layers': 1, 'dim_feedforward': 256, 'dropout': 0.1, 'activation': 'gelu', 'norm': 'LayerNorm'}
encoder1 = TSTransformerEncoder(**encoder_args)

model_erp = EEG_ERP(learning_rate = 0.001,convnet = conv, encoder = encoder1, decoder = ERP_decoder(d_model = 64, ts_steps=128))

wandb_logger = WandbLogger(
    name="Classifier ERP",
    project="ERP detection",
    log_model='all'
)

# Initialize the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_f1',
    dirpath='./erp_checkpoints/',
    filename='Classifier-{epoch:02d}-{val_f1:.2f}',
    save_top_k=1,
    mode='max',
)

# Initialize Gradient Accumulation Callback
acc_gradient = GradientAccumulationScheduler(
    scheduling={0: 8}
)

# Initialize the trainer with hyperparameters from wandb.config
trainer = Trainer(
    max_epochs=100,  # You can adjust this as needed
    accelerator='gpu',
    devices=1,
    precision="16-mixed",
    #accumulate_grad_batches=8,
    callbacks=[checkpoint_callback],
    logger=wandb_logger
)

# Train the model
trainer.fit(model_erp, datamodule=dm)