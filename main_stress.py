import os
import h5py
import numpy as np

import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger  # Import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, GradientAccumulationScheduler
from pytorch_lightning.tuner import Tuner

from data.Dataset import StressDataset
from data.DataModules import ERPDataModule
from models import SSL_EEG, EEG_ERP
from modules.decoders import MaskedDecoder, ERP_decoder
from modules.loss import MaskedMSELoss
from modules.encoders import TSTransformerEncoder
from modules.encoders import ConvNet, Conv1DNet


dataset_path = 'data/deap_stress.h5'

dataset = StressDataset(dataset_path, normalize='normalization')   
dm = ERPDataModule(dataset)

ckpt = os.path.join('checkpoints', 'SSL-1s-v17.ckpt')

# Load the model from the checkpoint
model = SSL_EEG.load_from_checkpoint(ckpt, decoder = MaskedDecoder, loss_fn= MaskedMSELoss)
model.eval()

conv = model.covnet
#Load the encoder from the model
encoder = model.encoder
#Freeze the encoder
for param in encoder.parameters():
    param.requires_grad = False

for param in conv.parameters():
    param.requires_grad = False


encoder_args = {'feat_dim': 16, 'seq_len':512, 'd_model':64, 'n_heads': 1, 'num_layers': 1, 'dim_feedforward': 256, 'dropout': 0.2, 'activation': 'gelu', 'norm': 'LayerNorm'}
encoder1 = TSTransformerEncoder(**encoder_args)
conv1 = Conv1DNet()
model_erp = EEG_ERP(learning_rate = 0.001, convnet = conv1, encoder = encoder, decoder = ERP_decoder(d_model = 64, ts_steps=512))

wandb_logger = WandbLogger(
    name="Classifier Stress",
    project="Classifier Stress",
    log_model='all'
)

# Initialize the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_f1',
    dirpath='./stress_checkpoints/',
    filename='ckpt-{epoch:02d}-{val_f1:.2f}',
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