import os 
import torch.nn as nn

import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger  # Import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, GradientAccumulationScheduler
from pytorch_lightning.tuner import Tuner

from data.DataModules import SSLDataModule, ERPDataModule
from data.Dataset import MaskedDataset, EmotionDataset
from models import SSL_EEG, EEG_Emotion
from modules.decoders import MaskedDecoder, Emotion_decoder
from modules.loss import MaskedMSELoss
from modules.encoders import ConvNet

from utils import normalization, normalize_min_max

ckpt = os.path.join('checkpoints', 'SSL-10s-v1.ckpt')

# Load the model from the checkpoint
model = SSL_EEG.load_from_checkpoint(ckpt, learning_rate = 1e-3, decoder = MaskedDecoder, loss_fn= nn.MSELoss())
model.eval()

conv = model.covnet
encoder = model.encoder

for param in encoder.parameters():
    param.requires_grad = False

for param in conv.parameters():
    param.requires_grad = False


model_emotion = EEG_Emotion(learning_rate=0.0001, convnet = conv, encoder = encoder, decoder = Emotion_decoder(d_model = 64, ts_steps=1280, n_emotions=5,dropout=0.2))

# Load the dataset
seed_dataset = os.path.join(os.getcwd(), 'data', 'seed_128.h5')
dataset = EmotionDataset(seed_dataset, normalize='normalization')
dm = ERPDataModule(dataset)

wandb_logger = WandbLogger(
    name="Classifier Emotion",
    project="Emotion EEG",
    log_model='all'
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_f1',
    dirpath='./emotion_checkpoints/',
    filename='Classifier-{epoch:02d}-{val_f1:.2f}',
    save_top_k=1,
    mode='max',
)

acc_gradient = GradientAccumulationScheduler(
    scheduling={0: 8}
)

trainer = Trainer(
    max_epochs=500,  # You can adjust this as needed
    accelerator='gpu',
    devices=1,
    precision="16-mixed",
    accumulate_grad_batches=8,
    callbacks=[checkpoint_callback],
    logger=wandb_logger
)

trainer.fit(model_emotion, datamodule=dm)