import os 

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger  # Import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, GradientAccumulationScheduler
from pytorch_lightning.tuner import Tuner
import wandb

from data.DataModules import SSLDataModule
from data.Dataset import MaskedDataset, MaskedDataset1, MaskedDataset2
from models import SSL_EEG
from modules.decoders import MaskedDecoder, ERP_decoder
from modules.loss import MaskedMSELoss

"""
This script is used to train the pre-trained model. The pretext task is to predict the masked EEG signals.
"""
# DESCRIPTION:
 # Script for training the downstream task. It loads the encoder from the pre-trained model and freezes it. It trains the classifier
 # with the encoder as input.

def main():
    wandb.init(project = 'SSL EEG')

    # raw_csv = os.path.join(os.getcwd(), 'preprocess_data/tuh_signals.csv')
    # masked_csv = os.path.join(os.getcwd(), 'preprocess_data/tuh_masked_signals.csv')
    # mask_csv = os.path.join(os.getcwd(), 'preprocess_data/tuh_mask_signals.csv')
    # masked_dataset = MaskedDataset(raw_csv=raw_csv, masked_csv=masked_csv, mask=mask_csv, nrows= 40000, seq_length=1280, normalize='normalization')
    # # Create the data module with batch_size from wandb.config
    # datamodule = SSLDataModule(dataset = masked_dataset,
    #     batch_size=32,
    # )

    eeg = os.path.join(os.getcwd(), 'data/TUEH-mask.h5')
    masked_dataset = MaskedDataset2(hdf5_file=eeg, normalize='normalization')
    datamodule = SSLDataModule(dataset = masked_dataset,
        batch_size=32,
    )

    encoder_args = {'feat_dim': 8, 'seq_len':512, 'd_model':64, 'n_heads': 8, 'num_layers': 1, 'dim_feedforward': 256, 'dropout': 0.1, 'activation': 'gelu', 'norm': 'LayerNorm'}
    pretext = SSL_EEG(learning_rate=1e-3, encoder_args=encoder_args, decoder = MaskedDecoder, loss_fn=MaskedMSELoss())

    wandb_logger = WandbLogger(
        name="SSL EEG",
        project="SSL EEG",
        log_model='all'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints/',
        filename='SSL-1s',
        save_top_k=1,
        mode='min',
    )

    trainer = Trainer(
        max_epochs=100,  # You can adjust this as needed
        accelerator='gpu',
        devices=1,
        precision="16-mixed",
        #accumulate_grad_batches=8,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )

    trainer.fit(pretext, datamodule=datamodule)

if __name__ == "__main__":
    main()
