
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule, LightningDataModule
from typing import List
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
import torch

from modules.encoders import TSTransformerEncoder, ConvNet, Conv1DNet
from modules.decoders import MaskedDecoder
from modules.inception_modules import EEGInceptionBlock1, EEGInception
from modules.loss import MaskedMSELoss

"""
This script define all the models used in the project. The models are defined using PyTorch Lightning.
"""

class SSL_EEG(LightningModule):
    """
    This class defines the model used for the self-supervised learning task. The model is used to predict the masked EEG signals.
    """
    def __init__(self,  
                 learning_rate: float,
                 encoder_args: dict, 
                 decoder: nn.Module, 
                 loss_fn: nn.Module
                 ):
        super().__init__()
        self.save_hyperparameters( 
            "learning_rate", "encoder_args")      

        self.learning_rates = learning_rate

        self.encoder = TSTransformerEncoder(
            feat_dim=encoder_args["feat_dim"],
            seq_len=encoder_args["seq_len"],
            d_model=encoder_args["d_model"],
            n_heads=encoder_args["n_heads"],
            num_layers=encoder_args["num_layers"],
            dim_feedforward=encoder_args["dim_feedforward"],
            dropout=encoder_args["dropout"],
            activation=encoder_args["activation"],
            norm=encoder_args["norm"]
        )

        #self.covnet = EEGInceptionBlock1(in_channels = 1, ncha=8, filters_per_branch=8, scales_samples=(500, 250, 125), dropout_rate=0.25, activation='ELU')
        #self.covnet = Conv1DNet()
        self.decoder = decoder(d_model=64, feat_dim = 8)
        self.loss_fn = loss_fn
        
        
    def forward(self, x):

        # x = x.reshape(x.shape[0],1, x.shape[1],
        #   x.shape[2])
        
        #x = self.covnet(x)

        # x = x.squeeze(-1)
        x, attn = self.encoder(x)
        x =  self.decoder(x)
        return x, attn

    def shared_step(self, batch, stage):
        masked, raw, mask = batch
        output, attn = self(masked)
        loss = self.loss_fn(output, raw,mask)
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer =  optim.Adam(self.parameters(), lr=self.learning_rates, betas=(0.9, 0.999))
        return optimizer
    
class EEG_ERP(LightningModule):
    """
    This class defines the model used for the ERP detection task. The model is used to detect the ERP signals.
    It uses the encoder from the pre-trained model. 
    """
    def __init__(self,  
                 learning_rate: float,
                convnet: nn.Module,
                encoder: nn.Module,
                decoder: nn.Module
                 ):
        super().__init__()
        self.metrics = {"accuracy": torchmetrics.Accuracy(task="binary").to("cuda"),
                        "f1": torchmetrics.F1Score(task="binary").to("cuda"),
                        "AUC": torchmetrics.AUROC(task="binary").to("cuda")}
       
        self.learning_rate = learning_rate
        self.covnet = convnet
        self.encoder = encoder
        self.decoder = decoder
        positive_weight = torch.tensor([0.83 / 0.17])
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        #x = self.covnet(x)
        x, _ = self.encoder(x)
        x =  self.decoder(x)
        return x

    def shared_step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        logs = {f"{stage}_loss": loss}
        self.log(f"{stage}_loss",loss, on_epoch=True,on_step=False, prog_bar=True, logger=True)
        for name, metric in self.metrics.items():
            metric_val = metric(y_hat, y)
            logs[f"{stage}_{name}"] = metric_val
            self.log(f"{stage}_{name}", metric_val, on_epoch=True,on_step=False, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer =  optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class EEG_Emotion(LightningModule):
    """
    This class defines the model used for the emotion detection task.
    It uses the encoder from the pre-trained model.
    """
    def __init__(self,  
                learning_rate: float,
                convnet: nn.Module,
                encoder: nn.Module,
                decoder: nn.Module
                 ):
        super().__init__()
        self.metrics = {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes = 5).to("cuda"),
                        "f1": torchmetrics.F1Score(task="multiclass", num_classes = 5, average = 'macro').to("cuda"),
                        "AUC": torchmetrics.AUROC(task="multiclass", num_classes = 5).to("cuda")}
       
        self.learning_rates = learning_rate
        self.covnet = convnet
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # x = x.reshape(x.shape[0],1, x.shape[1],
        #    x.shape[2])
        # x = self.covnet(x)
        x, attn = self.encoder(x)
        x =  self.decoder(x)
        return x

    def shared_step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        logs = {f"{stage}_loss": loss}
        self.log(f"{stage}_loss",loss, on_epoch=True,on_step=False, prog_bar=True, logger=True)
        for name, metric in self.metrics.items():
            metric_val = metric(y_hat, y)
            logs[f"{stage}_{name}"] = metric_val
            self.log(f"{stage}_{name}", metric_val, on_epoch=True,on_step=False, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    # def test_step(self, batch, batch_idx):
    #     return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer =  optim.Adam(self.parameters(), lr=self.learning_rates)
        return optimizer    


class EEGInceptionPL(LightningModule):
    """
    This class defines the model with the EEG-Inception architecture.
    """
    def __init__(self, input_time=1000, fs=128, ncha=8, filters_per_branch=8, scales_time=(500, 250, 125), dropout_rate=0.25, activation='ELU', n_classes=1, learning_rate=0.001):
        super(EEGInceptionPL, self).__init__()

        self.model = EEGInception(input_time, fs, ncha, filters_per_branch, scales_time, dropout_rate, activation, n_classes, learning_rate)
        self.learning_rate = learning_rate
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes = n_classes).to("cuda"),
                        "f1": torchmetrics.F1Score(task="multiclass",num_classes = n_classes).to("cuda"),
                        "AUC": torchmetrics.AUROC(task="multiclass", num_classes = n_classes).to("cuda")}

    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        logs = {f"{stage}_loss": loss}
        self.log(f"{stage}_loss",loss, on_epoch=True,on_step=False, prog_bar=True, logger=True)
        for name, metric in self.metrics.items():
            metric_val = metric(y_hat, y)
            logs[f"{stage}_{name}"] = metric_val
            self.log(f"{stage}_{name}", metric_val, on_epoch=True,on_step=False, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val") 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1024, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1024)
    


