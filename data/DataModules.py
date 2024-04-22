import math
import os
import torch 
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split, Dataset
from .Dataset import MaskedDataset

import pytorch_lightning as L
from typing import List

import multiprocessing

""" 
DataModules are a way of decoupling data-related hooks from the LightningModule so you can develop dataset agnostic models.

"""
BATCH_SIZE = 16

class SSLDataModule(L.LightningDataModule):
    def __init__(self, dataset: Dataset, batch_size: int = BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size

        self.current_task = 0
        self.dataset = dataset

        self.num_workers = multiprocessing.cpu_count()  # maximum number of workers

    def prepare_data(self, seed : int = 42):
        length = len(self.dataset)
        print(f"The length of the dataset is: {length}")
        train_split = math.ceil(length * 0.6)  # rounds up
        test_split = int(length * 0.2)  # rounds down
        val_split = length - train_split - test_split
        self.train_dataset, self. val_dataset, self.test_dataset = random_split(
            self.dataset, [train_split, val_split, test_split], generator=torch.Generator().manual_seed(seed))
                    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    raw_csv = os.path.join(parent_dir, 'reshaped_signals.csv')
    masked_csv = os.path.join(parent_dir, 'masked_signals.csv')
    mask_csv = os.path.join(parent_dir, 'mask.csv')

    dataset = MaskedDataset(raw_csv, masked_csv, mask_csv)
    dm = SSLDataModule(dataset)
    dm.prepare_data()
    print(dm.train_dataloader())
    print(dm.val_dataloader())
    print(len(dm.train_dataloader()))

class ERPDataModule(L.LightningDataModule):
    def __init__(self, dataset: Dataset, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset

        self.num_workers = multiprocessing.cpu_count()  # maximum number of workers

    def prepare_data(self, seed : int = 42):
        length = len(self.dataset)
        print(f"The length of the dataset is: {length}")
        train_split = math.ceil(length * 0.6)  # rounds up
        val_split = int(length * 0.2)  # rounds down
        test_split = length - train_split- val_split
        # self.train_dataset = Subset(self.dataset, range(train_split))
        # self.val_dataset = Subset(self.dataset, range(train_split, length))
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_split, val_split, test_split], generator=torch.Generator().manual_seed(seed))
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)