import math
import os
import torch 
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from .Dataset import MaskedDataset, StressDataset

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
        train_split = math.ceil(length * 0.7)  # rounds up
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

class ERPDataModule(L.LightningDataModule):
    def __init__(self, dataset: Dataset, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_workers = multiprocessing.cpu_count()  # maximum number of workers

    def setup(self, stage=None, seed: int = 42):
        # np.random.seed(seed)
        # unique_subjects = np.unique(self.dataset.subjects)
        # np.random.shuffle(unique_subjects)

        # num_subjects = len(unique_subjects)
        # train_size = math.ceil(num_subjects * 0.7)  # rounds up
        # val_size = int(num_subjects * 0.2)  # rounds down
        # test_size = num_subjects - train_size - val_size

        # train_subjects = unique_subjects[:train_size]
        # val_subjects = unique_subjects[train_size:train_size + val_size]
        # test_subjects = unique_subjects[train_size + val_size:]

        # train_indices = np.where(np.isin(self.dataset.subjects, train_subjects))[0]
        # val_indices = np.where(np.isin(self.dataset.subjects, val_subjects))[0]
        # test_indices = np.where(np.isin(self.dataset.subjects, test_subjects))[0]

        # self.train_dataset = Subset(self.dataset, train_indices)
        # self.val_dataset = Subset(self.dataset, val_indices)
        # self.test_dataset = Subset(self.dataset, test_indices)
        length = len(self.dataset)
        train_split = math.ceil(length * 0.7)  # rounds up
        test_split = int(length * 0.2)  # rounds down
        val_split = length - train_split - test_split
        self.train_dataset, self. val_dataset, self.test_dataset = random_split(
            self.dataset, [train_split, val_split, test_split], generator=torch.Generator().manual_seed(seed))

        print(f"Train: {len(self.train_dataset)} Val: {len(self.val_dataset)} Test: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
if __name__ == "__main__":
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        stress_data = os.path.join(os.getcwd(), 'deap_stress_4s_subject.h5')
        dataset = StressDataset(stress_data)
        dm = ERPDataModule(dataset)
        dm.setup(seed=42)
        