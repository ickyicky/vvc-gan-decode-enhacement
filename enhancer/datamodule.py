import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
import pytorch_lightning as pl

from .dataset import VVCDataset


class VVCDataModule(pl.LightningDataModule):
    """VVCDataModule."""

    def __init__(
        self,
        chunk_folder: str,
        orig_chunk_folder: str,
        chunk_height: int = 128,
        chunk_width: int = 128,
        batch_size: int = 256,
        val_percentage: int = 20,
        test_percentage: int = 20,
    ):
        """__init__.

        :param chunk_folder:
        :type chunk_folder: str
        :param orig_chunk_folder:
        :type orig_chunk_folder: str
        :param chunk_height:
        :type chunk_height: int
        :param chunk_width:
        :type chunk_width: int
        :param batch_size:
        :type batch_size: int
        :param val_percentage:
        :type val_percentage: int
        :param test_percentage:
        :type test_percentage: int
        """
        super().__init__()
        self.batch_size = batch_size

        self.chunk_folder = chunk_folder
        self.orig_chunk_folder = orig_chunk_folder

        self.chunk_height = chunk_height
        self.chunk_width = chunk_width

        self.val_percentage = val_percentage
        self.test_percentage = test_percentage

    def setup(self, stage=None):
        """setup.

        :param stage:
        """
        dataset = VVCDataset(
            chunk_folder=self.chunk_folder,
            orig_chunk_folder=self.orig_chunk_folder,
            chunk_transform=self.chunk_transform(),
            metadata_transform=self.metadata_transform(),
            chunk_height=self.chunk_height,
            chunk_width=self.chunk_width,
        )

        indices = torch.randperm(len(dataset)).tolist()
        val_items = int(self.val_percentage / 100 * len(dataset))
        test_items = int(self.test_percentage / 100 * len(dataset))

        self.dataset_val = Subset(dataset, indices[:val_items])
        self.dataset_test = Subset(dataset, indices[val_items:test_items])
        self.dataset_train = Subset(dataset, indices[val_items + test_items :])

    def train_dataloader(self):
        """train_dataloader."""
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self):
        """test_dataloader."""
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        """val_dataloader."""
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )

    def chunk_transform(self):
        """chunk_transform."""
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def metadata_transform(self):
        """metadata_transform."""
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


if __name__ == "__main__":
    import sys

    d = VVCDataModule(*sys.argv[1:])
    import ipdb

    ipdb.set_trace()
