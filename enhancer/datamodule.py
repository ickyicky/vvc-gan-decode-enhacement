import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
import pytorch_lightning as pl
from typing import Optional, Tuple

from .dataset import VVCDataset


class LoaderWrapper:
    """LoaderWrapper."""

    def __init__(self, dataloader: DataLoader, n_step: int):
        """__init__.

        :param dataloader:
        :type dataloader: DataLoader
        :param n_step:
        :type n_step: int
        """
        self.n_step = n_step
        self.idx = 0
        self.dataloader = dataloader
        self.iter_loader = iter(dataloader)

    def __iter__(self) -> "LoaderWrapper":
        """__iter__.

        :rtype: "LoaderWrapper"
        """
        return self

    def __len__(self) -> int:
        """__len__.

        :rtype: int
        """
        return self.n_step

    def __next__(self):
        """__next__."""
        if self.idx == self.n_step:
            self.idx = 0
            raise StopIteration
        else:
            self.idx += 1

        try:
            return next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.dataloader)
            return next(self.iter_loader)


class VVCDataModule(pl.LightningDataModule):
    """VVCDataModule."""

    def __init__(
        self,
        chunk_folder: str,
        orig_chunk_folder: str,
        test_chunk_folder: Optional[str] = None,
        test_orig_chunk_folder: Optional[str] = None,
        chunk_height: int = 132,
        chunk_width: int = 132,
        batch_size: int = 8,
        val_batch_size: int = 96,
        test_batch_size: int = 96,
        n_step: int = 1000,
        n_step_valid: int = 5,
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
        """
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.chunk_folder = chunk_folder
        self.orig_chunk_folder = orig_chunk_folder

        self.test_chunk_folder = test_chunk_folder
        self.test_orig_chunk_folder = test_orig_chunk_folder

        self.chunk_height = chunk_height
        self.chunk_width = chunk_width

        self.n_step = n_step
        self.n_step_valid = n_step_valid

        self.dataset_val = None
        self.dataset_test = None
        self.dataset_train = None

    def setup(self, stage=None):
        """setup.

        :param stage:
        """
        if stage == "fit":
            self.dataset_train = VVCDataset(
                chunk_folder=self.chunk_folder,
                orig_chunk_folder=self.orig_chunk_folder,
                chunk_transform=self.chunk_transform(),
                metadata_transform=self.metadata_transform(),
                chunk_height=self.chunk_height,
                chunk_width=self.chunk_width,
            )

            self.dataset_val = VVCDataset(
                chunk_folder=self.test_chunk_folder,
                orig_chunk_folder=self.test_orig_chunk_folder,
                chunk_transform=self.chunk_transform(),
                metadata_transform=self.metadata_transform(),
                chunk_height=self.chunk_height,
                chunk_width=self.chunk_width,
            )

        if stage == "test":
            self.dataset_test = VVCDataset(
                chunk_folder=self.test_chunk_folder,
                orig_chunk_folder=self.test_orig_chunk_folder,
                chunk_transform=self.chunk_transform(),
                metadata_transform=self.metadata_transform(),
                chunk_height=self.chunk_height,
                chunk_width=self.chunk_width,
            )

    def train_dataloader(self):
        """train_dataloader."""
        data_loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )
        return LoaderWrapper(
            data_loader,
            self.n_step,
        )

    def test_dataloader(self):
        """test_dataloader."""
        data_loader = DataLoader(
            self.dataset_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )
        return data_loader

    def val_dataloader(self):
        """val_dataloader."""
        data_loader = DataLoader(
            self.dataset_val,
            batch_size=self.val_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )
        return LoaderWrapper(
            data_loader,
            self.n_step_valid,
        )

    def chunk_transform(self):
        """chunk_transform."""
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        return transform

    def metadata_transform(self):
        """metadata_transform."""

        def transform(metadata):
            return torch.as_tensor(metadata).float().view(len(metadata), 1, 1)

        return transform


if __name__ == "__main__":
    import sys

    d = VVCDataModule(*sys.argv[1:])
    d.setup()
    train = d.train_dataloader()
    for x in train:
        print(x[0].shape)
        print(x[2].shape)
        break
