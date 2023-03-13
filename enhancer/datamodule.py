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
        n_step: int = 1000,
        val_percentage: int = 5,
        test_percentage: int = 5,
        mean: Tuple[float, float, float] = (0.4037, 0.3721, 0.3697),
        std: Tuple[float, float, float] = (0.2841, 0.2851, 0.2928),
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

        self.test_chunk_folder = test_chunk_folder
        self.test_orig_chunk_folder = test_orig_chunk_folder

        self.chunk_height = chunk_height
        self.chunk_width = chunk_width

        self.val_percentage = val_percentage
        self.test_percentage = test_percentage

        self.n_step = n_step
        self.n_step_valid = int(n_step / 100 * self.val_percentage)
        self.n_step_test = int(n_step / 100 * self.test_percentage)

        self.dataset_val = None
        self.dataset_test = None
        self.dataset_train = None

        self.mean = mean
        self.std = std

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

        if (
            self.test_chunk_folder is not None
            and self.test_orig_chunk_folder is not None
        ):
            self.dataset_test = VVCDataset(
                chunk_folder=self.test_chunk_folder,
                orig_chunk_folder=self.test_orig_chunk_folder,
                chunk_transform=self.chunk_transform(),
                metadata_transform=self.metadata_transform(),
                chunk_height=self.chunk_height,
                chunk_width=self.chunk_width,
            )
            self.dataset_train = Subset(dataset, indices[val_items:])
        else:
            self.dataset_test = Subset(dataset, indices[val_items:test_items])
            self.dataset_train = Subset(dataset, indices[val_items + test_items :])

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
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )
        return LoaderWrapper(
            data_loader,
            self.n_step_test,
        )

    def val_dataloader(self):
        """val_dataloader."""
        data_loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
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
                # transforms.Normalize(self.mean, self.std),
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
