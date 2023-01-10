from .models.discriminator import Discriminator
from .models.enhancer import Enhancer
from .datamodule import VVCDataModule
from .gan_module import GANModule
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import torch


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--chunks-dir",
        "-d",
        metavar="FILE",
        default="chunks",
        help="directory with chunks",
    )
    parser.add_argument(
        "--orig-chunks-dir",
        "-o",
        metavar="FILE",
        default="orig_chunks",
        help="directory with original chunks",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=5,
        help="number of epochs",
    )

    args = parser.parse_args()

    enhancer = Enhancer()
    discriminator = Discriminator()

    data_module = VVCDataModule(
        args.chunks_dir,
        args.orig_chunks_dir,
    )

    module = GANModule(
        enhancer,
        discriminator,
    )

    wandb_logger = WandbLogger(
        project="vvc-enhancer",
    )

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=args.epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=wandb_logger,
    )
    trainer.fit(module, data_module)
