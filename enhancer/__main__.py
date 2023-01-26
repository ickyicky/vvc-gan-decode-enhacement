from .models.discriminator import Discriminator
from .models.enhancer import Enhancer
from .datamodule import VVCDataModule
from .gan_module import GANModule
from .utils import weights_init
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
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
    parser.add_argument(
        "-c",
        "--checkpoint",
        metavar="FILE",
        required=False,
        action="store",
        default=None,
        help="checkpoint to load",
    )

    args = parser.parse_args()

    data_module = VVCDataModule(
        args.chunks_dir,
        args.orig_chunks_dir,
    )

    if args.checkpoint:
        module = GANModule.load_from_checkpoint(args.checkpoint)
    else:
        enhancer = Enhancer()
        discriminator = Discriminator()
        enhancer.apply(weights_init)
        discriminator.apply(weights_init)
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
            ModelCheckpoint(dirpath="checkpoints", filename="{epoch}"),
        ],
        logger=wandb_logger,
    )
    trainer.fit(module, data_module)
