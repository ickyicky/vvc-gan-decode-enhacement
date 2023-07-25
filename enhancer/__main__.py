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
        "-i",
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
        "--test-chunks-dir",
        "-x",
        metavar="FILE",
        default="test_chunks",
        help="directory with chunks",
    )
    parser.add_argument(
        "--test-orig-chunks-dir",
        "-y",
        metavar="FILE",
        default="test_orig_chunks",
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
        help="checkpoint to load for enhancer",
    )
    parser.add_argument(
        "-d",
        "--discriminator",
        metavar="FILE",
        required=False,
        action="store",
        default=None,
        help="checkpoint to load for discriminator",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="number of epochs",
    )
    parser.add_argument(
        "--run",
        "-r",
        action="store_true",
        help="number of epochs",
    )
    parser.add_argument(
        "--mode",
        "-m",
        action="store",
        default="gan",
        type=str,
        help="mode of operation",
        choices=["gan", "enhancer", "discriminator"],
    )

    args = parser.parse_args()

    data_module = VVCDataModule(
        args.chunks_dir,
        args.orig_chunks_dir,
        args.test_chunks_dir,
        args.test_orig_chunks_dir,
    )

    if args.checkpoint:
        enhancer = GANModule.load_from_checkpoint(args.checkpoint).enhancer
    else:
        enhancer = Enhancer()
        enhancer.apply(weights_init)

    if args.discriminator:
        discriminator = GANModule.load_from_checkpoint(args.discriminator).discriminator
    else:
        discriminator = Discriminator()
        discriminator.apply(weights_init)

    module = GANModule(
        enhancer,
        discriminator,
        mode=args.mode,
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
            ModelCheckpoint(dirpath="checkpoints", filename="{}"),
        ],
        logger=wandb_logger,
    )
    if args.test:
        trainer.test(module, data_module)
    elif args.run:
        trainer.predict(module, data_module)
    else:
        trainer.fit(module, data_module)
