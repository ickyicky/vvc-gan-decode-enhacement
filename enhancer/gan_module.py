import torch
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.transforms.functional import crop
from ignite.mtetrics import PSNR
from typing import Tuple
from .crosslid import compute_crosslid


class GANModule(pl.LightningModule):
    def __init__(
        self,
        enhancer,
        discriminator,
        enhancer_lr: float = 1e-4,
        discriminator_lr: float = 1e-5,
        betas: Tuple[float, float] = (0.5, 0.999),
        num_samples: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="psnr")

        self.enhancer = enhancer
        self.discriminator = discriminator

        self.enhancer_lr = enhancer_lr
        self.discriminator_lr = discriminator_lr

        self.betas = betas

        self.num_samples = num_samples

        self.psnr = PSNR(output_transform=self.psnr_transform)

    def psnr_transform(self, output):
        y_pred, y = output
        return (
            crop(
                y_pred,
                2,
                2,
                128,
                128,
            ),
            crop(
                y,
                2,
                2,
                128,
                128,
            ),
        )

    def forward(self, chunks, metadata):
        return self.enhancer(chunks, metadata)

    def adversarial_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def crosslid(self, y_hat_features, y_features):
        b_size = y_features.shape[0]
        return compute_crosslid(y_hat_features.cpu(), y_features.cpu(), b_size, b_size)

    def training_step(self, batch, batch_idx, optimizer_idx):
        chunks, orig_chunks, metadata = batch

        # train ENHANCE!
        if optimizer_idx == 0:

            # ENHANCE!
            enhanced = self(chunks, metadata)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(chunks.size(0), 1)
            valid = valid.type_as(chunks)

            # adversarial loss is binary cross-entropy
            preds = self.discriminator(enhanced)
            g_loss = self.adversarial_loss(preds, valid)

            self.log("g_loss", g_loss, prog_bar=True)

            if batch_idx % 20 == 0:
                self.logger.experiment.log(
                    {
                        "enhanced": [
                            wandb.Image(
                                x,
                                caption=f"Pred: {pred.item()}",
                            )
                            for x, pred in zip(
                                enhanced[: self.num_samples],
                                preds.cpu()[: self.num_samples],
                            )
                        ],
                        "uncompressed": [
                            wandb.Image(
                                x,
                                caption=f"uncompressed image {i}",
                            )
                            for i, x in enumerate(orig_chunks[: self.num_samples])
                        ],
                        "decompressed": [
                            wandb.Image(
                                x,
                                caption=f"decompressed image {i}",
                            )
                            for i, x in enumerate(chunks[: self.num_samples])
                        ],
                    }
                )
            return g_loss

        # train discriminator
        if optimizer_idx == 1:

            # Measure discriminator's ability to classify real from generated samples
            # how well can it label as real?
            valid = torch.ones(orig_chunks.size(0), 1)
            valid = valid.type_as(orig_chunks)

            real_loss = self.adversarial_loss(self.discriminator(orig_chunks), valid)

            # how well can it label as fake?
            fake = torch.zeros(orig_chunks.size(0), 1)
            fake = fake.type_as(orig_chunks)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(chunks, metadata).detach()), fake
            )

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def validation_step(self, batch, batch_idx):
        chunks, orig_chunks, metadata = batch

        # create holder for features
        target = {}
        hook = self.discriminator.register_hook(target)

        # ENHANCE!
        enhanced = self(chunks, metadata)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(chunks.size(0), 1)
        valid = valid.type_as(chunks)

        # adversarial loss is binary cross-entropy
        preds = self.discriminator(enhanced)
        y_hat_features = target["features"]

        g_loss = self.adversarial_loss(preds, valid)

        self.log("val_g_loss", g_loss, prog_bar=True)
        if batch_idx % 20 == 0:
            self.logger.experiment.log(
                {
                    "validation_enhanced": [
                        wandb.Image(
                            x,
                            caption=f"Pred: {pred.item()}",
                        )
                        for x, pred in zip(
                            enhanced[: self.num_samples],
                            preds.cpu()[: self.num_samples],
                        )
                    ],
                    "validation_uncompressed": [
                        wandb.Image(
                            x,
                            caption=f"uncompressed image {i}",
                        )
                        for i, x in enumerate(orig_chunks[: self.num_samples])
                    ],
                    "validation_decompressed": [
                        wandb.Image(
                            x,
                            caption=f"decompressed image {i}",
                        )
                        for i, x in enumerate(chunks[: self.num_samples])
                    ],
                }
            )

        real_loss = self.adversarial_loss(self.discriminator(orig_chunks), valid)
        y_features = target["features"]

        # how well can it label as fake?
        fake = torch.zeros(orig_chunks.size(0), 1)
        fake = fake.type_as(orig_chunks)

        fake_loss = self.adversarial_loss(preds, fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("val_d_loss", d_loss, prog_bar=True)

        # calculate crosslid
        crosslid = self.crosslid(y_hat_features, y_features)
        self.log("val_crosslid", crosslid, prog_bar=True)
        hook.remove()

    def test_step(self, batch, batch_idx):
        # TODO log everything
        chunks, orig_chunks, metadata = batch

        # create holder for features
        target = {}
        hook = self.discriminator.register_hook(target)

        # ENHANCE!
        enhanced = self(chunks, metadata)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(chunks.size(0), 1)
        valid = valid.type_as(chunks)

        # adversarial loss is binary cross-entropy
        preds = self.discriminator(enhanced)
        y_hat_features = target["features"]

        g_loss = self.adversarial_loss(preds, valid)

        self.log("test_g_loss", g_loss, prog_bar=True)
        if batch_idx % 20 == 0:
            self.logger.experiment.log(
                {
                    "test_enhanced": [
                        wandb.Image(
                            x,
                            caption=f"Pred: {pred.item()}",
                        )
                        for x, pred in zip(
                            enhanced[: self.num_samples],
                            preds.cpu()[: self.num_samples],
                        )
                    ],
                    "test_uncompressed": [
                        wandb.Image(
                            x,
                            caption=f"uncompressed image {i}",
                        )
                        for i, x in enumerate(orig_chunks[: self.num_samples])
                    ],
                    "test_decompressed": [
                        wandb.Image(
                            x,
                            caption=f"decompressed image {i}",
                        )
                        for i, x in enumerate(chunks[: self.num_samples])
                    ],
                }
            )

        real_loss = self.adversarial_loss(self.discriminator(orig_chunks), valid)
        y_features = target["features"]

        # how well can it label as fake?
        fake = torch.zeros(orig_chunks.size(0), 1)
        fake = fake.type_as(orig_chunks)

        fake_loss = self.adversarial_loss(preds, fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("test_d_loss", d_loss, prog_bar=True)

        # calculate crosslid, this time as well for decompressed images
        crosslid = self.crosslid(y_hat_features, y_features)
        self.log("test_enhanced_crosslid", crosslid, prog_bar=True)

        self.adversarial_loss(self.discriminator(chunks), fake)
        orig_features = target["features"]
        orig_crosslid = self.crosslid(orig_features, y_features)
        self.log("test_orig_crosslid", orig_crosslid, prog_bar=True)

        hook.remove()

        # calculate psnr
        enhanced_psnr = self.psnr(enhanced, orig_chunks)
        self.log("test_enhancer_psnr", enhanced_psnr, prog_bar=True)
        orig_psnr = self.psnr(chunks, orig_chunks)
        self.log("test_orig_psnr", orig_psnr, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.enhancer.parameters(), lr=self.enhancer_lr, betas=self.betas
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.discriminator_lr, betas=self.betas
        )

        # example scheduler to use, didn't use it finally
        lr_schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    opt_d,
                    milestones=[10],
                    gamma=0.1,
                ),
                "interval": "step",
                "frequency": 1,
            },
        ]

        return [opt_g, opt_d], []
