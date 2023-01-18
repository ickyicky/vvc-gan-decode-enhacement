import torch
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Tuple


class GANModule(pl.LightningModule):
    def __init__(
        self,
        enhancer,
        discriminator,
        enhancer_lr: float = 0.0001,
        discriminator_lr: float = 0.00009,
        betas: Tuple[float, float] = (0.5, 0.999),
        num_samples: int = 6,
    ):
        super().__init__()

        self.enhancer = enhancer
        self.discriminator = discriminator

        self.enhancer_lr = enhancer_lr
        self.discriminator_lr = discriminator_lr

        self.betas = betas

        self.num_samples = num_samples

    def forward(self, chunks, metadata):
        return self.enhancer(chunks, metadata)

    def adversarial_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

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

        # ENHANCE!
        enhanced = self(chunks, metadata)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(chunks.size(0), 1)
        valid = valid.type_as(chunks)

        # adversarial loss is binary cross-entropy
        preds = self.discriminator(enhanced)
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

        # how well can it label as fake?
        fake = torch.zeros(orig_chunks.size(0), 1)
        fake = fake.type_as(orig_chunks)

        fake_loss = self.adversarial_loss(preds, fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("val_d_loss", d_loss, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.enhancer.parameters(), lr=self.enhancer_lr, betas=self.betas
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.discriminator_lr, betas=self.betas
        )

        lr_schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    opt_d,
                    milestones=[1000],
                    gamma=0.5,
                ),
                "interval": "step",
                "frequency": 1,
            },
            {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    opt_g,
                    milestones=[1000],
                    gamma=0.5,
                ),
                "interval": "step",
                "frequency": 1,
            },
        ]

        return [opt_g, opt_d], []
