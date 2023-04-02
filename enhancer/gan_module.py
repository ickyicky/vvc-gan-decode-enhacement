import torch
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import crop
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from typing import Tuple
from .crosslid import compute_crosslid
from .csv_logger import log


class GANModule(pl.LightningModule):
    def __init__(
        self,
        enhancer,
        discriminator,
        enhancer_lr: float = 2e-4,
        discriminator_lr: float = 1e-5,
        betas: Tuple[float, float] = (0.5, 0.999),
        num_samples: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.enhancer = enhancer
        self.discriminator = discriminator

        self.enhancer_lr = enhancer_lr
        self.discriminator_lr = discriminator_lr

        self.betas = betas

        self.num_samples = num_samples

    def psnr_transform(self, output):
        # crop removes area that is gradiented
        return crop(
            output,
            4,
            4,
            124,
            124,
        )

    def forward(self, chunks, metadata):
        return self.enhancer(chunks, metadata)

    def adversarial_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def crosslid(self, y_hat_features, y_features, without_mean=False):
        b_size = y_features.shape[0]
        return compute_crosslid(
            y_hat_features.cpu(),
            y_features.cpu(),
            b_size,
            b_size,
            without_mean=without_mean,
        )

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

            if batch_idx % 100 == 0:
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

        # calculate crosslid
        crosslid = self.crosslid(y_hat_features, y_features)

        self.adversarial_loss(self.discriminator(chunks), fake)
        orig_features = target["features"]
        orig_crosslid = self.crosslid(orig_features, y_features)
        hook.remove()

        # calculate psnr and ssim
        enhanced_psnr = psnr(
            self.psnr_transform(enhanced), self.psnr_transform(orig_chunks)
        )
        orig_psnr = psnr(self.psnr_transform(chunks), self.psnr_transform(orig_chunks))
        enhanced_ssim = ssim(
            self.psnr_transform(enhanced), self.psnr_transform(orig_chunks)
        )
        orig_ssim = ssim(self.psnr_transform(chunks), self.psnr_transform(orig_chunks))

        # log everything
        self.log_dict(
            {
                "val_g_loss": g_loss,
                "val_d_loss": d_loss,
                "val_crosslid": crosslid,
                "val_ref_crosslid": orig_crosslid,
                "val_psnr": enhanced_psnr,
                "val_ref_psnr": orig_psnr,
                "val_ssim": enhanced_ssim,
                "val_ref_ssim": orig_ssim,
            },
        )

    def test_step(self, batch, batch_idx):
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

        # calculate crosslid, this time as well for decompressed images
        crosslid = self.crosslid(y_hat_features, y_features, True)

        self.adversarial_loss(self.discriminator(chunks), fake)
        orig_features = target["features"]
        orig_crosslid = self.crosslid(orig_features, y_features, True)
        hook.remove()

        # calculate psnr and ssim
        enhancer_psnrs = [
            psnr(self.psnr_transform(enh), self.psnr_transform(orig_chunk))
            for enh, orig_chunk in zip(enhanced.split(1), orig_chunks.split(1))
        ]
        orig_psnrs = [
            psnr(self.psnr_transform(chunk), self.psnr_transform(orig_chunk))
            for chunk, orig_chunk in zip(chunks.split(1), orig_chunks.split(1))
        ]
        enhancer_ssims = [
            ssim(self.psnr_transform(enh), self.psnr_transform(orig_chunk))
            for enh, orig_chunk in zip(enhanced.split(1), orig_chunks.split(1))
        ]
        orig_ssims = [
            ssim(self.psnr_transform(chunk), self.psnr_transform(orig_chunk))
            for chunk, orig_chunk in zip(chunks.split(1), orig_chunks.split(1))
        ]

        # log everything
        self.log_dict(
            {
                "test_g_loss": g_loss,
                "test_d_loss": d_loss,
                "test_crosslid": np.mean(crosslid),
                "test_ref_crosslid": np.mean(orig_crosslid),
                "test_psnr": torch.mean(torch.tensor(enhancer_psnrs)),
                "test_ref_psnr": torch.mean(torch.tensor(orig_psnrs)),
                "test_ssim": torch.mean(torch.tensor(enhancer_ssims)),
                "test_ref_ssim": torch.mean(torch.tensor(orig_ssims)),
            },
        )
        self.log_test(
            {
                "test_crosslid": crosslid,
                "test_ref_crosslid": orig_crosslid,
                "test_psnr": enhancer_psnrs,
                "test_ref_psnr": orig_psnrs,
                "test_ssim": enhancer_ssims,
                "test_ref_ssim": orig_ssims,
                "test_metadata": metadata,
            },
        )

    def log_test(self, data):
        log(data)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.enhancer.parameters(),
            lr=self.enhancer_lr,
            betas=self.betas,
            weight_decay=0.01,
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
            betas=self.betas,
            weight_decay=0.01,
        )

        lr_schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    opt_d,
                    milestones=[30, 60, 90],
                    gamma=0.1,
                ),
                "interval": "epoch",
                "frequency": 1,
            },
            {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    opt_g,
                    milestones=[30, 60, 90],
                    gamma=0.1,
                ),
                "interval": "epoch",
                "frequency": 1,
            },
        ]

        return [opt_g, opt_d], lr_schedulers
