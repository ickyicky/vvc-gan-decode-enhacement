import torch
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from typing import Tuple
from .dataset import VVCDataset
from pytorch_msssim import SSIM, MS_SSIM


class GANModule(pl.LightningModule):
    def __init__(
        self,
        enhancer,
        discriminator,
        enhancer_lr: float = 1e-4,
        discriminator_lr: float = 2e-6,
        betas: Tuple[float, float] = (0.5, 0.999),
        num_samples: int = 6,
        enhancer_min_loss: float = 0.25,
        discriminator_min_loss: float = 0.1,
        enhancer_max_loss: float = 0.6,
        discriminator_max_loss: float = 0.25,
        probe: int = 10,
        mode: str = "gan",
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.enhancer = enhancer
        self.discriminator = discriminator

        self.enhancer_lr = enhancer_lr
        self.discriminator_lr = discriminator_lr

        self.betas = betas
        self.discriminator_min_loss = discriminator_min_loss
        self.enhancer_min_loss = enhancer_min_loss
        self.discriminator_max_loss = discriminator_max_loss
        self.enhancer_max_loss = enhancer_max_loss

        self.enhancer_losses = [1.0]
        self.discriminator_losses = [1.0]
        self.probe = probe

        self.num_samples = num_samples
        self.ssim = SSIM(data_range=1.0, win_size=9)
        self.msssim = MS_SSIM(data_range=1.0, win_size=9)

        self.mode = mode

    def psnr_transform(self, output):
        # crop removes area that is gradiented
        return output

    def forward(self, chunks, metadata):
        return self.enhancer(chunks, metadata)

    def adversarial_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        chunks, orig_chunks, metadata, _ = batch

        if self.mode == "enhancer":
            e_train = True
            d_train = False
        elif self.mode == "discriminator":
            e_train = False
            d_train = True
        else:
            e_loss = np.mean(self.enhancer_losses)
            d_loss = np.mean(self.discriminator_losses)

            e_train = (
                e_loss >= self.enhancer_min_loss and d_loss <= self.discriminator_max_loss
            )
            d_train = (
                d_loss >= self.discriminator_min_loss and e_loss <= self.enhancer_max_loss
            )

            if not e_train and not d_train:
                e_train = d_train = True

        # train ENHANCE!
        # ENHANCE!
        if self.mode in ("enhancer", "gan"):
            g_opt.zero_grad()
            enhanced = self(chunks, metadata)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(chunks.size(0), 1)
            valid = valid.type_as(chunks)

            # adversarial loss is binary cross-entropy
            ssim_loss = 1 - self.ssim(orig_chunks, enhanced)
            msssim_loss = 1 - self.msssim(orig_chunks, enhanced)
            mse_loss = F.mse_loss(enhanced, orig_chunks)
            self.log("g_ssim_loss", ssim_loss, prog_bar=False)
            self.log("g_msssim_loss", msssim_loss, prog_bar=False)
            self.log("g_mse_loss", mse_loss, prog_bar=False)

            if self.mode == "gan":
                preds = self.discriminator(enhanced)
                gd_loss = self.adversarial_loss(preds, valid)
                g_loss = (
                    0.4 * gd_loss + 0.2 * msssim_loss + 0.2 * ssim_loss + 0.2 * mse_loss
                )
                self.log("g_d_loss", gd_loss, prog_bar=True)
            else:
                g_loss = 0.3 * msssim_loss + 0.3 * ssim_loss + 0.4 * mse_loss

            self.enhancer_losses.append(gd_loss.item())
            self.enhancer_losses = self.enhancer_losses[: self.probe]

            self.log("g_loss", g_loss, prog_bar=True)

            if e_train:
                if batch_idx % 100 == 0:
                    log = {"enhanced": [], "uncompressed": [], "decompressed": []}
                    for i in range(self.num_samples):
                        enh = enhanced[i].cpu()
                        orig = orig_chunks[i].cpu()
                        dec = chunks[i].cpu()

                        log["enhanced"].append(
                            wandb.Image(dec + enh, caption=f"Pred: {preds[i].item()}")
                        )
                        log["uncompressed"].append(
                            wandb.Image(orig, caption=f"uncompressed image {i}")
                        )
                        log["decompressed"].append(
                            wandb.Image(dec, caption=f"decompressed image {i}")
                        )
                    self.logger.experiment.log(log)

                self.manual_backward(g_loss)
                g_opt.step()

        # train discriminator
        if d_train:
            d_opt.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            # how well can it label as real?
            valid = torch.ones(orig_chunks.size(0), 1)
            valid = valid.type_as(orig_chunks)

            real_loss = self.adversarial_loss(self.discriminator(orig_chunks), valid)

            # how well can it label as fake?
            fake = torch.zeros(orig_chunks.size(0), 1)
            fake = fake.type_as(orig_chunks)

            if self.mode == "gan":
                fake_loss = self.adversarial_loss(
                    self.discriminator(self(chunks, metadata).detach()), fake
                )
            else:
                fake_loss = self.adversarial_loss(
                    self.discriminator(chunks), fake
                )

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2

            self.discriminator_losses.append(d_loss.item())
            self.discriminator_losses = self.discriminator_losses[: self.probe]

            self.log("d_loss", d_loss, prog_bar=True)
            self.log("d_real_loss", real_loss, prog_bar=False)
            self.log("d_fake_loss", fake_loss, prog_bar=False)

            self.manual_backward(d_loss)
            d_opt.step()

    def on_train_epoch_end(self):
        schs = self.lr_schedulers()

        if schs is None:
            return

        if not isinstance(schs, (tuple, list)):
            schs = [schs]

        for sch in schs:
            sch.step()

    def validation_step(self, batch, batch_idx):
        chunks, orig_chunks, metadata, _ = batch

        # ENHANCE!
        enhanced = self(chunks, metadata)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(chunks.size(0), 1)
        valid = valid.type_as(chunks)

        # adversarial loss is binary cross-entropy
        preds = self.discriminator(enhanced)

        g_loss = self.adversarial_loss(preds, valid)

        if batch_idx % 20 == 0:
            log = {"enhanced": [], "uncompressed": [], "decompressed": []}
            for i in range(self.num_samples):
                enh = enhanced[i].cpu()
                orig = orig_chunks[i].cpu()
                dec = chunks[i].cpu()

                log["enhanced"].append(
                    wandb.Image(enh, caption=f"Pred: {preds[i].item()}")
                )
                log["uncompressed"].append(
                    wandb.Image(orig, caption=f"uncompressed image {i}")
                )
                log["decompressed"].append(
                    wandb.Image(dec, caption=f"decompressed image {i}")
                )
            self.logger.experiment.log({f"validation_{k}": v for k, v in log.items()})

        real_loss = self.adversarial_loss(self.discriminator(orig_chunks), valid)

        # how well can it label as fake?
        fake = torch.zeros(orig_chunks.size(0), 1)
        fake = fake.type_as(orig_chunks)

        fake_loss = self.adversarial_loss(preds, fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2

        # calculate psnr, ssim,
        transformed_enhacned = self.psnr_transform(enhanced)
        transformed_orig = self.psnr_transform(orig_chunks)
        transformed_chunks = self.psnr_transform(chunks)

        enhanced_psnr = psnr(
            transformed_enhacned,
            transformed_orig,
        )
        orig_psnr = psnr(
            transformed_chunks,
            transformed_orig,
        )
        enhanced_ssim = ssim(
            transformed_enhacned,
            transformed_orig,
        )
        orig_ssim = ssim(
            transformed_chunks,
            transformed_orig,
        )

        # log everything
        self.log_dict(
            {
                "val_g_loss": g_loss,
                "val_d_loss": d_loss,
                "val_d_real_loss": real_loss,
                "val_d_fakeloss": fake_loss,
                "val_psnr": enhanced_psnr,
                "val_ref_psnr": orig_psnr,
                "val_ssim": enhanced_ssim,
                "val_ref_ssim": orig_ssim,
            },
        )

    def test_step(self, batch, batch_idx):
        chunks, orig_chunks, metadata, _ = batch

        # ENHANCE!
        enhanced = self(chunks, metadata)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(chunks.size(0), 1)
        valid = valid.type_as(chunks)

        # adversarial loss is binary cross-entropy
        preds = self.discriminator(enhanced)

        g_loss = self.adversarial_loss(preds, valid)

        if batch_idx % 20 == 0:
            log = {"enhanced": [], "uncompressed": [], "decompressed": []}
            for i in range(self.num_samples):
                enh = enhanced[i].cpu()
                orig = orig_chunks[i].cpu()
                dec = chunks[i].cpu()

                log["enhanced"].append(
                    wandb.Image(enh, caption=f"Pred: {preds[i].item()}")
                )
                log["uncompressed"].append(
                    wandb.Image(orig, caption=f"uncompressed image {i}")
                )
                log["decompressed"].append(
                    wandb.Image(dec, caption=f"decompressed image {i}")
                )
            self.logger.experiment.log({f"test_{k}": v for k, v in log.items()})

        real_loss = self.adversarial_loss(self.discriminator(orig_chunks), valid)

        # how well can it label as fake?
        fake = torch.zeros(orig_chunks.size(0), 1)
        fake = fake.type_as(orig_chunks)

        fake_loss = self.adversarial_loss(preds, fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2

        transformed_enhacned = self.psnr_transform(enhanced)
        transformed_orig = self.psnr_transform(orig_chunks)
        transformed_chunks = self.psnr_transform(chunks)

        enhanced_psnr = psnr(
            transformed_enhacned,
            transformed_orig,
        )
        orig_psnr = psnr(
            transformed_chunks,
            transformed_orig,
        )
        enhanced_ssim = ssim(
            transformed_enhacned,
            transformed_orig,
        )
        orig_ssim = ssim(
            transformed_chunks,
            transformed_orig,
        )

        # log everything
        self.log_dict(
            {
                "test_g_loss": g_loss,
                "test_d_loss": d_loss,
                "test_psnr": enhanced_psnr,
                "test_ref_psnr": orig_psnr,
                "test_ssim": enhanced_ssim,
                "test_ref_ssim": orig_ssim,
            },
        )

    def predict_step(self, batch, batch_idx):
        chunks, _, metadata, chunk_objs = batch

        # ENHANCE!
        enhanced = self(chunks, metadata)

        for i, chunk_data in enumerate(enhanced):
            chunk = [c[i].cpu() if hasattr(c[i], "cpu") else c[i] for c in chunk_objs]
            VVCDataset.save_chunk(chunk, chunk_data.cpu().numpy())

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
            # {
            #     "scheduler": torch.optim.lr_scheduler.MultiStepLR(
            #         opt_d,
            #         milestones=[10],
            #         gamma=0.1,
            #     ),
            #     "interval": "epoch",
            #     "frequency": 1,
            # },
            # {
            #     "scheduler": torch.optim.lr_scheduler.MultiStepLR(
            #         opt_g,
            #         milestones=[10],
            #         gamma=0.1,
            #     ),
            #     "interval": "epoch",
            #     "frequency": 1,
            # },
        ]

        return [opt_g, opt_d], lr_schedulers
