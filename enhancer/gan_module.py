import torch
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.transforms.functional import crop
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from typing import Tuple
from .crosslid import compute_crosslid
from .models.discriminator import WrapperInception
from .dataset import VVCDataset
from pytorch_msssim import SSIM, MS_SSIM


class GANModule(pl.LightningModule):
    def __init__(
        self,
        enhancer,
        discriminator,
        enhancer_lr: float = 1e-4,
        discriminator_lr: float = 3e-6,
        betas: Tuple[float, float] = (0.5, 0.999),
        num_samples: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.enhancer = enhancer
        self.discriminator = discriminator
        self.wrapper_inception = WrapperInception(discriminator)

        self.enhancer_lr = enhancer_lr
        self.discriminator_lr = discriminator_lr

        self.betas = betas

        self.num_samples = num_samples
        self.ssim = SSIM(data_range=1.0, win_size=9)
        self.msssim = MS_SSIM(data_range=1.0, win_size=9)

    def psnr_transform(self, output):
        # crop removes area that is gradiented
        return output

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
        chunks, orig_chunks, metadata, _ = batch

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
            ssim_loss = 1 - self.ssim(orig_chunks, enhanced)
            msssim_loss = 1 - self.msssim(orig_chunks, enhanced)
            mse_loss = F.mse_loss(enhanced, orig_chunks)
            g_loss = 0.4 * g_loss + 0.2 * msssim_loss + 0.2 * ssim_loss + 0.2 * mse_loss

            self.log("g_loss", g_loss, prog_bar=True)

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
        chunks, orig_chunks, metadata, _ = batch

        # ENHANCE!
        enhanced = self(chunks, metadata)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(chunks.size(0), 1)
        valid = valid.type_as(chunks)

        # adversarial loss is binary cross-entropy
        preds = self.discriminator(enhanced)
        y_hat_features = self.wrapper_inception(enhanced)

        g_loss = self.adversarial_loss(preds, valid)
        ssim_loss = 1 - self.ssim(orig_chunks, enhanced)
        msssim_loss = 1 - self.msssim(orig_chunks, enhanced)
        mse_loss = F.mse_loss(enhanced, orig_chunks)
        g_loss = 0.4 * g_loss + 0.2 * msssim_loss + 0.2 * ssim_loss + 0.2 * mse_loss

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
        y_features = self.wrapper_inception(orig_chunks)

        # how well can it label as fake?
        fake = torch.zeros(orig_chunks.size(0), 1)
        fake = fake.type_as(orig_chunks)

        fake_loss = self.adversarial_loss(preds, fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2

        # calculate crosslid
        crosslid = self.crosslid(y_hat_features, y_features)

        orig_features = self.wrapper_inception(orig_chunks)
        orig_crosslid = self.crosslid(orig_features, y_features)

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
                "val_crosslid": crosslid,
                "val_ref_crosslid": orig_crosslid,
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
        y_hat_features = self.wrapper_inception(enhanced)

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
        y_features = self.wrapper_inception(orig_chunks)

        # how well can it label as fake?
        fake = torch.zeros(orig_chunks.size(0), 1)
        fake = fake.type_as(orig_chunks)

        fake_loss = self.adversarial_loss(preds, fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2

        # calculate crosslid, this time as well for decompressed images
        crosslid = self.crosslid(y_hat_features, y_features)
        orig_features = self.wrapper_inception(chunks)
        orig_crosslid = self.crosslid(orig_features, y_features)

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
                "test_crosslid": crosslid,
                "test_ref_crosslid": orig_crosslid,
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

        return [opt_g, opt_d], []
