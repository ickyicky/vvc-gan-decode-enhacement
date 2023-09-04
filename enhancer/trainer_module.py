import torch
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional.classification import accuracy
from pytorch_msssim import SSIM, MS_SSIM
from .config import TrainerConfig, TrainingMode
from .dataset import VVCDataset


class TrainerModule(pl.LightningModule):
    def __init__(
        self,
        config: TrainerConfig,
        enhancer,
        discriminator,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.enhancer = enhancer
        self.discriminator = discriminator
        self.mode = config.mode
        self.config = config.current

        self.separation_epochs = config.separation_epochs

        self.enhancer_lr = self.config.enhancer_lr
        self.discriminator_lr = self.config.discriminator_lr

        self.betas = self.config.betas
        self.momentum = self.config.momentum
        self.discriminator_min_loss = self.config.discriminator_min_loss
        self.enhancer_min_loss = self.config.enhancer_min_loss

        self.enhancer_losses = [1.0]
        self.discriminator_losses = [1.0]
        self.probe = self.config.probe

        self.num_samples = self.config.num_samples
        self.ssim = SSIM(data_range=1.0, win_size=9)
        self.msssim = MS_SSIM(data_range=1.0, win_size=9)

    def forward(self, chunks, metadata):
        return self.enhancer(chunks, metadata)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def what_to_train(self):
        if self.mode == TrainingMode.ENHANCER:
            return True, None
        elif self.mode == TrainingMode.DISCRIMINATOR:
            return None, True
        else:
            e_loss = np.mean(self.enhancer_losses)
            d_loss = np.mean(self.discriminator_losses)

            e_train = e_loss >= self.enhancer_min_loss
            d_train = d_loss >= self.discriminator_min_loss

            if not e_train and not d_train:
                return True, True

            return bool(e_train), bool(d_train)

        return True, True

    def g_step(self, chunks, orig_chunks, metadata, stage="train"):
        enhanced = self(chunks, metadata)

        prefix = "" if stage == "train" else stage + "_"

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(chunks.size(0), 1)
        valid = valid.type_as(chunks)

        # adversarial loss is binary cross-entropy
        ssim_loss = 1 - self.ssim(orig_chunks, enhanced)
        msssim_loss = 1 - self.msssim(orig_chunks, enhanced)
        mse_loss = F.mse_loss(enhanced, orig_chunks)
        l1_loss = torch.nn.functional.l1_loss(enhanced, orig_chunks)

        self.log(f"{prefix}g_l1_loss", l1_loss, prog_bar=False)
        self.log(f"{prefix}g_ssim_loss", ssim_loss, prog_bar=False)
        self.log(f"{prefix}g_msssim_loss", msssim_loss, prog_bar=False)
        self.log(f"{prefix}g_mse_loss", mse_loss, prog_bar=False)

        if self.mode == TrainingMode.GAN and self.current_epoch > self.separation_epochs:
            preds = self.discriminator(enhanced)
            gd_loss = self.adversarial_loss(preds, valid)
            g_loss = (
                0.1 * msssim_loss + 0.1 * ssim_loss + 0.35 * mse_loss + 0.35 * l1_loss + 0.1 * gd_loss
            )
            self.log(f"{prefix}g_d_loss", gd_loss, prog_bar=True)
            self.enhancer_losses.append(gd_loss.item())
            self.enhancer_losses = self.enhancer_losses[: self.probe]
        else:
            preds = None
            g_loss = (
                0.1 * msssim_loss + 0.1 * ssim_loss + 0.4 * mse_loss + 0.4 * l1_loss
            )

        self.log(f"{prefix}g_loss", g_loss, prog_bar=True)

        if stage != "train":
            enhanced_psnr = psnr(
                enhanced,
                orig_chunks,
            )
            orig_psnr = psnr(
                chunks,
                orig_chunks,
            )
            enhanced_ssim = ssim(
                enhanced,
                orig_chunks,
            )
            orig_ssim = ssim(
                chunks,
                orig_chunks,
            )
            self.log_dict(
                {
                    f"{prefix}psnr": enhanced_psnr,
                    f"{prefix}ref_psnr": orig_psnr,
                    f"{prefix}ssim": enhanced_ssim,
                    f"{prefix}ref_ssim": orig_ssim,
                },
            )

        return enhanced, preds, g_loss

    def d_step(self, fake_chunks, orig_chunks, stage="train"):
        prefix = "" if stage == "train" else stage + "_"

        valid = torch.ones(orig_chunks.size(0), 1)
        valid = valid.type_as(orig_chunks)
        real_pred = self.discriminator(orig_chunks)
        real_loss = self.adversarial_loss(real_pred, valid)
        real_accuracy = accuracy(real_pred, valid, task="binary")

        # how well can it label as fake?
        fake = torch.zeros(orig_chunks.size(0), 1)
        fake = fake.type_as(orig_chunks)

        fake_preds = self.discriminator(fake_chunks)
        fake_loss = self.adversarial_loss(fake_preds, fake)
        fake_accuracy = accuracy(fake_preds, fake, task="binary")

        acc = (real_accuracy + fake_accuracy) / 2

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2

        self.discriminator_losses.append(d_loss.item())
        self.discriminator_losses = self.discriminator_losses[: self.probe]

        self.log(f"{prefix}d_loss", d_loss, prog_bar=True)
        self.log(f"{prefix}d_real_loss", real_loss, prog_bar=False)
        self.log(f"{prefix}d_fake_loss", fake_loss, prog_bar=False)
        self.log(f"{prefix}d_real_acc", real_accuracy, prog_bar=False)
        self.log(f"{prefix}d_fake_acc", fake_accuracy, prog_bar=False)
        self.log(f"{prefix}d_acc", acc, prog_bar=True)

        return fake_preds, real_pred, d_loss

    def log_images(
        self, enhanced, chunks, orig_chunks, preds, real_preds, stage="train"
    ):
        prefix = "" if stage == "train" else stage + "_"

        log = {"uncompressed": [], "decompressed": []}

        if self.mode != TrainingMode.DISCRIMINATOR:
            log["enhanced"] = []

        for i in range(self.num_samples):
            orig = orig_chunks[i].cpu()
            dec = chunks[i].cpu()

            if self.mode != TrainingMode.DISCRIMINATOR:
                enh = enhanced[i].cpu()
                log["enhanced"].append(
                    wandb.Image(
                        enh,
                        caption=f"Pred: {preds[i].item()}"
                        if self.mode == TrainingMode.GAN
                        else f"ENH: {i}",
                    )
                )

            log["uncompressed"].append(
                wandb.Image(
                    orig,
                    caption=f"Pred: {real_preds[i].item()}"
                    if self.mode != TrainingMode.ENHANCER
                    else f"UNC: {i}",
                )
            )

            log["decompressed"].append(
                wandb.Image(
                    dec,
                    caption=f"Pred: {preds[i].item()}"
                    if self.mode == TrainingMode.DISCRIMINATOR
                    else f"DEC: {i}",
                )
            )

        log = {prefix + key: value for key, value in log.items()}
        self.logger.experiment.log(log)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        chunks, orig_chunks, metadata, _ = batch
        e_train, d_train = self.what_to_train()
        preds = None

        # train ENHANCE!
        # ENHANCE!
        if e_train:
            # with gradient
            g_opt.zero_grad()

            enhanced, preds, g_loss = self.g_step(chunks, orig_chunks, metadata)
            fake_chunks = enhanced.detach()

            self.manual_backward(g_loss)
            g_opt.step()
        elif e_train is not None:
            # just log enhancer loss
            enhanced, preds, g_loss = self.g_step(chunks, orig_chunks, metadata)
            fake_chunks = enhanced.detach()
        else:
            fake_chunks = chunks
            enhanced = None

        # train discriminator
        if d_train:
            d_opt.zero_grad()

            fake_preds, real_preds, d_loss = self.d_step(fake_chunks, orig_chunks)

            self.manual_backward(d_loss)
            d_opt.step()
        elif d_train is not None:
            fake_preds, real_preds, d_loss = self.d_step(fake_chunks, orig_chunks)
        else:
            real_preds = None
            fake_preds = preds

        if batch_idx % 100 == 0:
            self.log_images(
                enhanced,
                chunks,
                orig_chunks,
                fake_preds,
                real_preds,
            )

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
        preds = None

        # train ENHANCE!
        # ENHANCE!
        if self.mode != TrainingMode.DISCRIMINATOR:
            # with gradient
            enhanced, preds, g_loss = self.g_step(chunks, orig_chunks, metadata, "val")
            fake_chunks = enhanced.detach()
        else:
            fake_chunks = chunks
            enhanced = None

        # train discriminator
        if self.mode != TrainingMode.ENHANCER:
            fake_preds, real_preds, d_loss = self.d_step(
                fake_chunks, orig_chunks, "val"
            )
        else:
            real_preds = None
            fake_preds = preds

        if batch_idx % 100 == 0:
            self.log_images(
                enhanced,
                chunks,
                orig_chunks,
                fake_preds,
                real_preds,
                "val",
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
            VVCDataset.save_chunk(
                chunk, chunk_data.cpu().numpy(), self.config.saved_chunk_folder
            )

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.enhancer.parameters(),
            lr=self.enhancer_lr,
            betas=self.betas,
            weight_decay=self.enhancer_lr / 10,
        )
        opt_d = torch.optim.SGD(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
            momentum=self.momentum,
            weight_decay=self.discriminator_lr / 10,
        )

        lr_schedulers = []

        if self.config.enhancer_scheduler is True:
            lr_schedulers.append(
                {
                    "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                        opt_g,
                        milestones=self.config.enhancer_scheduler_milestones,
                        gamma=self.config.enhancer_scheduler_gamma,
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                }
            )

        if self.config.discriminator_scheduler is True:
            lr_schedulers.append(
                {
                    "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                        opt_d,
                        milestones=self.config.discriminator_scheduler_milestones,
                        gamma=self.config.discriminator_scheduler_gamma,
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                }
            )

        return [opt_g, opt_d], lr_schedulers
