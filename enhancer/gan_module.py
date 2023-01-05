import torch
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Tuple


class GANModule(pl.LightningModule):
    def __init__(
        self,
        enhancer,
        discriminator,
        enhancer_lr: float = 0.0002,
        discriminator_lr: float = 0.0001,
        betas: Tuple[float, float] = (0.5, 0.999),
    ):
        super().__init__()

        self.enhancer = enhancer
        self.discriminator = discriminator

        self.enhancer_lr = enhancer_lr
        self.discriminator_lr = discriminator_lr

        self.betas = betas

    def forward(self, chunks, metadata):
        return self.enhancer(chunks, metadata)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        chunks, orig_chunks, metadata = batch

        # train ENHANCE!
        if optimizer_idx == 0:

            # ENHANCE!
            self.enhanced = self(chunks, metadata)

            # log ENHANCED!
            sample = self.enhanced[:6]
            grid = torchvision.utils.make_grid(sample)
            self.logger.experiment.add_image("enhanced", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(chunks.size(0), 1)
            valid = valid.type_as(chunks)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self.enhanced), valid)
            self.log("g_loss", g_loss, prog_bar=True)
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

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.enhancer.parameters(), lr=self.enhancer_lr, betas=self.betas
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.discriminator_lr, betas=self.betas
        )

        lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
            for optimizer in (opt_g, opt_d)
        ]

        return [opt_g, opt_d], lr_schedulers

    def on_validation_epoch_end(self):
        # log sampled images
        sample_imgs = self.enhanced
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("enhanced", grid, self.current_epoch)
