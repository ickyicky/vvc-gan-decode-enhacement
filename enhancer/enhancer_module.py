import torch
import wandb
import pytorch_lightning as pl
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from typing import Tuple


class EnhancerModule(pl.LightningModule):
    def __init__(
        self,
        enhancer,
        enhancer_lr: float = 0.005,
        betas: Tuple[float, float] = (0.5, 0.999),
        num_samples: int = 6,
    ):
        super().__init__()

        self.enhancer = enhancer

        self.enhancer_lr = enhancer_lr

        self.betas = betas

        self.num_samples = num_samples
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
            kernel_size=7, data_range=1.0, normalize="simple"
        )

    def forward(self, chunks, metadata):
        return self.enhancer(chunks, metadata)

    def training_step(self, batch, batch_idx):
        chunks, orig_chunks, metadata = batch

        # ENHANCE!
        enhanced = self(chunks, metadata)

        # adversarial loss is binary cross-entropy
        g_loss = self.ms_ssim(enhanced, orig_chunks)
        self.log("loss", g_loss, prog_bar=True)
        if batch_idx % 20 == 0:
            self.logger.experiment.log(
                {
                    "examples": [
                        wandb.Image(
                            x,
                            caption="enhanced image {i}",
                        )
                        for i, x in enumerate(enhanced[: self.num_samples])
                    ],
                    "reference": [
                        wandb.Image(
                            x,
                            caption="reference image {i}",
                        )
                        for i, x in enumerate(orig_chunks[: self.num_samples])
                    ],
                }
            )
        return g_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.enhancer.parameters(), lr=self.enhancer_lr, betas=self.betas
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(opt_g, step_size=3, gamma=0.1)

        return [opt_g], [lr_scheduler]
