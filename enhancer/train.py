from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch
import wandb

from dataclasses import asdict

from .dataset import VVCDataset
from .config import TrainingConfiguration

from .generator import DenseGenerator
from .discriminator import Discriminator


def get_dataloader(
    data_path: str,
    encoded_path: str,
    batch_size: int = 1,
    num_workers: int = 4,
):
    dataset = VVCDataset(data_path, encoded_path)
    return (
        DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        ),
        dataset,
    )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_net(
    generator,
    discriminator,
    device,
    config: TrainingConfiguration,
):
    generator.to(device=device)
    discriminator.to(device=device)

    data_loader, dataset = get_dataloader(**asdict(config.data))
    criterion = nn.BCELoss()
    real_label, fake_label = 1.0, 0.0

    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999)
    )
    optimizer_g = optim.Adam(
        generator.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999)
    )

    # initialize experiment
    experiment = wandb.init(project="DenseGAN", resume="allow", anonymous="must")
    experiment.config.update(
        dict(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
        )
    )

    step = 0
    for epoch in range(1, config.epochs + 1):
        with tqdm(
            total=len(dataset), desc=f"Epoch {epoch}/{config.epochs}", unit="chunk"
        ) as pbar:
            for batch in data_loader:
                inputs = batch[0]
                originals = batch[1]
                metadatas = batch[2]
                _batch_size = inputs.shape[0]

                # perform discriminator training on all original data
                discriminator.zero_grad()
                originals = originals.to(device)
                label = torch.full(
                    (_batch_size,), real_label, dtype=torch.float, device=device
                )
                output = discriminator(originals)
                output = output.view(-1)
                err_dis_real = criterion(output, label)
                err_dis_real.backward()
                D_x = output.mean().item()

                # perform discriminator training on all fake data
                inputs = inputs.to(device)
                metadatas = metadatas.to(device)
                fake = generator(inputs, metadatas)
                label.fill_(fake_label)
                output = discriminator(fake.detach()).view(-1)
                err_dis_fake = criterion(output, label)
                err_dis_fake.backward()
                D_G_z1 = output.mean().item()

                err_dis = err_dis_fake + err_dis_real
                optimizer_d.step()

                # update generator network
                generator.zero_grad()
                label.fill_(real_label)
                output = discriminator(fake.detach()).view(-1)
                err_gen = criterion(output, label)
                err_gen.backward()
                D_G_z2 = output.mean().item()
                optimizer_g.step()
                pbar.update(_batch_size)

                step += 1
                if step % config.upload_rate == 0:
                    # update training stats
                    experiment.log(
                        {
                            "learning rate": {
                                "generator": optimizer_g.param_groups[0]["lr"],
                                "discriminator": optimizer_d.param_groups[0]["lr"],
                            },
                            "images": {
                                "input": wandb.Image(inputs[0].cpu()),
                                "real": wandb.Image(originals[0].cpu()),
                                "fake": wandb.Image(fake[0].cpu()),
                            },
                            "epoch": epoch,
                            "D": {
                                "x": D_x,
                                "z1": D_G_z1,
                                "z2": D_G_z2,
                            },
                            "error": {
                                "dis_real": err_dis_real.item(),
                                "dis_fake": err_dis_fake.item(),
                                "dis": err_dis.item(),
                                "gen": err_gen.item(),
                            },
                        }
                    )


if __name__ == "__main__":
    config = TrainingConfiguration.parse_args()
    discriminator = Discriminator()
    discriminator.apply(weights_init)
    generator = DenseGenerator()
    generator.apply(weights_init)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_net(generator, discriminator, device, config)
