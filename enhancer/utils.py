import torch
from tqdm import tqdm


def weights_init(model) -> None:
    """weights_init.
    :param model:
    :rtype: None
    """
    classname = model.__class__.__name__

    if "Conv" in classname:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0)


def calculate_mean_std(orig_chunks_dir, test_orig_chunks_dir):
    from .dataset import OnlyOrigVVCDataset
    from torchvision import transforms
    from torch.utils.data import DataLoader, ConcatDataset

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    def metadata_transform(metadata):
        return torch.as_tensor(metadata).float().view(len(metadata), 1, 1)

    dataset = OnlyOrigVVCDataset(orig_chunks_dir, transform)
    test_dataset = OnlyOrigVVCDataset(
        test_orig_chunks_dir,
        transform,
    )
    loader = DataLoader(
        ConcatDataset([dataset, test_dataset]),
        batch_size=16 * 1024,
        num_workers=0,
        shuffle=False,
    )

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    num_batches = len(loader)

    for orig_chunks in tqdm(
        loader,
        desc="calculating mean std..",
        leave=True,
        position=0,
    ):
        channels_sum += torch.mean(orig_chunks, dim=[0, 2, 3]) / num_batches
        channels_squared_sum += (
            torch.mean(orig_chunks**2, dim=[0, 2, 3]) / num_batches
        )

    mean = channels_sum
    std = (channels_squared_sum - mean**2) ** 0.5
    return (mean, std)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--calculate-mean-std",
        "-c",
        action="store_true",
        help="calculate mean and std for given chunk folder",
    )

    parser.add_argument(
        "--chunks-dir",
        "-d",
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

    args = parser.parse_args()

    if args.calculate_mean_std:
        mean, std = calculate_mean_std(
            args.orig_chunks_dir,
            args.test_orig_chunks_dir,
        )
        print("mean: ", mean)
        print("std: ", std)
