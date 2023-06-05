import torch


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
