"""
Modified from torchmetrics implementation:
https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/fid.py
"""
import torch
from torch import Tensor
from torch.autograd import Function
from typing import Any
import scipy
import numpy as np


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    All credit to `Square Root of a Positive Definite Matrix`_
    """

    @staticmethod
    def forward(ctx: Any, input_data: Tensor) -> Tensor:
        """Forward pass for the matrix square root."""
        # TODO: update whenever pytorch gets an matrix square root function
        # Issue: https://github.com/pytorch/pytorch/issues/9983
        m = input_data.detach().cpu().numpy().astype(np.float_)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
        sqrtm = torch.from_numpy(scipy_res.real).to(input_data)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        """Backward pass for matrix square root."""
        if not ctx.needs_input_grad[0]:
            return None
        (sqrtm,) = ctx.saved_tensors
        sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
        gm = grad_output.data.cpu().numpy().astype(np.float_)

        # Given a positive semi-definite matrix X,
        # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
        # matrix square root dX^{1/2} by solving the Sylvester equation:
        # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
        grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

        return torch.from_numpy(grad_sqrtm).to(grad_output)


sqrtm = MatrixSquareRoot.apply


def _compute_fid(
    mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = 1e-6
) -> Tensor:
    r"""Compute adjusted version of `Fid Score`_.
    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).
    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant - used if sigma_1 @ sigma_2 matrix is singular
    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2

    covmean = sqrtm(sigma1.mm(sigma2))
    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


def compute_fid(
    real_features: Tensor, fake_features: Tensor, eps: float = 1e-6
) -> Tensor:
    real_features_sum = real_features.sum(dim=0)
    real_features_cov_sum = real_features.t().mm(real_features)
    real_features_num_samples = real_features.shape[0]

    fake_features_sum = fake_features.sum(dim=0)
    fake_features_cov_sum = fake_features.t().mm(fake_features)
    fake_features_num_samples = fake_features.shape[0]

    mean_real = (real_features_sum / real_features_num_samples).unsqueeze(0)
    mean_fake = (fake_features_sum / fake_features_num_samples).unsqueeze(0)

    cov_real_num = real_features_cov_sum - real_features_num_samples * mean_real.t().mm(
        mean_real
    )
    cov_real = cov_real_num / (real_features_num_samples - 1)
    cov_fake_num = fake_features_cov_sum - fake_features_num_samples * mean_fake.t().mm(
        mean_fake
    )
    cov_fake = cov_fake_num / (fake_features_num_samples - 1)
    return _compute_fid(
        mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake
    ).to(real_features.dtype)
