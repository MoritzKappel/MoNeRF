# -- coding: utf-8 --

"""HierarchicalNeRF/Loss.py: Loss implementation for the hierarchical NeRF method."""

from torch import Tensor
import torch
import torchmetrics

from Cameras.utils import RayPropertySlice
from Losses.Base import BaseLoss
from Losses.utils import LossMetricItem, QualityMetricItem


class HierarchicalNeRFLoss(BaseLoss):
    """Defines a class for all sub-losses of the hierarchical NeRF method."""

    def __init__(self, lambda_color: float, lambda_alpha: float, activate_logging: bool = False) -> None:
        super().__init__(
            loss_metrics=[
                LossMetricItem(name='rgb', metric_func=torch.nn.functional.mse_loss, weight=lambda_color),
                LossMetricItem(name='rgb_coarse', metric_func=torch.nn.functional.mse_loss, weight=lambda_color),
                LossMetricItem(name='alpha', metric_func=torch.nn.functional.mse_loss, weight=lambda_alpha),
                LossMetricItem(name='alpha_coarse', metric_func=torch.nn.functional.mse_loss, weight=lambda_alpha)
            ],
            quality_metrics=[
                QualityMetricItem(name='PSNR', metric_func=torchmetrics.functional.peak_signal_noise_ratio),
                QualityMetricItem(name='PSNR_coarse', metric_func=torchmetrics.functional.peak_signal_noise_ratio)
            ],
            activate_logging=activate_logging
        )

    def forward(self, outputs: dict[str, Tensor | None], rays: Tensor) -> Tensor:
        """Defines loss calculation."""
        return super().forward({
            'rgb': {'input': outputs['rgb'], 'target': rays[:, RayPropertySlice.rgb]},
            'rgb_coarse': {'input': outputs['rgb_coarse'], 'target': rays[:, RayPropertySlice.rgb]},
            'alpha': {'input': outputs['alpha'], 'target': rays[:, RayPropertySlice.alpha]},
            'alpha_coarse': {'input': outputs['alpha_coarse'], 'target': rays[:, RayPropertySlice.alpha]},
            'PSNR': {'preds': outputs['rgb'], 'target': rays[:, RayPropertySlice.rgb], 'data_range': 1.0},
            'PSNR_coarse': {'preds': outputs['rgb_coarse'], 'target': rays[:, RayPropertySlice.rgb], 'data_range': 1.0}
        })
