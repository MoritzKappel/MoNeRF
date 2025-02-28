# -- coding: utf-8 --

"""MoNeRF/Loss.py: Full MoNeRF training loss function."""

import torch
import torchmetrics

from Cameras.utils import RayPropertySlice
from Optim.Losses.Base import BaseLoss
from Optim.Losses.BackgroundEntropy import backgroundEntropy
from Optim.Losses.Distortion import DistortionLoss
from Optim.Losses.Magnitude import magnitudeLoss


class MoNeRFLoss(BaseLoss):
    def __init__(self, lambda_bg_entropy: float, lambda_flow_magnitude: float, lambda_distortion: float) -> None:
        super().__init__()
        self.addLossMetric('MSE_Color', torch.nn.functional.mse_loss, 1.0)
        self.addLossMetric('Flow_Magnitude', magnitudeLoss, lambda_flow_magnitude)
        self.addLossMetric('Background_Entropy', backgroundEntropy, lambda_bg_entropy)
        self.addLossMetric('Distortion', DistortionLoss(), lambda_distortion)
        self.addQualityMetric('PSNR', torchmetrics.functional.peak_signal_noise_ratio)

    def forward(self, outputs: dict[str, torch.Tensor | None], rays: torch.Tensor, bg_color: torch.Tensor) -> torch.Tensor:
        color_gt = rays[:, RayPropertySlice.rgb]
        if bg_color is not None:
            color_gt = (rays[:, RayPropertySlice.alpha] * color_gt) + ((1.0 - rays[:, RayPropertySlice.alpha]) * bg_color)
        return super().forward({
            'MSE_Color': {'input': outputs['rgb'], 'target': color_gt},
            'Flow_Magnitude': {'input': outputs['delta_x']},
            'Background_Entropy': {'input': outputs['alpha']},
            'Distortion': {'ws': outputs['ws'], 'deltas': outputs['deltas'], 'ts': outputs['ts'], 'rays_a': outputs['rays_a']},
            'PSNR': {'preds': outputs['rgb'], 'target': color_gt, 'data_range': 1.0}
        })
