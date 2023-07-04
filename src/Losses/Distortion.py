# -- coding: utf-8 --

"""Losses/Distortion.py: Distortion loss as introduced in MipNeRF360 / DVGOv2."""

from typing import Any
import torch
import torchmetrics

import Implementations


class DistortionLoss:

    def __init__(self) -> None:
        self.loss_cls = Implementations.CudaExtensions.getExtension('VolumeRenderingV2').DistortionLoss

    def __call__(self, ws: torch.Tensor, deltas: torch.Tensor, ts: torch.Tensor, rays_a: torch.Tensor) -> torch.Tensor:
        return self.loss_cls.apply(ws, deltas, ts, rays_a).mean()


class DistortionLossTorchMetrics(torchmetrics.Metric):
    """torchmetrics implementation of the efficient Distortion loss introduced in MipNeRF360 / DVGOv2"""
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("running_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.loss_cls = Implementations.CudaExtensions.getExtension('VolumeRenderingV2').DistortionLoss

    def update(self, ws: torch.Tensor, deltas: torch.Tensor, ts: torch.Tensor, rays_a: torch.Tensor) -> None:
        """Update state."""
        x = self.loss_cls.apply(ws, deltas, ts, rays_a)
        y = x.sum()
        self.running_sum = self.running_sum + y
        self.total += x.numel()

    def compute(self) -> torch.Tensor:
        """Computes distortion loss over state."""
        return (self.running_sum / self.total) if self.total > 0 else torch.tensor(0.0)
