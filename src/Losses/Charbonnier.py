# -- coding: utf-8 --

"""Losses/Magnitude.py: Charbonnier loss as used by MipNeRF 360."""

from typing import Any

import torch
import torchmetrics


def charbonnierLoss(result: torch.Tensor, gt: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """functional"""
    return ((result - gt) ** 2 + eps).sqrt().mean()


class CharbonnierLoss(torchmetrics.Metric):
    """torchmetrics implementation"""
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, eps: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("running_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.eps = eps

    def update(self, result: torch.Tensor, gt: torch.Tensor) -> None:
        """torchmetrics update override"""
        x = ((result - gt) ** 2 + self.eps).sqrt()
        self.running_sum += x.sum()
        self.total += x.numel()

    def compute(self) -> torch.Tensor:
        """torchmetrics compute override"""
        return (self.running_sum / self.total) if self.total > 0 else (torch.tensor(0.0))
