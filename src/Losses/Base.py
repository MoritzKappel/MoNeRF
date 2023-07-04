# -- coding: utf-8 --

"""Losses/Base.py: Base Loss class for accumulation and logging."""

from typing import Any, Callable

import torch

import Framework
from Losses.utils import LossError, QualityMetricItem, LossMetricItem


class BaseLoss(torch.nn.Module):
    """Simple configurable loss container for accumulation and wandb logging"""

    def __init__(self,
                 loss_metrics: list[LossMetricItem] | None = None,
                 quality_metrics: list[QualityMetricItem] | None = None,
                 activate_logging: bool = True) -> None:
        super().__init__()
        self.loss_metrics: list[LossMetricItem] = loss_metrics or []
        self.quality_metrics: list[QualityMetricItem] = quality_metrics or []
        self.activate_logging: bool = activate_logging

    def addLossMetric(self, name: str, metric: Callable, weight: float = None) -> None:
        self.loss_metrics.append(LossMetricItem(
            name=name,
            metric_func=metric,
            weight=weight
        ))

    def addQualityMetric(self, name: str, metric: Callable) -> None:
        self.quality_metrics.append(QualityMetricItem(
            name=name,
            metric_func=metric,
        ))

    def reset(self) -> None:
        for item in self.loss_metrics + self.quality_metrics:
            item.reset()

    def log(self, iteration: int) -> None:
        if self.activate_logging:
            for item in self.loss_metrics + self.quality_metrics:
                val_train, val_eval = item.getAverage()
                Framework.wandb.log({f'{item.name}': {'train': val_train, 'eval': val_eval}}, step=iteration)

    def forward(self, configurations: dict[str, dict[str, Any]]) -> torch.Tensor:
        try:
            if self.activate_logging:
                with torch.no_grad():
                    for item in self.quality_metrics:
                        item.apply(train=self.training, accumulate=True, kwargs=configurations[item.name])
            return torch.stack([item.apply(train=self.training, accumulate=self.activate_logging, kwargs=configurations[item.name]) for item in self.loss_metrics]).sum()
        except NameError:
            raise LossError(f'missing argument configuration for loss "{item.name}"')
        except TypeError:
            raise LossError(f'invalid argument configuration for loss "{item.name}"')
