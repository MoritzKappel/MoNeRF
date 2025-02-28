# -- coding: utf-8 --

"""
MoNeRF/utils.py: Utility functions.
"""

import torch


def pseudoColorSceneFlow(flow: torch.Tensor, alpha: torch.Tensor | None = None, norm_max_val: float | None = None) -> torch.Tensor:
    """pseudo-colors rendered 3D deformations using rgb cube"""
    if norm_max_val is None:
        norm_max_val = torch.max(torch.abs(flow)) + 1.0e-8
    flow_normalized: torch.Tensor = torch.clamp(((flow / norm_max_val) + 1.0) * 0.5, min=0.0, max=1.0)
    if alpha is not None:
        flow_normalized *= alpha
    return flow_normalized
