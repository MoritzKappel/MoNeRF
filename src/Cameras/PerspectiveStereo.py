# -- coding: utf-8 --

"""Cameras/PerspectiveStereo.py: Implementation of a perspective camera model generating stereo views."""

from torch import Tensor
import torch

from Cameras.Perspective import PerspectiveCamera


class PerspectiveStereoCamera(PerspectiveCamera):
    """Defines a perspective camera model that generates rays of vertically concatenated stereo views."""

    def __init__(self, near_plane: float, far_plane: float, baseline: float = 0.062) -> None:
        super(PerspectiveStereoCamera, self).__init__(near_plane, far_plane)
        self.half_baseline: float = baseline / 2.0

    def getRayOrigins(self) -> Tensor:
        """Returns a tensor containing the origin of each ray."""
        x_axis_world_space: Tensor = self.properties.c2w[:3, 0][None]
        # adjust height for vertical concatenation
        self.properties.height = self.properties.height // 2
        origin_world_space: Tensor = super().getRayOrigins()
        self.properties.height = self.properties.height * 2
        origins_left: Tensor = origin_world_space - (x_axis_world_space * self.half_baseline)
        origins_right: Tensor = origin_world_space + (x_axis_world_space * self.half_baseline)
        return torch.cat([origins_left, origins_right], dim=0)

    def getRayDirections(self) -> Tensor:
        """Returns a tensor containing the direction of each ray."""
        # adjust height for vertical concatenation
        self.properties.height = self.properties.height // 2
        directions: Tensor = super().getRayDirections()
        self.properties.height = self.properties.height * 2
        return torch.cat([directions, directions], dim=0)

    def projectPoint(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """projects points (Nx3) to image plane. returns xy coordinates and mask of points that hit sensor"""
        raise NotImplementedError('point projection not yet implemented for stereo cameras')
