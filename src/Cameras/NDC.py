# -- coding: utf-8 --

"""Cameras/NDC.py: A perspective RGB camera model generating rays in normalized device space."""

from torch import Tensor
import torch

from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import RayPropertySlice


class NDCCamera(PerspectiveCamera):
    """Defines a perspective RGB camera model that generates rays in normalized device space."""

    def __init__(self, cube_scale: float = 1.0) -> None:
        super(NDCCamera, self).__init__(0.01, 1.0)
        self.cube_scale: float = cube_scale

    def generateRays(self) -> Tensor:
        # generate perspective rays
        rays: Tensor = super().generateRays()
        origins: Tensor = rays[:, RayPropertySlice.origin]
        directions: Tensor = rays[:, RayPropertySlice.direction]
        # shift rays to near plane
        t: Tensor = -(1.0 + origins[..., 2]) / directions[..., 2]
        origins: Tensor = origins + t[..., None] * directions
        # precompute some intermediate results
        w2fx: float = self.properties.width / (2. * self.properties.focal_x)
        h2fy: float = self.properties.height / (2. * self.properties.focal_y)
        ox_oz: Tensor = origins[..., 0] / origins[..., 2]
        oy_oz: Tensor = origins[..., 1] / origins[..., 2]
        # projection to normalized device space
        o0: Tensor = -1. / w2fx * ox_oz
        o1: Tensor = -1. / h2fy * oy_oz
        o2: Tensor = 1. + 2. * 1.0 / origins[..., 2]
        d0: Tensor = -1. / w2fx * (directions[..., 0] / directions[..., 2] - ox_oz)
        d1: Tensor = -1. / h2fy * (directions[..., 1] / directions[..., 2] - oy_oz)
        d2: Tensor = 1. - o2
        ndc_rays: Tensor = torch.cat([
            torch.stack([o0, o1, o2], -1) * self.cube_scale,
            torch.stack([d0, d1, d2], -1) * self.cube_scale,
            rays[:, slice(6, 18)]
        ], dim=-1)
        return ndc_rays
