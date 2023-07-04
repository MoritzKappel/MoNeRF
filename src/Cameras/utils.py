# -- coding: utf-8 --

"""Cameras/utils.py: Contains utility functions used for the implementation of the available camera models."""

from typing import Callable, ClassVar
from torch import Tensor
from dataclasses import dataclass, fields, replace
from enum import Enum

import Framework


@dataclass
class CameraProperties:
    """
    A Class for all kinds of image sensor data.

    Attributes:
        width               Integer representing the image's width.
        height              Integer representing the image's height.
        rgb                 Torch tensor of shape (3, H, W) containing the RGB image.
        alpha               Torch tensor of shape (1, H, W) containing the foreground mask.
        depth               Torch tensor of shape (1, H, W) containing the depth image.
        c2w                 Torch tensor of shape (4, 4) containing the camera to world matrix.
        w2c                 Torch tensor of shape (4, 4) containing the world to camera matrix.
        principal_offset_x  Float stating the principal point's offset from the image center in pixels (positive value -> right)
        principal_offset_y  Float stating the principal point's offset from the image center in pixels (positive value -> down)
        focal_x             Float representing the image sensor's focal length (in pixels) in x direction.
        focal_y             Float representing the image sensor's focal length (in pixels) in y direction.
        timestamp           Float stating normalized chronological timestamp at which the sample was recorded.
    """
    width: int | None = None
    height: int | None = None
    rgb: Tensor | None = None
    alpha: Tensor | None = None
    depth: Tensor | None = None
    c2w: Tensor | None = None
    w2c: Tensor | None = None
    principal_offset_x: float = 0.0
    principal_offset_y: float = 0.0
    focal_x: float | None = None
    focal_y: float | None = None
    timestamp: float = 0.0
    _precomputed_rays: Tensor | None = None

    def toDefaultDevice(self) -> 'CameraProperties':
        """returns a shallow copy where all torch Tensors are cast to the default device"""
        return replace(self, **{field.name: getattr(self, field.name).type(Framework.config.GLOBAL.DEFAULT_TENSOR_TYPE)
                                for field in fields(self) if isinstance(getattr(self, field.name), Tensor)})


@dataclass(frozen=True)
class RayPropertySlice:
    """Stores slice objects used to access specific elements within ray tensors."""
    origin: ClassVar[slice] = slice(0, 3)
    origin_xy: ClassVar[slice] = slice(0, 2)
    origin_xz: ClassVar[slice] = slice(0, 3, 2)
    origin_yz: ClassVar[slice] = slice(1, 3)
    direction: ClassVar[slice] = slice(3, 6)
    direction_xy: ClassVar[slice] = slice(3, 5)
    direction_xz: ClassVar[slice] = slice(3, 6, 2)
    direction_yz: ClassVar[slice] = slice(4, 6)
    view_direction: ClassVar[slice] = slice(6, 9)
    rgb: ClassVar[slice] = slice(9, 12)
    alpha: ClassVar[slice] = slice(12, 13)
    rgba: ClassVar[slice] = slice(9, 13)
    depth: ClassVar[slice] = slice(13, 14)
    timestamp: ClassVar[slice] = slice(14, 15)
    xy_coordinates: ClassVar[slice] = slice(15, 17)
    x_coordinate: ClassVar[slice] = slice(15, 16)
    y_coordinate: ClassVar[slice] = slice(16, 17)


@dataclass
class CoordinateSystemTransformations(Enum):
    """Stores lambda functions for axis orientation of common coordinate systems."""
    OPENGL: Callable = lambda x, y, z: [x, y, z]  # no changes because this is the internal representation
    RIGHT_HAND: Callable = lambda x, y, z: [x, -y, z]
    LEFT_HAND: Callable = lambda x, y, z: [x, y, -z]
    PYTORCH3D: Callable = lambda x, y, z: [-x, -y, -z]
    REPLICA: Callable = lambda x, y, z: [x, -y, -z]


class CameraError(Framework.FrameworkError):
    """Raise in case of an exception regarding a camera model."""
