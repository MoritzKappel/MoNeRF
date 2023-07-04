# -- coding: utf-8 --

"""Cameras/Base.py: Implementation of the basic camera model used for ray generation and scene rendering options."""

from abc import ABC, abstractmethod
import torch
from torch import Tensor

import Framework
from Cameras.utils import CameraProperties


class BaseCamera(ABC):
    """Defines the basic camera template for ray generation."""

    def __init__(self, near_plane: float, far_plane: float) -> None:
        super().__init__()
        self.near_plane: float = near_plane
        self.far_plane: float = far_plane
        self.background_color: Tensor = torch.tensor([1.0, 1.0, 1.0]).type(Framework.config.GLOBAL.DEFAULT_TENSOR_TYPE)
        self.properties: CameraProperties = CameraProperties()

    def setBackgroundColor(self, r: float = 1.0, g: float = 1.0, b: float = 1.0):
        self.background_color = torch.tensor([r, g, b]).type(Framework.config.GLOBAL.DEFAULT_TENSOR_TYPE)

    def setProperties(self, properties: CameraProperties) -> 'BaseCamera':
        """Sets the given sensor data sample for the camera."""
        self.properties = properties
        return self

    def generateRays(self) -> Tensor:
        """Generates tensor containing all rays and their properties according to camera model."""
        # check if precomputed rays exist
        if self.properties._precomputed_rays is not None:
            return self.properties._precomputed_rays
        # get directions
        directions: Tensor = self.getRayDirections()
        view_directions: Tensor = directions / torch.norm(directions, p=2, dim=-1, keepdim=True)
        # get origins
        origins: Tensor = self.getRayOrigins()
        # get annotations
        annotations: Tensor = self.getRayAnnotations()
        # build rays tensor
        rays: Tensor = torch.cat([origins, directions, view_directions, annotations], dim=-1)
        return rays

    def getRayAnnotations(self) -> Tensor:
        """Returns a tensor containing the annotations (e.g. color, alpha, and depth) of each ray."""
        # add color if ground truth rgb image is available
        if self.properties.rgb is not None:
            img_flat: Tensor = self.properties.rgb.permute(1, 2, 0).reshape(-1, 3)
        else:
            # add empty color if no ground truth view is available
            img_flat: Tensor = torch.zeros((self.properties.height * self.properties.width, 3))
        # add alpha if ground truth alpha mask is available
        if self.properties.alpha is not None:
            alpha_flat: Tensor = self.properties.alpha.permute(1, 2, 0).reshape(-1, 1)
        else:
            # assume every pixel is valid
            alpha_flat: Tensor = torch.ones((self.properties.height * self.properties.width, 1))
        # add depth if ground truth depth mask is available
        if self.properties.depth is not None:
            depth_flat: Tensor = self.properties.depth.permute(1, 2, 0).reshape(-1, 1)
        else:
            # set depth to -1 for every pixel
            depth_flat: Tensor = torch.full((self.properties.height * self.properties.width, 1), fill_value=-1)
        # add timestamp
        if self.properties.timestamp is not None:
            timestamps_flat = torch.full(
                (self.properties.height * self.properties.width, 1), fill_value=float(self.properties.timestamp)
            )
        else:
            # set to 0
            timestamps_flat = torch.zeros((self.properties.height * self.properties.width, 1))
        # add xy image coordinates
        x_coord, y_coord = self.getPixelCoordinates()
        x_coord = x_coord[:, None].reshape(-1, 1)
        y_coord = y_coord[:, None].reshape(-1, 1)
        # combine and return
        return torch.cat([img_flat, alpha_flat, depth_flat, timestamps_flat, x_coord, y_coord], dim=-1)

    @abstractmethod
    def getRayOrigins(self) -> Tensor:
        """Returns a tensor containing the origin of each ray."""
        return Tensor()

    @abstractmethod
    def getRayDirections(self) -> Tensor:
        """Returns a tensor containing the direction of each ray."""
        return Tensor()

    def getPositionAndViewdir(self) -> Tensor | None:
        """Returns the current camera position and direction in world coordinates."""
        if self.properties is None:
            return None
        data: Tensor = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0]])
        return torch.matmul(self.properties.c2w, data[:, :, None]).squeeze()

    def getPixelCoordinates(self) -> tuple[Tensor, Tensor]:
        y_direction, x_direction = torch.meshgrid(
            torch.linspace(0, self.properties.height - 1, self.properties.height),
            torch.linspace(0, self.properties.width - 1, self.properties.width),
            indexing="ij"
        )
        return x_direction, y_direction

    @abstractmethod
    def projectPoint(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """Projects points (Nx3) to image plane. returns xy coordinates and mask of points that hit sensor."""
        pass
