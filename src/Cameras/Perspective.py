# -- coding: utf-8 --

"""Cameras/Perspective.py: Implementation of a perspective RGB camera model."""

from torch import Tensor
import torch

from Cameras.Base import BaseCamera


class PerspectiveCamera(BaseCamera):
    """Defines a perspective RGB camera model for ray generation."""

    def __init__(self, near_plane: float, far_plane: float) -> None:
        super(PerspectiveCamera, self).__init__(near_plane, far_plane)

    def getRayOrigins(self) -> Tensor:
        """Returns a tensor containing the origin of each ray."""
        return self.properties.c2w[:3, -1].expand((self.properties.width * self.properties.height, 3))

    def getRayDirections(self) -> Tensor:
        """Returns a tensor containing the direction of each ray."""
        # calculate initial directions
        x_direction, y_direction = self.getPixelCoordinates()
        x_direction: Tensor = ((x_direction + 0.5) - (0.5 * self.properties.width + self.properties.principal_offset_x)) / self.properties.focal_x
        y_direction: Tensor = ((y_direction + 0.5) - (0.5 * self.properties.height + self.properties.principal_offset_y)) / self.properties.focal_y
        z_direction: Tensor = torch.full((self.properties.height, self.properties.width), fill_value=-1)
        # transform directions into the camera's coordinate system
        directions_camera_space: Tensor = torch.stack(
            (x_direction, y_direction, z_direction), dim=-1
        ).reshape(-1, 3)
        # transform directions to world space
        directions_world_space: Tensor = torch.matmul(
            self.properties.c2w[:3, :3],
            directions_camera_space[:, :, None]
        ).squeeze()
        return directions_world_space

    def projectPoint(self, points: Tensor, return_z: bool = False) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
        """projects points (Nx3) to image plane. returns xy coordinates and mask of points that hit sensor"""
        points = torch.cat((points, torch.ones((points.shape[0], 1))), dim=1)
        points_cam_space = torch.matmul(self.properties.w2c, points[:, :, None])[..., 0]
        coords_x = ((points_cam_space[:, 0] * self.properties.focal_x / (-points_cam_space[:, 2] + 1e-8)) +
                    (self.properties.width * 0.5 + self.properties.principal_offset_x))
        coords_y = ((points_cam_space[:, 1] * self.properties.focal_y / (-points_cam_space[:, 2] + 1e-8)) +
                    (self.properties.height * 0.5 + self.properties.principal_offset_y))
        valid_mask = ((points_cam_space[:, 2] <= -self.near_plane) * (points_cam_space[:, 2] >= -self.far_plane) * (coords_x >= 0) *
                      (coords_x <= (self.properties.width - 1)) * (coords_y >= 0) * (coords_y <= (self.properties.height - 1)))
        if return_z:
            return torch.stack((coords_x, coords_y), dim=1), valid_mask, -points_cam_space[:, 2]
        return torch.stack((coords_x, coords_y), dim=1), valid_mask

    def getPerspectiveProjectionMatrix(self) -> Tensor:
        projection_matrix = torch.zeros((3, 4))
        projection_matrix[0, 0] = self.properties.focal_x
        projection_matrix[1, 1] = self.properties.focal_y
        projection_matrix[2, 2] = -1.0
        projection_matrix[0, 3] = 0.0
        projection_matrix[1, 3] = 0.0
        return projection_matrix
