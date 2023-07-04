# -- coding: utf-8 --

"""Datasets/Base.py: Basic dataset class features."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import torch

import Framework
from Cameras.Base import BaseCamera
from Cameras.utils import CameraProperties
from Datasets.utils import DatasetError
from Logger import Logger


@Framework.Configurable.configure(
    PATH='path/to/dataset/directory',
    IMAGE_SCALE_FACTOR=None,
    NORMALIZE_RADIUS=-1,
    NORMALIZE_RECENTER=False,
    PRECOMPUTE_RAYS=False,
    TO_DEVICE=False,
    BACKGROUND_COLOR=[1.0, 1.0, 1.0],
)
class BaseDataset(Framework.Configurable, ABC, torch.utils.data.Dataset):
    """Implements common functionalities of all datasets."""

    def __init__(self, path: str, camera: 'BaseCamera', cs_transform: Callable) -> None:
        Framework.Configurable.__init__(self, 'DATASET')
        ABC.__init__(self)
        torch.utils.data.Dataset.__init__(self)

        # check dataset path
        if path is None:
            raise DatasetError('Dataset path is "None"')
        self.dataset_path: Path = Path(path)
        Logger.log(f'loading dataset: {self.dataset_path}')

        # define subsets and load data
        self.subsets: tuple[str, str, str] = ('train', 'test', 'val')
        self.camera: 'BaseCamera' = camera
        self.camera.setBackgroundColor(*self.BACKGROUND_COLOR)
        self.mode: str = 'train'
        self.data: dict[str, list[CameraProperties]] = self.load()
        self.native_to_internal_coordinate_system_transform = cs_transform
        self.convertToInternalCoordinateSystem()
        self.normalizePoses('train', radius=self.NORMALIZE_RADIUS, recenter=self.NORMALIZE_RECENTER)
        self.ray_collection: dict[str, torch.Tensor | None] = {subset: None for subset in self.subsets}
        self.on_device = False
        if self.TO_DEVICE:
            self.toDefaultDevice(['train'])
        if self.PRECOMPUTE_RAYS:
            self.precomputeRays(['train'])

    def setMode(self, mode: str) -> 'BaseDataset':
        """Sets the dataset's mode to a given string."""
        self.mode = mode
        if self.mode not in self.subsets:
            raise DatasetError(f'requested invalid dataset mode: "{mode}"\n'
                               f'available option are: {self.subsets}')
        return self

    def train(self) -> 'BaseDataset':
        """Sets the dataset's mode to training."""
        return self.setMode('train')

    def test(self) -> 'BaseDataset':
        """Sets the dataset's mode to testing."""
        self.mode = 'test'
        return self.setMode('test')

    def eval(self) -> 'BaseDataset':
        """Sets the dataset's mode to validation."""
        return self.setMode('val')

    @abstractmethod
    def load(self) -> dict[str, list[CameraProperties] | None]:
        """Loads the dataset into a dict containing lists of CameraProperties for training, evaluation, and testing."""
        return {}

    def __len__(self) -> int:
        """Returns the size of the dataset depending on its current mode."""
        return len(self.data[self.mode])

    def __getitem__(self, index: int) -> CameraProperties:
        """Fetch specified item(s) from dataset."""
        element: CameraProperties = self.data[self.mode][index]
        return element.toDefaultDevice()

    def toDefaultDevice(self, subsets: list[str] | None = None) -> None:
        """Moves the specified dataset subsets to the default device."""
        if subsets is None:
            subsets = self.data.keys()
        for subset in subsets:
            self.data[subset] = [i.toDefaultDevice() for i in self.data[subset]]
        self.on_device = True

    def precomputeRays(self, subsets: list[str] | None = None) -> None:
        """Precomputes the rays for the specified dataset subsets."""
        if subsets is None:
            subsets = self.data.keys()
        for subset in subsets:
            self.setMode(subset)
            self.getAllRays()

    def getAllRays(self) -> torch.Tensor:
        """Returns all rays of the current dataset mode."""
        if self.ray_collection[self.mode] is None:
            # generate rays for all data points
            cp = self.camera.properties
            self.ray_collection[self.mode] = torch.cat([self.camera.setProperties(i).generateRays().cpu() for i in self], dim=0)
            if self.on_device:
                self.ray_collection[self.mode] = self.ray_collection[self.mode].type(Framework.config.GLOBAL.DEFAULT_TENSOR_TYPE)
            last_index = 0
            for properties in self.data[self.mode]:
                num_pixels = properties.width * properties.height
                if properties._precomputed_rays is None:
                    properties._precomputed_rays = self.ray_collection[self.mode][last_index:last_index + num_pixels]
                last_index += num_pixels
            self.camera.setProperties(cp)
        return self.ray_collection[self.mode]

    def normalizePoses(self, reference_set: str = None, radius: float = 1, recenter: bool = True) -> None:
        """Rescales the dataset camera poses inplace to sphere with given radius."""
        if radius > 0:
            Logger.logInfo('normalizing dataset')
            data_all = []
            if reference_set is None:
                for data_set in self.data.values():
                    data_all += data_set
            else:
                data_all += self.data[reference_set]
            positions_all: torch.Tensor = torch.stack([
                camera_properties.c2w[:3, -1] for camera_properties in data_all if camera_properties is not None
            ], dim=0)
            # mean_position: torch.Tensor = positions_all.mean(dim=0, keepdim=True)
            mean_position: torch.Tensor = (
                    (positions_all.max(dim=0, keepdim=True).values + positions_all.min(dim=0, keepdim=True).values) * 0.5
            ) if recenter else torch.zeros((1, 3), dtype=torch.float32)
            max_dist: torch.Tensor = torch.linalg.norm(positions_all - mean_position, dim=-1, ord=2).max() / radius
            for data_set in self.data.values():
                for camera_properties in data_set:
                    if camera_properties is not None:
                        camera_properties.c2w[:3, -1] = (camera_properties.c2w[:3, -1] - mean_position) / max_dist
                        camera_properties.w2c = torch.linalg.inv(camera_properties.c2w)
                        if camera_properties.depth is not None:
                            camera_properties.depth /= max_dist.item()
            self.camera.near_plane /= max_dist.item()
            self.camera.far_plane /= max_dist.item()

    def convertToInternalCoordinateSystem(self) -> None:
        """Converts the dataset camera matrices to the framework's internal representation."""
        for data_set in self.data.values():
            for camera_properties in data_set:
                if camera_properties is not None:
                    if camera_properties.c2w is None and camera_properties.w2c is not None:
                        camera_properties.c2w = torch.linalg.inv(camera_properties.w2c)
                    if camera_properties.c2w is not None:
                        native_x, native_y, native_z = camera_properties.c2w[:3, :3].t()
                        internal_x, internal_y, internal_z = self.native_to_internal_coordinate_system_transform(native_x, native_y, native_z)
                        camera_properties.c2w[:3, 0] = internal_x
                        camera_properties.c2w[:3, 1] = internal_y
                        camera_properties.c2w[:3, 2] = internal_z
                        camera_properties.w2c = torch.linalg.inv(camera_properties.c2w)
