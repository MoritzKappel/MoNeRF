# -- coding: utf-8 --

"""
Datasets/MMVA.py: Provides a dataset class for Monocularized Multi-View Avatars (MMVA) scenes.
Data available at https://nextcloud.mpi-klsb.mpg.de/index.php/s/EHtctQJZDWWcfqj.
"""

import json
from pathlib import Path
from typing import Any

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import CameraProperties, CoordinateSystemTransformations
from Datasets.Base import BaseDataset
from Datasets.utils import applyBGColor, DatasetError, loadImagesParallel


@Framework.Configurable.configure(
    PATH='dataset/mmva/Squat',
)
class CustomDataset(BaseDataset):
    """Dataset class for Monocularized Multi-View Avatars (MMVA) scenes."""

    def __init__(self, path: str) -> None:
        super().__init__(
            path,
            PerspectiveCamera(0.01, 1.0),
            CoordinateSystemTransformations.RIGHT_HAND
        )

    def load(self) -> dict[str, list[CameraProperties]]:
        """Loads the dataset into a dict containing lists of CameraProperties for training, evaluation, and testing."""
        data: dict[str, list[CameraProperties]] = {subset: [] for subset in self.subsets}
        scale_factor_intrinsics = 1.0 if self.IMAGE_SCALE_FACTOR is None else self.IMAGE_SCALE_FACTOR
        for subset in self.subsets:
            metadata_filepath: Path = self.dataset_path / f'transforms_{subset}.json'
            try:
                with open(metadata_filepath, 'r') as f:
                    metadata_file: dict[str, Any] = json.load(f)
            except IOError:
                raise DatasetError(f'Invalid dataset metadata file path "{metadata_filepath}"')
            # load near/far plane
            if subset == 'train':
                self.camera.near_plane = metadata_file['near_all']
                self.camera.far_plane = metadata_file['far_all']
            # load images
            image_filenames = [str(self.dataset_path / frame['file_path']) for frame in metadata_file['frames']]
            rgbs, alphas = loadImagesParallel(image_filenames, self.IMAGE_SCALE_FACTOR, num_threads=4, desc=subset)
            # create split CameraProperties objects
            for frame, rgb, alpha in zip(metadata_file['frames'], rgbs, alphas):
                # apply background color where alpha < 1
                rgb = applyBGColor(rgb, alpha, self.camera.background_color)
                # load camera extrinsics
                c2w = torch.as_tensor(frame['transform_matrix'], dtype=torch.float32)
                # insert loaded values
                data[subset].append(CameraProperties(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    rgb=rgb,
                    alpha=alpha,
                    c2w=c2w,
                    focal_x=frame['focal_x'] * scale_factor_intrinsics,
                    focal_y=frame['focal_y'] * scale_factor_intrinsics,
                    principal_offset_x=frame['principal_x'] * scale_factor_intrinsics,
                    principal_offset_y=frame['principal_y'] * scale_factor_intrinsics,
                    timestamp=frame['time']
                ))
        return data
