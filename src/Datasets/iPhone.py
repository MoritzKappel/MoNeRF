# -- coding: utf-8 --

"""
Datasets/iPhone.py: Provides a dataset class for DyCheck iPhone scenes.
Data available at https://github.com/KAIR-BAIR/dycheck/blob/main/docs/DATASETS.md (last accessed 2023-05-25).
"""

import json
from pathlib import Path
from typing import Any

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import CameraProperties, CoordinateSystemTransformations
from Datasets.Base import BaseDataset
from Datasets.utils import DatasetError, loadImagesParallel


@Framework.Configurable.configure(
    PATH='dataset/iphone/paper-windmill',
    BACKGROUND_COLOR=[0.0, 0.0, 0.0]
)
class CustomDataset(BaseDataset):
    """Dataset class forDyCheck iPhone scenes."""

    def __init__(self, path: str) -> None:
        super().__init__(
            path,
            PerspectiveCamera(0.01, 2.0),
            CoordinateSystemTransformations.LEFT_HAND
        )

    def load(self) -> dict[str, list[CameraProperties]]:
        """Loads the dataset into a dict containing lists of CameraProperties for training and testing."""
        data: dict[str, list[CameraProperties]] = {subset: [] for subset in self.subsets}
        # load scene info
        scene_info_filepath: Path = self.dataset_path / 'scene.json'
        try:
            with open(scene_info_filepath, 'r') as f:
                scene_info_dict: dict[str, Any] = json.load(f)
        except IOError:
            raise DatasetError(f'Invalid scene info file path "{scene_info_filepath}"')
        center = torch.as_tensor(scene_info_dict['center'], dtype=torch.float32)
        scale = scene_info_dict['scale']
        self.camera.near_plane = scene_info_dict['near']
        self.camera.far_plane = scene_info_dict['far']

        # load dataset info
        dataset_info_filepath: Path = self.dataset_path / 'dataset.json'
        try:
            with open(dataset_info_filepath, 'r') as f:
                dataset_info_dict: dict[str, Any] = json.load(f)
        except IOError:
            raise DatasetError(f'Invalid dataset info file path "{dataset_info_filepath}"')
        max_time_id = dataset_info_dict['num_exemplars'] - 1

        # load extra info
        extra_info_filepath: Path = self.dataset_path / 'extra.json'
        try:
            with open(extra_info_filepath, 'r') as f:
                extra_info_dict: dict[str, Any] = json.load(f)
        except IOError:
            raise DatasetError(f'Invalid extra info file path "{extra_info_filepath}"')
        factor = extra_info_dict['factor']
        # fps = extra_info_dict['fps']
        # bbox = torch.as_tensor((extra_info_dict['bbox'], dtype=torch.float32)
        # lookat = torch.as_tensor(extra_info_dict['lookat'], dtype=torch.float32)
        # up = torch.as_tensor(extra_info_dict['up'], dtype=torch.float32)

        image_directory_path = self.dataset_path / 'rgb' / f'{factor}x'
        for split in ['train', 'val']:
            # load split info
            split_info_filepath: Path = self.dataset_path / 'splits' / f'{split}.json'
            try:
                with open(split_info_filepath, 'r') as f:
                    split_info_dict: dict[str, Any] = json.load(f)
            except IOError:
                raise DatasetError(f'Invalid {split} split info file path "{split_info_filepath}"')
            # load the authors' val split into the test set of our dataset
            split = split if split == 'train' else 'test'
            # load images
            image_filenames = [str(image_directory_path / f'{frame}.png') for frame in split_info_dict['frame_names']]
            rgbs, _ = loadImagesParallel(image_filenames, self.IMAGE_SCALE_FACTOR, num_threads=4, desc=split)
            # create split CameraProperties objects
            for rgb, frame_name, time_id in zip(rgbs, split_info_dict['frame_names'], split_info_dict['time_ids']):
                # load camera info
                camera_info_filepath: Path = self.dataset_path / 'camera' / f'{frame_name}.json'
                try:
                    with open(camera_info_filepath, 'r') as f:
                        camera_info_dict: dict[str, Any] = json.load(f)
                except IOError:
                    raise DatasetError(f'Invalid camera info file path "{camera_info_filepath}"')
                # make sure there is no skew or distortion
                if camera_info_dict['skew'] != 0.0:
                    raise DatasetError('Camera axis skew not supported')
                if torch.count_nonzero(torch.as_tensor(camera_info_dict['radial_distortion'])).item() > 0:
                    raise DatasetError('Radial camera distortion not supported')
                if torch.count_nonzero(torch.as_tensor(camera_info_dict['tangential_distortion'])).item() > 0:
                    raise DatasetError('Tangential camera distortion not supported')
                # focal length
                focal_length = camera_info_dict['focal_length'] / factor
                pixel_aspect_ratio = camera_info_dict['pixel_aspect_ratio']
                focal_x = focal_length
                focal_y = focal_length * pixel_aspect_ratio
                # principal point
                principal_point_x, principal_point_y = torch.as_tensor(camera_info_dict['principal_point'], dtype=torch.float32) / factor
                # adjust intrinsics when images are resized
                if self.IMAGE_SCALE_FACTOR is not None:
                    focal_x *= self.IMAGE_SCALE_FACTOR
                    focal_y *= self.IMAGE_SCALE_FACTOR
                    principal_point_x *= self.IMAGE_SCALE_FACTOR
                    principal_point_y *= self.IMAGE_SCALE_FACTOR
                # c2w matrix
                rotation = torch.linalg.inv(torch.as_tensor(camera_info_dict['orientation'], dtype=torch.float32))
                translation = (torch.as_tensor(camera_info_dict['position'], dtype=torch.float32) - center) * scale
                c2w = torch.cat([
                    torch.cat([rotation, translation[..., None]], dim=-1),
                    torch.tensor([[0, 0, 0, 1]], dtype=torch.float32),
                ], dim=-2)

                data[split].append(CameraProperties(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    rgb=rgb,
                    alpha=None,
                    c2w=c2w,
                    focal_x=focal_x,
                    focal_y=focal_y,
                    principal_offset_x=principal_point_x - (rgb.shape[2] * 0.5),
                    principal_offset_y=principal_point_y - (rgb.shape[1] * 0.5),
                    timestamp=time_id / max_time_id
                ))
        if not data['test']:
            for i in data['train']:
                data['test'].append(CameraProperties(
                    width=data['train'][0].width,
                    height=data['train'][0].height,
                    rgb=data['train'][0].rgb.copy(),
                    alpha=None,
                    c2w=data['train'][0].c2w.copy(),
                    focal_x=data['train'][0].focal_x,
                    focal_y=data['train'][0].focal_y,
                    principal_offset_x=data['train'][0].principal_offset_x,
                    principal_offset_y=data['train'][0].principal_offset_y,
                    timestamp=i.timestamp
                ))
        return data
