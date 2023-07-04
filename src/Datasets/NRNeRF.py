# -- coding: utf-8 --

"""
Datasets/NRNeRF.py: Provides a dataset class for the NR-NeRF example sequence.
Data available at https://github.com/facebookresearch/nonrigid_nerf (last accessed 2023-05-25).
"""

import json

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import CameraProperties, CoordinateSystemTransformations
from Datasets.Base import BaseDataset
from Datasets.utils import list_sorted_files, loadImagesParallel


@Framework.Configurable.configure(
    PATH='dataset/nrnerf'
)
class CustomDataset(BaseDataset):
    """Dataset for the NR-NeRF example sequence."""

    def __init__(self, path: str) -> None:
        super().__init__(
            path,
            PerspectiveCamera(0.1, 1.0),
            CoordinateSystemTransformations.RIGHT_HAND
        )

    def load(self) -> dict[str, list[CameraProperties]]:
        """Loads the dataset into a dict containing lists of CameraProperties for training, evaluation, and testing."""
        data: dict[str, list[CameraProperties]] = {subset: [] for subset in self.subsets}
        scale_factor_intrinsics = self.IMAGE_SCALE_FACTOR if self.IMAGE_SCALE_FACTOR is not None else 1.0
        with open(self.dataset_path / 'precomputed.json', 'r') as json_file:
            precomputed = json.load(json_file)
        # load poses
        poses = torch.as_tensor(precomputed["poses"])
        render_poses = torch.as_tensor(precomputed["render_poses"])
        # load near/far plane
        bds = torch.as_tensor(precomputed["bds"])
        self.camera.near_plane = bds.min() * 0.9
        self.camera.far_plane = bds.max() * 1.0
        # load images
        images_path = self.dataset_path / 'images'
        image_filenames = [str(images_path / file) for file in list_sorted_files(images_path) if '.png' in file]
        rgbs, alphas = loadImagesParallel(image_filenames, self.IMAGE_SCALE_FACTOR, num_threads=4, desc='images')
        # create split CameraProperties objects
        for i, (rgb, alpha) in enumerate(zip(rgbs, alphas)):
            c2w = torch.cat((poses[i, :, :4], torch.tensor([[0.0, 0.0, 0.0, 1.0]])), dim=0)
            focal = poses[i, 2, 4] * scale_factor_intrinsics
            data['train'].append(CameraProperties(
                width=rgb.shape[2],
                height=rgb.shape[1],
                rgb=rgb,
                alpha=alpha,
                c2w=c2w,
                focal_x=focal,
                focal_y=focal,
                timestamp=(i / (len(rgbs) - 1))
            ))
        for i in range(render_poses.shape[0]):
            c2w = torch.cat((render_poses[i, :, :4], torch.tensor([[0.0, 0.0, 0.0, 1.0]])), dim=0)
            focal = render_poses[i, 2, 4] * scale_factor_intrinsics
            data['test'].append(CameraProperties(
                width=int(render_poses[i, 1, 4] * scale_factor_intrinsics),
                height=int(render_poses[i, 0, 4] * scale_factor_intrinsics),
                rgb=None,
                alpha=None,
                c2w=c2w,
                focal_x=focal,
                focal_y=focal,
                timestamp=(i / (render_poses.shape[0] - 1))
            ))
        # return the dataset
        return data
